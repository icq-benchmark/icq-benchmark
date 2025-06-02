# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from transformers import AutoProcessor, AutoModelForPreTraining, EarlyStoppingCallback
from packaging import version
from torch.utils.data import Dataset, DataLoader
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.conversation import conv_templates
import uuid
import numpy as np
import random
from datasets import Dataset as HFDataset
from torch.utils.data import Subset
from datasets import DatasetDict, interleave_datasets
from transformers import set_seed
from PIL import Image
from llava.mm_utils import tokenizer_image_token
from llava.model import *
from llava import conversation as conversation_lib
from llava.train.llava_trainer import LLaVATrainer
from torch.utils.data import Dataset
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
import tokenizers
import transformers
from torchvision import transforms
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
torch.cuda.empty_cache()


logging.basicConfig(level=logging.INFO)

seed_value = 60
# 8 60 99 119
set_seed(seed_value)
logging.info(f"Random seed set to: {seed_value}")

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(
    tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(
        default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    coco_train_data_path: str = field(default=None,
                                      metadata={"help": "Path to the training data."})
    flickr30k_data_path: str = field(default=None,
                                     metadata={"help": "Path to the training data."})

    split_ratio: float = field(
        default=0.95,
        metadata={
            "help": "The ratio to split the dataset into training and validation sets."}
    )
    coco_train_img_dir_path: Optional[str] = field(default=None)
    coco_val_img_dir_path: Optional[str] = field(default=None)
    flickr30k_image_dir_path: Optional[str] = field(default=None)

    dataset_seed: Optional[int] = field(default=42)

    subset_size: Optional[int] = field(default=None)

    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    instruct_type: Optional[str] = field(default="icq_suit")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    early_stopping_patience: int = field(
        default=500, metadata={"help": "Patience for early stopping."})
    early_stopping_threshold: float = field(
        default=3.0, metadata={"help": "Threshold for early stopping."})
    early_stopping_metric: str = field(default="eval_loss", metadata={
                                       "help": "Metric to monitor for early stopping."})
    load_best_model_at_end: bool = field(default=True, metadata={
                                         "help": "Whether to load the best model at the end of training."})  # Add this

    lora_target_modules: str = field(
        default="gate_up_down",
        metadata={"help": "target modules for LoRA"}
    )


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k,
                     t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True)
                 for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(
        key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu()
                 for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def find_part_linear_names(model, from_layer=16):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        print(f"name: {name}")
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls) and any(str(i) in name for i in range(from_layer, model.config.num_hidden_layers)):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')

    print(f"part linear names: {lora_module_names}")
    return list(lora_module_names)


def find_target_module_names(model, target_module):
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if any(tm in name for tm in target_module):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')

    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(
                    parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(
                    mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(
                    output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(
                    DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + \
                    '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(
                        DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    index_1 = None
    index_2 = None
    index_3 = None
    index_4 = None

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):

        if not source:
            logging.warning(
                f"Source at index {i} is empty, skipping this source.")
            logging.info(f"conv: {conv}, \nroles: {roles}")
            index_1 = i
            continue

        if "from" not in source[0]:
            logging.warning(
                f"Source at index {i} missing 'from' field: {source[0]}")
            index_2 = i
            continue

        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            index_3 = i
            logging.info(
                f"Skipping the first entry in source at index {i} as it is not from 'human'. \nSource: {source}\nconv.roles[0]: {conv.roles[0]}")
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            # role = roles[sentence["from"]]

            role = roles.get(sentence.get("from"), None)
            if role is None:
                index_4 = i
                logging.warning(
                    f"Skipping message in source at index {i} due to missing role: {sentence}")
                continue

            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if index_1 is not None or index_2 is not None or index_3 is not None or index_4 is not None:
        logging.info(f"Final conversations: {conversations}")
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    if index_1 is not None or index_2 is not None or index_3 is not None or index_4 is not None:
        logging.info(f"Shape of input_ids: {input_ids.shape}")
        logging.info(f"Shape of targets: {targets.shape}")

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(
                    tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(
                    tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(
                rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(
                    tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + \
            conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(
        prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(
            source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations

    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(
            prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len(
                [header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn(
                [header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split())
                               for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split())
                          for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(
            sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(
                image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255)
                                      for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')[
                    'pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')[
                    'pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(
                3, crop_size['height'], crop_size['width'])
        return data_dict


class CaptionDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, hf_dataset: HFDataset,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # idx = idx % len(self.hf_dataset)
        num_trials = 10
        trial_count = 0
        max_idx = len(self.hf_dataset) - 1

        while trial_count < num_trials:
            if idx > max_idx:
                logging.warning(
                    f"Index {idx} is out of bounds, resetting to 0.")
                idx = 0

            try:
                sample = self.hf_dataset[idx]

                if sample is None or not sample.get("refinements") or not sample.get("negative_caption"):
                    logging.warning(
                        f"Sample at index {idx} is empty or missing required fields. Trying next index.")
                    idx += 1
                    trial_count += 1
                    continue

                conversations = []
                for semantic_type, refinements in sample["refinements"].items():
                    if not refinements:
                        continue
                    prompt = self.generate_prompt(semantic_type, refinements)
                    target_caption = sample["negative_caption"].get(
                        semantic_type, "No caption available")
                    if idx % 3000 == 0:
                        logging.info(
                            f"Generated prompt for sample {idx}: {prompt}")
                        logging.info(
                            f"semantic type {semantic_type}, \ntarget_caption {target_caption}")

                    conversations.append(
                        {"from": "human", "value": f"<image>\n{prompt}"})
                    conversations.append(
                        {"from": "gpt", "value": target_caption})

                if "image" in sample:
                    if idx % 10000 == 0:
                        logging.info(f"image in sample.")
                    image_file = sample['image']
                    if sample['dataset_name'] == "coco_train":
                        image_folder = self.data_args.coco_train_img_dir_path
                    elif sample['dataset_name'] == "coco_val":
                        image_folder = self.data_args.coco_val_img_dir_path
                    elif sample['dataset_name'] == "flickr30k":
                        image_folder = self.data_args.flickr30k_image_dir_path
                    else:
                        image_folder = ""
                    image_path = os.path.join(image_folder, image_file)
                    processor = self.data_args.image_processor
                    image = Image.open(image_path).convert('RGB')
                    if self.data_args.image_aspect_ratio == 'pad':
                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(
                                    pil_img.mode, (width, width), background_color)
                                result.paste(
                                    pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(
                                    pil_img.mode, (height, height), background_color)
                                result.paste(
                                    pil_img, ((height - width) // 2, 0))
                                return result
                        image = expand2square(image, tuple(
                            int(x*255) for x in processor.image_mean))
                        image = processor.preprocess(image, return_tensors='pt')[
                            'pixel_values'][0]
                    else:
                        image = processor.preprocess(image, return_tensors='pt')[
                            'pixel_values'][0]

                    if idx % 8000 == 0:
                        logging.info(
                            f"Processing multimodal data for sample {idx}")
                    sources = preprocess_multimodal(
                        [conversations], self.data_args)

                else:
                    logging.info(f"Processing text-only data for sample {idx}")
                    sources = [conversations]

                if idx % 4000 == 0:
                    logging.info(f"sources: {sources}")
                    # INFO:root:sources:
                    # [[{'from': 'human',
                    # 'value': "<image>\nDescribe the given image while replacing 'man' with 'woman'; and replacing 'man' with 'woman'."},
                    # {'from': 'gpt',
                    # 'value': 'A woman and a boy are playing tennis on a court, with the woman holding a tennis racket and the boy holding a basket of tennis balls.'}]]

                data_dict = preprocess(
                    sources,
                    self.tokenizer,
                    has_image=True  # ("image" in sample)
                )

                if idx % 8000 == 0:
                    logging.info(f"Data dict: {data_dict}")

                data = {
                    "input_ids": data_dict["input_ids"][0],
                    "labels": data_dict["labels"][0]
                }
                if "image" in sample:
                    data['image'] = image
                elif self.data_args.is_multimodal:
                    crop_size = self.data_args.image_processor.crop_size
                    data['image'] = torch.zeros(
                        3, crop_size['height'], crop_size['width'])

                if idx % 10000 == 0:
                    logging.info(f"Data: {data}")

                return data

            except Exception as e:
                logging.error(
                    f"An error occurred at index {idx}. Trial count: {trial_count}")
                logging.error(
                    f"Sample data: {sample if 'sample' in locals() else 'N/A'}")
                logging.error(
                    f"Conversations: {conversations if 'conversations' in locals() else 'N/A'}")
                logging.error(
                    f"Sources: {sources if 'sources' in locals() else 'N/A'}")
                logging.error(f"Exception message: {str(e)}")

                idx += 1
                trial_count += 1

        raise ValueError(
            "All trials exhausted. Could not find valid data after 10 attempts.")

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            conversation_lengths = 0
            if "negative_caption" in sample:
                for conv in sample["negative_caption"].values():
                    conversation_lengths += len(conv.split())
            length_list.append(conversation_lengths + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = 0
            if "negative_caption" in sample:
                for conv in sample["negative_caption"].values():
                    cur_len += len(conv.split())
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def text_template(self):
        if self.data_args.instruct_type == "icq_suit":
            PROMPT_TEMPLATES = [
                "I have an image. Adjust the {semantic_type} by {refinement_type}. The revised caption should remain coherent and logical without introducing any additional details",
                "Look at the image! Change the {semantic_type} by {refinement_type}. Then, write a new caption that fits and doesn't add new stuff. Only give the caption, no extra words",
                "Here's an image. Can you change {semantic_type} by {refinement_type}? After that, make a new caption that makes sense and doesn't add anything extra. Just write the caption, no explanations needed.",
            ]

        REFINEMENT_TEMPLATES_LIST = [
            {
                "change": "changing '{old_word}' to '{new_word}'",
                "add": "adding '{added_word}'",
                "remove": "removing '{removed_word}'"
            },
            {
                "change": "modifying '{old_word}' to '{new_word}'",
                "add": "inserting '{added_word}'",
                "remove": "deleting '{removed_word}'"
            },
            {
                "change": "replacing '{old_word}' with '{new_word}'",
                "add": "including '{added_word}'",
                "remove": "dropping '{removed_word}'"
            }
        ]
        return PROMPT_TEMPLATES, REFINEMENT_TEMPLATES_LIST

    def generate_prompt(self, semantic_type, refinements):
        PROMPT_TEMPLATES, REFINEMENT_TEMPLATES_LIST = self.text_template()
        REFINEMENT_TEMPLATES = random.choice(REFINEMENT_TEMPLATES_LIST)

        refinement_types = []
        for ref in refinements:
            if "change" in ref:
                parts = ref.replace("change ", "").replace(
                    "\"", "").split(" to ", 1)
                if len(parts) == 2:
                    old_word, new_word = parts
                    ref_type = REFINEMENT_TEMPLATES["change"].format(
                        old_word=old_word, new_word=new_word)
                else:
                    ref_type = ""
                    continue
            elif "add" in ref:
                added_word = ref.replace("add ", "").replace("\"", "")
                ref_type = REFINEMENT_TEMPLATES["add"].format(
                    added_word=added_word)
            elif "remove" in ref:
                removed_word = ref.replace("remove ", "").replace("\"", "")
                ref_type = REFINEMENT_TEMPLATES["remove"].format(
                    removed_word=removed_word)
            else:
                continue
            refinement_types.append(ref_type)

        ref_text = "; and ".join(refinement_types)
        template = random.choice(PROMPT_TEMPLATES)
        return template.format(semantic_type=semantic_type, refinement_type=ref_text)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """
    Collate examples for supervised fine-tuning.
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


@dataclass
class DataCollatorForEvaluationDataset(object):
    """
    Collate examples for supervised fine-tuning.
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __init__(self, tokenizer, image_transform=None):
        self.tokenizer = tokenizer
        self.image_transform = image_transform or transforms.ToTensor()

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            images = [self.image_transform(img) if isinstance(
                img, Image.Image) else img for img in images]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


class EvalDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, hf_dataset: HFDataset,
                 tokenizer: transformers.PreTrainedTokenizer,
                 model_config,
                 data_args: DataArguments):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.data_args = data_args

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        # idx = idx % len(self.hf_dataset)
        num_trials = 10
        trial_count = 0
        max_idx = len(self.hf_dataset) - 1

        while trial_count < num_trials:
            if idx > max_idx:
                logging.warning(
                    f"Index {idx} is out of bounds, resetting to 0.")
                idx = 0

            try:
                sample = self.hf_dataset[idx]
                refinements = sample["refinements"]
                negative_caption = sample["negative_caption"]
                target_caption = negative_caption

                if sample is None or not sample.get("refinements") or not sample.get("negative_caption"):
                    logging.warning(
                        f"Sample at index {idx} is empty or missing required fields. Trying next index.")
                    idx += 1
                    trial_count += 1
                    continue

                for semantic_type, refinements in sample["refinements"].items():
                    if not refinements:
                        continue
                    prompt = self.generate_prompt(semantic_type, refinements)
                    target_caption = sample["negative_caption"].get(
                        semantic_type, "No caption available")

                if "image" in sample:
                    image_file = sample['image']
                    if sample['dataset_name'] == "coco_train":
                        image_folder = self.data_args.coco_train_img_dir_path
                    elif sample['dataset_name'] == "coco_val":
                        image_folder = self.data_args.coco_val_img_dir_path
                    elif sample['dataset_name'] == "flickr30k":
                        image_folder = self.data_args.flickr30k_image_dir_path
                    else:
                        image_folder = ""
                    image_path = os.path.join(image_folder, image_file)
                    image = Image.open(image_path).convert('RGB')

                image_tensor = process_images(
                    [image], self.data_args.image_processor, self.model_config)[0]
                # image_tensor = self.data_args.image_processor(images=image, return_tensors='pt')['pixel_values'][0]
                input_ids = tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

                # return input_ids, image_tensor, image.size

                return {
                    "input_ids": input_ids,
                    "image_tensor": image_tensor,
                    "image_size": image.size,
                    "image": image_file,
                    'target_caption': target_caption,
                    "refinements": refinements,
                }

            except Exception as e:
                logging.error(
                    f"An error occurred at index {idx}. Trial count: {trial_count}")
                logging.error(
                    f"Sample data: {sample if 'sample' in locals() else 'N/A'}")
                logging.error(f"Exception message: {str(e)}")

                idx += 1
                trial_count += 1

        raise ValueError(
            "All trials exhausted. Could not find valid data after 10 attempts.")

    def text_template(self):
        if self.data_args.instruct_type == "icq_suit":
            PROMPT_TEMPLATES = [
                "I have an image. Adjust the {semantic_type} by {refinement_type}. The revised caption should remain coherent and logical without introducing any additional details",
                "Look at the image! Change the {semantic_type} by {refinement_type}. Then, write a new caption that fits and doesn't add new stuff. Only give the caption, no extra words",
                "Here's an image. Can you change {semantic_type} by {refinement_type}? After that, make a new caption that makes sense and doesn't add anything extra. Just write the caption, no explanations needed.",
            ]

        REFINEMENT_TEMPLATES_LIST = [
            {
                "change": "changing '{old_word}' to '{new_word}'",
                "add": "adding '{added_word}'",
                "remove": "removing '{removed_word}'"
            },
            {
                "change": "modifying '{old_word}' to '{new_word}'",
                "add": "inserting '{added_word}'",
                "remove": "deleting '{removed_word}'"
            },
            {
                "change": "replacing '{old_word}' with '{new_word}'",
                "add": "including '{added_word}'",
                "remove": "dropping '{removed_word}'"
            }
        ]
        return PROMPT_TEMPLATES, REFINEMENT_TEMPLATES_LIST

    def generate_prompt(self, semantic_type, refinements):
        PROMPT_TEMPLATES, REFINEMENT_TEMPLATES_LIST = self.text_template()
        REFINEMENT_TEMPLATES = random.choice(REFINEMENT_TEMPLATES_LIST)

        refinement_types = []
        for ref in refinements:
            if "change" in ref:
                parts = ref.replace("change ", "").replace(
                    "\"", "").split(" to ", 1)
                if len(parts) == 2:
                    old_word, new_word = parts
                    ref_type = REFINEMENT_TEMPLATES["change"].format(
                        old_word=old_word, new_word=new_word)
                else:
                    ref_type = ""
                    continue
            elif "add" in ref:
                added_word = ref.replace("add ", "").replace("\"", "")
                ref_type = REFINEMENT_TEMPLATES["add"].format(
                    added_word=added_word)
            elif "remove" in ref:
                removed_word = ref.replace("remove ", "").replace("\"", "")
                ref_type = REFINEMENT_TEMPLATES["remove"].format(
                    removed_word=removed_word)
            else:
                continue
            refinement_types.append(ref_type)

        ref_text = "; and ".join(refinement_types)
        template = random.choice(PROMPT_TEMPLATES)
        return template.format(semantic_type=semantic_type, refinement_type=ref_text)


# def collate_fn(batch):
#     input_ids, image_tensors, image_sizes = zip(*batch)
#     input_ids = torch.stack(input_ids, dim=0)
#     image_tensors = torch.stack(image_tensors, dim=0)
#     return input_ids, image_tensors, image_sizes

def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    image_tensors = torch.stack([item["image_tensor"]
                                for item in batch], dim=0)
    image_sizes = [item["image_size"]
                   for item in batch]  # Keep as list if sizes vary
    # File names do not need further processing
    image_files = [item["image"] for item in batch]
    target_captions = [item["target_caption"]
                       for item in batch]  # Keep as list of strings
    refinements = [item["refinements"]
                   for item in batch]  # Keep as list of dictionaries

    return {
        "input_ids": input_ids,
        "image_tensors": image_tensors,
        "image_sizes": image_sizes,
        "image_files": image_files,
        "target_captions": target_captions,
        "refinements": refinements
    }


def get_eval_data_loader(tokenizer: transformers.PreTrainedTokenizer, model_config, data_args) -> Dict:
    from datasets import Dataset, Value, Sequence

    dataset_seed = data_args.dataset_seed
    split_ratio = data_args.split_ratio
    coco_train_data_path = data_args.coco_train_data_path
    flickr30k_data_path = data_args.flickr30k_data_path
    subset_size = data_args.subset_size

    with open(coco_train_data_path, 'r') as f:
        coco_data = json.load(f)["_default"]
    coco_data_list = [v for v in coco_data.values()]
    for item in coco_data_list:
        image_id = item['image_id']
        item['id'] = image_id
        item['dataset_name'] = 'coco_train'
        item['image'] = f"{int(str(image_id)):012d}.jpg"
        item['subset_id'] = str(item.get('subset_id', ''))
        item['image_id'] = int(image_id)

    coco_hf_dataset = Dataset.from_list(coco_data_list)
    coco_hf_dataset = coco_hf_dataset.remove_columns('original_captions')
    logging.info(f"COCO dataset length before split: {len(coco_hf_dataset)}")

    with open(flickr30k_data_path, 'r') as f:
        flickr30k_data = json.load(f)["_default"]
    flickr30k_data_list = [v for v in flickr30k_data.values()]
    for item in flickr30k_data_list:
        image_id = item['image_id']
        image_id_numeric = int(image_id.replace(".jpg", ""))
        item['id'] = image_id_numeric
        item['dataset_name'] = 'flickr30k'
        item['image'] = image_id
        item['subset_id'] = str(item.get('subset_id', ''))
        item['image_id'] = image_id_numeric

    flickr30k_hf_dataset = Dataset.from_list(flickr30k_data_list)
    logging.info(
        f"Flickr30k dataset length before split: {len(flickr30k_hf_dataset)}")

    coco_split_data = coco_hf_dataset.train_test_split(
        test_size=1 - split_ratio, seed=dataset_seed)
    flickr30k_split_data = flickr30k_hf_dataset.train_test_split(
        test_size=1 - split_ratio, seed=dataset_seed)

    logging.info(
        f"COCO train length: {len(coco_split_data['train'])}, COCO test length: {len(coco_split_data['test'])}")
    logging.info(
        f"Flickr30k train length: {len(flickr30k_split_data['train'])}, Flickr30k test length: {len(flickr30k_split_data['test'])}")

    train_datasets = [coco_split_data['train'], flickr30k_split_data['train']]
    eval_datasets = [coco_split_data['test'], flickr30k_split_data['test']]

    total_train_length = sum(len(d) for d in train_datasets)
    train_probabilities = [len(d) / total_train_length for d in train_datasets]

    # train_data = interleave_datasets(train_datasets, seed=dataset_seed, probabilities=train_probabilities, stopping_strategy='all_exhausted')
    eval_data = interleave_datasets(eval_datasets, seed=dataset_seed,
                                    probabilities=train_probabilities, stopping_strategy='all_exhausted')

    # logging.info(f"Length of training interleave_datasets with all_exhausted: {len(train_data)}")
    logging.info(
        f"Length of evaluation interleave_datasets with all_exhausted: {len(eval_data)}")

    eval_data = eval_data.select(range(100))
    eval_dataset = EvalDataset(eval_data, tokenizer, model_config, data_args)

    eval_data_loader = DataLoader(
        eval_dataset, batch_size=1, num_workers=4, shuffle=False, collate_fn=collate_fn)

    logging.info(f"Length of final evaluation dataset: {len(eval_dataset)}")

    return eval_data_loader


def make_eval_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    from datasets import Dataset, Value, Sequence

    from torch.utils.data import Subset

    dataset_seed = data_args.dataset_seed
    split_ratio = data_args.split_ratio
    coco_train_data_path = data_args.coco_train_data_path
    flickr30k_data_path = data_args.flickr30k_data_path

    with open(coco_train_data_path, 'r') as f:
        coco_data = json.load(f)["_default"]
    coco_data_list = [v for v in coco_data.values()]
    for item in coco_data_list:
        image_id = item['image_id']
        item['id'] = image_id
        item['dataset_name'] = 'coco_train'
        item['image'] = f"{int(str(image_id)):012d}.jpg"
        item['subset_id'] = str(item.get('subset_id', ''))
        item['image_id'] = int(image_id)

    coco_hf_dataset = Dataset.from_list(coco_data_list)
    coco_hf_dataset = coco_hf_dataset.remove_columns('original_captions')

    with open(flickr30k_data_path, 'r') as f:
        flickr30k_data = json.load(f)["_default"]
    flickr30k_data_list = [v for v in flickr30k_data.values()]
    for item in flickr30k_data_list:
        image_id = item['image_id']
        image_id_numeric = int(image_id.replace(".jpg", ""))
        item['id'] = image_id_numeric
        item['dataset_name'] = 'flickr30k'
        item['image'] = image_id
        item['subset_id'] = str(item.get('subset_id', ''))
        item['image_id'] = image_id_numeric

    flickr30k_hf_dataset = Dataset.from_list(flickr30k_data_list)

    coco_split_data = coco_hf_dataset.train_test_split(
        test_size=1 - split_ratio, seed=dataset_seed)
    flickr30k_split_data = flickr30k_hf_dataset.train_test_split(
        test_size=1 - split_ratio, seed=dataset_seed)

    train_datasets = [coco_split_data['train'], flickr30k_split_data['train']]
    eval_datasets = [coco_split_data['test'], flickr30k_split_data['test']]

    total_train_length = sum(len(d) for d in train_datasets)
    train_probabilities = [len(d) / total_train_length for d in train_datasets]

    train_data = interleave_datasets(train_datasets, seed=dataset_seed,
                                     probabilities=train_probabilities, stopping_strategy='all_exhausted')
    eval_data = interleave_datasets(eval_datasets, seed=dataset_seed,
                                    probabilities=train_probabilities, stopping_strategy='all_exhausted')

    train_dataset = CaptionDataset(train_data, tokenizer, data_args)
    eval_dataset = CaptionDataset(eval_data, tokenizer, data_args)

    # subset_indices = list(range(20))
    # train_subset = Subset(train_dataset, subset_indices)

    logging.info(f"Length of final training dataset: {len(train_dataset)}")
    logging.info(f"Length of final evaluation dataset: {len(eval_dataset)}")
    # logging.info(f"Length of final subset: {len(train_subset)}")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )


def make_custom_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    from datasets import Dataset, Value, Sequence

    dataset_seed = data_args.dataset_seed
    split_ratio = data_args.split_ratio
    coco_train_data_path = data_args.coco_train_data_path
    flickr30k_data_path = data_args.flickr30k_data_path
    subset_size = data_args.subset_size

    with open(coco_train_data_path, 'r') as f:
        coco_data = json.load(f)["_default"]
    coco_data_list = [v for v in coco_data.values()]
    for item in coco_data_list:
        image_id = item['image_id']
        item['id'] = image_id
        item['dataset_name'] = 'coco_train'
        item['image'] = f"{int(str(image_id)):012d}.jpg"
        item['subset_id'] = str(item.get('subset_id', ''))
        item['image_id'] = int(image_id)

    coco_hf_dataset = Dataset.from_list(coco_data_list)
    coco_hf_dataset = coco_hf_dataset.remove_columns('original_captions')
    logging.info(f"COCO dataset length before split: {len(coco_hf_dataset)}")

    with open(flickr30k_data_path, 'r') as f:
        flickr30k_data = json.load(f)["_default"]
    flickr30k_data_list = [v for v in flickr30k_data.values()]
    for item in flickr30k_data_list:
        image_id = item['image_id']
        image_id_numeric = int(image_id.replace(".jpg", ""))
        item['id'] = image_id_numeric
        item['dataset_name'] = 'flickr30k'
        item['image'] = image_id
        item['subset_id'] = str(item.get('subset_id', ''))
        item['image_id'] = image_id_numeric

    flickr30k_hf_dataset = Dataset.from_list(flickr30k_data_list)
    logging.info(
        f"Flickr30k dataset length before split: {len(flickr30k_hf_dataset)}")

    # use subset_size to limit the size of the dataset
    if subset_size is not None:
        coco_hf_dataset = coco_hf_dataset.shuffle(seed=dataset_seed).select(
            range(min(subset_size, len(coco_hf_dataset))))
        flickr30k_hf_dataset = flickr30k_hf_dataset.shuffle(seed=dataset_seed).select(
            range(min(subset_size, len(flickr30k_hf_dataset))))
    logging.info(f"COCO dataset length after subset: {len(coco_hf_dataset)}")
    logging.info(
        f"Flickr30k dataset length after subset: {len(flickr30k_hf_dataset)}")

    coco_split_data = coco_hf_dataset.train_test_split(
        test_size=1 - split_ratio, seed=dataset_seed)
    flickr30k_split_data = flickr30k_hf_dataset.train_test_split(
        test_size=1 - split_ratio, seed=dataset_seed)

    logging.info(
        f"COCO train length: {len(coco_split_data['train'])}, COCO test length: {len(coco_split_data['test'])}")
    logging.info(
        f"Flickr30k train length: {len(flickr30k_split_data['train'])}, Flickr30k test length: {len(flickr30k_split_data['test'])}")

    train_datasets = [coco_split_data['train'], flickr30k_split_data['train']]
    eval_datasets = [coco_split_data['test'], flickr30k_split_data['test']]

    total_train_length = sum(len(d) for d in train_datasets)
    train_probabilities = [len(d) / total_train_length for d in train_datasets]

    train_data = interleave_datasets(train_datasets, seed=dataset_seed,
                                     probabilities=train_probabilities, stopping_strategy='all_exhausted')
    eval_data = interleave_datasets(eval_datasets, seed=dataset_seed,
                                    probabilities=train_probabilities, stopping_strategy='all_exhausted')

    logging.info(
        f"Length of training interleave_datasets with all_exhausted: {len(train_data)}")
    logging.info(
        f"Length of evaluation interleave_datasets with all_exhausted: {len(eval_data)}")

    train_dataset = CaptionDataset(train_data, tokenizer, data_args)
    eval_dataset = CaptionDataset(eval_data, tokenizer, data_args)

    logging.info(f"Length of final training dataset: {len(train_dataset)}")
    logging.info(f"Length of final evaluation dataset: {len(eval_dataset)}")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):

    # Check what version of PyTorch is installed
    print(torch.__version__)

    print(torch.cuda.device_count())

    # Check the current CUDA version being used
    print("CUDA Version: ", torch.version.cuda)
    # Check if CUDA is available and if so, print the device name
    print("Device name:", torch.cuda.get_device_properties("cuda").name)

    os.environ["MASTER_PORT"] = str((random.randint(1024, 65535)))
    print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

    job_id = os.environ.get("SLURM_JOB_ID", "N/A")
    node_list = os.environ.get("SLURM_NODELIST", "N/A")
    proc_id = os.environ.get("SLURM_PROCID", "N/A")

    print(f"Running on worker node(s): {node_list}")
    print(f"SLURM Job ID: {job_id}")
    print(f"SLURM Process ID: {proc_id}")

    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (
        torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(
                model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        # TODO: add model loader for LLaVA-1.6
        elif 'llava-v1.6-mistral' in model_args.model_name_or_path:
            # LlavaMistralForCausalLM, LlavaMistralConfig
            model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (
            torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # lora_target_modules = [
    #     "q_proj",
    #     "o_proj",
    #     "k_proj",
    #     "v_proj",
    #     "gate_proj",
    #     "up_proj",
    #     "down_proj",
    # ]

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        if training_args.lora_target_modules == "g_u_d":
            lora_target_modules = ["gate_proj", "up_proj",
                                   "down_proj"]  # mlp layer  0.69/0/86
        elif training_args.lora_target_modules == "q_k_v_o":
            lora_target_modules = ["q_proj", "k_proj",
                                   "v_proj", "o_proj"]  # attention layer
        else:
            lora_target_modules = find_all_linear_names(model)
        print("LoRA Target Modules:", lora_target_modules)
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            # target_modules=find_all_linear_names(model),
            # target_modules=find_part_linear_names(model, from_layer=16),
            # target_modules=find_target_module_names(model, lora_target_modules),
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # attention layer
            # target_modules=["gate_proj", "up_proj", "down_proj"], # mlp layer  0.69/0/86
            target_modules=lora_target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            # TODO: determine the task type
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    # TODO: add model loader for LLaVA-1.6
    elif 'llava-v1.6-mistral' in model_args.model_name_or_path:
        tokenizer = transformers.AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )  # for llava-hf: .tokenizer
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    print('version:', model_args.version)
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            print('version correct')
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version]
        else:
            print('version incorrect')
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "mistral_instruct"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    model.print_trainable_parameters()

    data_module = make_custom_supervised_data_module(tokenizer=tokenizer,
                                                     data_args=data_args)
    trainer = LLaVATrainer(model=model,
                           tokenizer=tokenizer,
                           args=training_args,
                           callbacks=[
                               EarlyStoppingCallback(
                                   early_stopping_patience=training_args.early_stopping_patience,
                                   early_stopping_threshold=training_args.early_stopping_threshold
                               )
                           ],
                           **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(
                training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(
                training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
