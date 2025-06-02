import os
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor
from transformers import Blip2ForConditionalGeneration, Blip2Processor, GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import LlavaForConditionalGeneration, AutoModelForPreTraining
from transformers import RobertaTokenizer, RobertaModel
from transformers import CLIPProcessor, CLIPModel, pipeline


def ModelLoader(model_name: str, device: str):

    local_model_path = ""
    if os.path.exists(local_model_path):
        model, processor = load_local_model(model_name, local_model_path)
    else:
        model, processor = load_remote_model(model_name)

    model = model.to(device)

    return model, processor


def load_local_model(model_name: str, local_model_path: str):

    if model_name == 'llava':
        model = LlavaForConditionalGeneration.from_pretrained(
            f"{local_model_path}/llava-1.5-7b-hf")
        processor = AutoProcessor.from_pretrained(
            f"{local_model_path}/llava-1.5-7b-hf")

    elif model_name == 'llama3':
        processor = AutoProcessor.from_pretrained(
            f"{local_model_path}/Llama-3-8B")
        model = AutoModelForPreTraining.from_pretrained(
            f"{local_model_path}/Llama-3-8B")

    elif model_name == 'Llama-3.2-3B-Instruct':
        processor = AutoProcessor.from_pretrained(
            f"{local_model_path}/Llama-3.2-3B-Instruct")
        model = AutoModelForCausalLM.from_pretrained(
            f"{local_model_path}/Llama-3.2-3B-Instruct")

    else:
        raise ValueError(f"Unsupported local model name: {model_name}")

    return model, processor


def load_remote_model(model_name: str):

    if model_name == 'blip2_opt':
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b")

    elif model_name == 'blip2_t5':
        processor = Blip2Processor.from_pretrained(
            "Salesforce/blip2-flan-t5-xl")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl")

    elif model_name == 'llava':
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf")
    elif model_name == 'llama3':
        processor = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B")
    elif model_name == 'llama3_vision':
        processor = AutoProcessor.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct")
        model = AutoModelForPreTraining.from_pretrained(
            "meta-llama/Llama-3.2-11B-Vision-Instruct")
    elif model_name == "roberta":
        processor = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        model = RobertaModel.from_pretrained("FacebookAI/roberta-base")
    elif model_name == "clip":
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError(f"Unsupported remote model name: {model_name}")

    return model, processor
