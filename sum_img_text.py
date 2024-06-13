from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import os
from PIL import Image
from utils import load_jsonl, save_file, remove_extra_spaces, remove_prompt
import argparse

processor = LlavaNextProcessor.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16,
    low_cpu_mem_usage=True, pad_token_id=processor.tokenizer.pad_token_id)
model.to("cuda:0")


def summarize_caption(raw_data, source_dir, style):
    adjusted_caps = []
    for idx, data in enumerate(raw_data):
        if style == "scribble":
            if data['has_new_detail']:
                print("qid:", data['qid'])
                path = os.path.join(source_dir, f"qid{data['qid']}.jpg")
                image = Image.open(path)
                new_types = data['new_detail_type']
                for i in range(len(new_types)):
                    new_type = data['new_detail_type'][i]
                    new_detail = data['added_details'][i]
                    prompt = f"""
                        [INST] <image>\nI have an image. Modify the image content by adding {new_type}
                        {new_detail} and then generate a caption. The revised caption should
                        remain coherent and logical without introducing any other additional details.
                        Please DON'T include any explanations and output only the caption. [/INST]
                        """
                    prompt = remove_extra_spaces(prompt)
                    inputs = processor(prompt, image, return_tensors="pt").to(
                        "cuda:0")
                    output = model.generate(**inputs, max_new_tokens=100)
                    new_caption = processor.decode(output[0],
                                                   skip_special_tokens=True)
                    new_caption = remove_prompt(new_caption)
                    new_caption = remove_extra_spaces(new_caption)
                    caption = new_caption
                print("new caption: ", caption)
                adjusted_caps.append({'qid': data['qid'], 'query': caption})
        else:
            if data['has_modification']:
                print("qid:", data['qid'])
                path = os.path.join(source_dir, f"qid{data['qid']}.jpg")
                image = Image.open(path)
                mod_types = data['modification_type']
                for i in range(len(mod_types)):
                    mod_type = data['modification_type'][i]
                    original_detail = data['removed_details'][i]
                    mod_detail = data['modified_details'][i]
                    prompt = f""" [INST] <image>\nI have an image. Modify
                    the image content by adjust {mod_type}
                        from {mod_detail} to {original_detail} and then
                        generate a caption. The revised caption should
                        remain coherent and logical without introducing any
                        other additional details. Please DON'T include any
                        explanations and output only the caption. [/INST] """
                    prompt = remove_extra_spaces(prompt)
                    inputs = processor(prompt, image, return_tensors="pt").to(
                        "cuda:0")
                    output = model.generate(**inputs, max_new_tokens=100)
                    new_caption = processor.decode(output[0],
                                                   skip_special_tokens=True)
                    new_caption = remove_prompt(new_caption)
                    new_caption = remove_extra_spaces(new_caption)
                    caption = new_caption
                print("new caption: ", caption)
                adjusted_caps.append({'qid': data['qid'], 'query': caption})
    print("New captions are generated.")
    return adjusted_caps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Summarization")
    parser.add_argument('--gt_file', type=str, required=True,
                        help="Path to the ground truth file.")
    parser.add_argument('--image_dir', type=str, required=False,
                        help="Path to the reference image folder.")
    parser.add_argument('--style', type=str, required=True,
                        help="Indicate the reference image style")
    parser.add_argument('--des_path', type=str, required=True,
                        help="Destination file")

    args = parser.parse_args()

    annotations = load_jsonl(args.gt_file)
    gen_captions = summarize_caption(annotations, args.image_dir, args.style)
    save_file(args.des_path, gen_captions)
