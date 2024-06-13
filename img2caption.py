from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
import os
from PIL import Image
import re
import glob
import argparse
from utils import save_file, remove_extra_spaces, remove_prompt


processor = LlavaNextProcessor.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16,
    low_cpu_mem_usage=True, pad_token_id=processor.tokenizer.pad_token_id)
model.to("cuda:0")


def _extract_qid_from_filename(filename):
    filename_only = os.path.basename(filename)
    match = re.search(rf'qid(\d+)', filename_only)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(
            f"Could not extract QID from filename: {filename_only}")


def gen_caption(gen_image_path):
    jpg_files = glob.glob(f"{gen_image_path}/*.jpg")
    caps = []
    for jpg in jpg_files:
        qid = _extract_qid_from_filename(jpg)
        image = Image.open(jpg)
        prompt = f"""
        [INST] <image>\nI have an image. Describe it in a sentence and
        generate a caption. The caption should remain coherent and logical
        without introducing any other additional details.
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
        caps.append({'qid': qid, 'query': caption})
    return caps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Captioning")
    parser.add_argument('--image_dir', type=str, required=False,
                        help="Path to the reference image folder.")
    parser.add_argument('--des_path', type=str, required=True,
                        help="Destination file")

    args = parser.parse_args()
    captions = gen_caption(args.image_dir)
    save_file(args.des_path, captions)
    print("Completed.")
