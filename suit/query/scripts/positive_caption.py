from PIL import Image
import os
import json

from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import LlavaForConditionalGeneration, AutoProcessor
from transformers import AutoProcessor, AutoModelForPreTraining, MllamaForConditionalGeneration, AutoTokenizer
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import base64

from .models import ModelLoader


class CaptionGenerator():
    def __init__(self,
                 model_name: str,
                 device: str = 'cpu',
                 dataloader: DataLoader = None):

        self.dataloader = dataloader
        self.model_name = model_name
        self.device = device
        self.model, self.processor = self.load_model(model_name, device)

    def load_model(self, model_name, device):
        if model_name == '':
            print("No model_name provided. Loading default model.")
            model_name = 'llava'
        return ModelLoader(model_name, device)

    def load_image(self, img_path):
        image = Image.open(img_path).convert("RGB")
        return image

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def generate_prompt(self, aspect, question):
        context = "You are an AI that provides detailed image analysis in semantic aspects."

        if aspect is None:
            prompt = question
        else:
            prompt = f"{context} Consider the semantic of image on {aspect} level, answer the following question:{question}"

        if self.model_name == 'llava':
            prompt = f"USER: <image>\n{prompt} ASSISTANT:"
        else:
            prompt = f"Question: {prompt} Answer:"

        return prompt

    def generate_single_caption(self, image, aspect, question):
        prompt = self.generate_prompt(aspect, question)

        try:
            if self.model_name == 'llava':
                inputs = self.processor(
                    text=prompt, images=image, return_tensors="pt").to(self.device)
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=200)
                caption = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                if "ASSISTANT:" in caption:
                    caption = caption.split("ASSISTANT:")[1].strip()

            elif self.model_name in ['blip2_opt', 'blip2_t5']:
                inputs = self.processor(
                    images=image, text=prompt, return_tensors="pt").to(self.device)
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=1000)
                caption = self.processor.batch_decode(
                    generated_ids, skip_special_tokens=True)[0].strip()

        except Exception as e:
            print(f"Error generating caption for {image}: {e}")
            caption = ""

        return caption

    def generate_image_caption(self, img_path, prompt):
        image = self.load_image(img_path)
        try:
            if isinstance(prompt, dict):
                captions = {}
                for aspect, question in prompt.items():
                    captions[aspect] = self.generate_single_caption(
                        image, aspect, question)
            elif isinstance(prompt, str):
                pos_captions = self.generate_single_caption(
                    image, None, prompt)
                extracted_pos_captions = self.parse_positive_caption(
                    pos_captions)
                general_caption = self.generate_general_caption(
                    extracted_pos_captions)
                captions = {
                    'general_caption': general_caption,
                    'pos_captions': pos_captions,
                    'extracted_pos_captions': extracted_pos_captions
                }
        finally:
            image.close()
        return captions

    def generate_general_image_caption(self, img_path, prompt):
        image = self.load_image(img_path)
        try:
            captions = self.generate_single_caption(image, None, prompt)
        finally:
            image.close()
        return captions

    def parse_positive_caption(self, pos_caption):
        """
        Parse the positive caption string to extract the semantic aspects.
        The input format is a string with [Object], [Attributes], [Actions], [Environment], and [Relations].
        """
        parsed_caption = {}

        # Look for each semantic aspect in the string and extract the corresponding value
        semantic_keys = ["Object", "Attributes",
                         "Actions", "Environment", "Relations"]

        for key in semantic_keys:
            start_token = f"* [{key}]:"
            end_token = "* [" if key != "Relations" else None

            if start_token in pos_caption:
                start_index = pos_caption.index(start_token) + len(start_token)

                if end_token:
                    end_index = pos_caption.find(end_token, start_index)
                    parsed_caption[key] = pos_caption[start_index:end_index].strip()
                else:
                    parsed_caption[key] = pos_caption[start_index:].strip()

        return parsed_caption

    def generate_general_caption(self, pos_captions):
        """
        Generate a general caption that combines all available semantic aspects into a coherent sentence.
        """
        general_caption_parts = []

        # Check for each semantic aspect and append it to the general caption if it's not empty
        if pos_captions.get('Object'):
            general_caption_parts.append(
                f"A {pos_captions['Attributes']} {pos_captions['Object']}")

        if pos_captions.get('Environment'):
            general_caption_parts.append(
                f"is in {pos_captions['Environment']}")

        if pos_captions.get('Actions'):
            general_caption_parts.append(f"and is {pos_captions['Actions']}")

        if pos_captions.get('Relations'):
            general_caption_parts.append(f"with {pos_captions['Relations']}")

        # Join all the parts into a final sentence and return
        general_caption = ". ".join(general_caption_parts).strip()

        # Handle case where none of the semantic aspects are available
        if not general_caption:
            general_caption = "No information available to generate a caption."

        return general_caption

    def run(self, batch, prompt):
        results = []
        for i, query_id in enumerate(batch['query_id']):
            reference_img_id = batch['reference_img_id'][i]
            relative_caption = batch['relative_caption'][i]
            shared_concept = batch['shared_concept'][i]

            image_path = self.dataloader.get_img_path(query_id)

            positive_caption = self.generate_image_caption(image_path, prompt)
            general_caption = self.generate_general_caption(positive_caption)

            result = {
                'query_id': query_id,
                'img_id': reference_img_id,
                'positive_caption': positive_caption,
                'general_caption': general_caption,
                'relative_caption': relative_caption,
                'shared_concept': shared_concept,
            }
            if 'target_img_id' in batch:
                target_img_id = batch['target_img_id'][i]
                gt_img_ids = batch.get('gt_img_ids', [[]])[i]
                semantic_aspects = batch.get('semantic_aspects', [[]])[i]
                result.update({
                    'target_img_id': target_img_id,
                    'gt_img_ids': gt_img_ids,
                    'semantic_aspects': semantic_aspects
                })
            results.append(result)
        return results
