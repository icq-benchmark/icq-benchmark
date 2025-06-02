import json
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


from tqdm import tqdm
from datetime import datetime
import copy

from .models import ModelLoader


class PostProcessor:
    def __init__(self, model_name: str, device="cpu"):
        self.device = device
        self.model_name = model_name
        self.model, self.processor = self.load_model()
        self.clip_model, self.clip_processor = self.load_clip_model()

    def load_model(self):
        if self.model_name == '':
            print("No model_name provided. Loading default model.")
            self.model_name = 'llama'
        return ModelLoader(self.model_name, self.device)

    def load_clip_model(self):
        return ModelLoader("clip", self.device)

    def llm_evaluation(self, pos_caption, neg_caption):
        prompt = f"Does the following negative caption logically contradict the positive caption?\n\nPositive Caption: {pos_caption}\n\nNegative Caption: {neg_caption}\nAnswer with 'Yes' or 'No':"
        inputs = self.processor(prompt, return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=15)
        answer = self.processor.decode(output[0], skip_special_tokens=True)
        # Return True if is a VALID 'negative caption'
        return "Yes" in answer

    def clip_similarity(self, pos_caption, neg_caption):
        inputs = self.clip_processor(text=[pos_caption, neg_caption], padding="max_length",
                                     truncation=True, return_tensors="pt", max_length=77).to(self.device)
        # Get text features for both positive and negative captions
        text_features = self.clip_model.get_text_features(**inputs)
        # Normalize the text features
        text_features = text_features / \
            text_features.norm(dim=-1, keepdim=True)
        # Calculate cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            text_features[0], text_features[1], dim=-1)
        return similarity.item()

    def roberta_similarity(self, embedding1, embedding2):
        return cosine_similarity(embedding1.detach().numpy(), embedding2.detach().numpy())[0][0]

    def get_sentence_embedding(self, sentence):
        inputs = self.processor(
            sentence, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # use [CLS] token output to represent the sentence
        return outputs.last_hidden_state[:, 0, :]

    def filter_negative_captions_subset(self, data):
        similarities = []
        for key in data["_default"]:
            item = data["_default"][key]
            positive_caption = item["positive_caption"]
            refinements = item["refinements"]
            clip_scores = {}
            roberta_scores = {}
            for neg_key, negative_caption in item["negative_caption"].items():
                if positive_caption and negative_caption:
                    clip_score = self.clip_similarity(
                        positive_caption, negative_caption)
                    pos_embedding = self.get_sentence_embedding(
                        positive_caption)
                    neg_embedding = self.get_sentence_embedding(
                        negative_caption)
                    roberta_score = self.roberta_similarity(
                        pos_embedding, neg_embedding)
                else:
                    # llm_result = "N/A"  # Mark as "N/A" if one of the captions is missing
                    clip_score = "N/A"
                    roberta_score = "N/A"
                clip_scores[neg_key] = float(
                    clip_score) if clip_score != "N/A" else "N/A"
                roberta_scores[neg_key] = float(
                    roberta_score) if roberta_score != "N/A" else "N/A"

                similarities.append({
                    "image_id": item["image_id"],
                    "positive_caption": positive_caption,
                    "negative_caption": negative_caption,
                    "refinements": refinements,
                    "negative_type": neg_key,
                    "clip_similarity": clip_scores,
                    "roberta_similarity": roberta_scores
                })

        return similarities

    def filter_negative_captions(self, data):
        # Assuming the original data structure has 'results' inside 'pos-neg-caption-pairs'
        data_entries = data.get('pos-neg-caption-pairs', {}).get('results', [])

        for entry in data_entries:
            pos_captions = entry['positive_caption']
            neg_captions = entry['negative_caption']

            # llm_results = {}
            clip_scores = {}
            semantic_keys = ['Object', 'Attributes',
                             'Actions', 'Environment', 'Relations']

            for key in semantic_keys:
                pos_caption = pos_captions  # Positive caption is a single sentence
                # Negative captions are stored under specific keys
                neg_caption = neg_captions.get(key, "")

                if pos_caption and neg_caption:
                    # Run LLaMA evaluation
                    # llm_result = self.llm_evaluation(pos_caption, neg_caption)

                    # Run CLIP similarity check
                    clip_score = self.clip_similarity(pos_caption, neg_caption)
                else:
                    # llm_result = "N/A"  # Mark as "N/A" if one of the captions is missing
                    clip_score = "N/A"

                # Store the results in the original entry structure
                # llm_results[key] = llm_result
                clip_scores[key] = clip_score

            # Append LLM and CLIP results to the entry
            entry['clip_similarity'] = clip_scores
            # entry['llm_results'] = llm_results

        return data

    # Filter positive captions by length

    def filter_positive_captions(self, data, min_length=5, max_length=2000):

        filtered_data = {
            "image-caption": []
        }

        for entry in data["image-caption"]:
            filtered_results = []
            for result in entry["results"]:
                positive_caption = result.get("positive_caption", "")
                image_id = result.get("image_id", "Unknown")
                if min_length <= len(positive_caption) <= max_length:
                    filtered_results.append(result)
                else:
                    print(
                        f"delete: image_id: {image_id}, caption: {positive_caption}")

            if filtered_results:
                filtered_entry = entry.copy()
                filtered_entry["results"] = filtered_results
                filtered_data["image-caption"].append(filtered_entry)

        return filtered_data
