from PIL import Image
import argparse
import torch
import json
import os, sys

import numpy as np
from collections import defaultdict

from datetime import datetime

from scripts.post_processing import PostProcessor
import uuid

# Argument parser
parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="Llama-3-8B")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--results_file_for_cap_pairs", type=str, default=None, help="JSON file to save results")
parser.add_argument("--results_file_for_pos_cap", type=str, default=None, help="JSON file to save results")

# Trial arguments
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--is_samples", type=str, default='true')
parser.add_argument("--num_samples", type=int, default=6)

parser.add_argument("--process_positive", action="store_true", default=False)
parser.add_argument("--process_negative", action="store_true", default=False)

# Data path
parser.add_argument("--pos_neg_pairs_json_path", type=str, default=None)
parser.add_argument("--pos_json_path", type=str, default=None)
parser.add_argument("--image_folder_path", type=str, default=None)

args = parser.parse_args()  # Parse arguments here

model_name = args.model_name
device = args.device

def load_json(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def constraint_len():
    file_path = args.pos_json_path
    print(f'Processing file: {file_path}')
    data = load_json(file_path)
    post_processor = PostProcessor(model_name=args.model_name,device=device)
    post_processed_results=post_processor.filter_positive_captions(data, min_length=2, max_length=1000)
    return post_processed_results

def cal_sim_rob_clip(data):
    post_processor = PostProcessor(model_name=args.model_name,device=device)
    similarities = post_processor.filter_negative_captions_subset(data=data)
    return similarities

def cal_sim_clip_llm():
    print(f"Loading Model: {args.model_name}")

    image_folder_path = args.image_folder_path
    file_path = args.pos_neg_pairs_json_path
    print(f'Processing file: {file_path}')
    data = load_json(file_path)

    post_processor = PostProcessor(device=device)
    post_processed_results = post_processor.filter_negative_captions(data=data)

    outputs = {
        "pos-neg-caption-pairs file": file_path,
        "result": post_processed_results
    }
    return outputs

def main():
    if args.process_positive:
        print("Post-processing Positive Captions")
        post_processed_results=constraint_len()
        # Save file
        current_dateTime = datetime.now()
        time = f'{current_dateTime.year}-{current_dateTime.month}-{current_dateTime.day}_{current_dateTime.hour}:{current_dateTime.minute}'

        output_file = (args.results_file_for_pos_cap + time + ".json") if args.results_file_for_cap_pairs else f'output/_{args.model_name}_{time}.json'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(post_processed_results, f, indent=4) 
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
    if args.process_negative:
        print("Post-processing Negative Captions")
        is_json_folder=True
        all_file_path = []
        all_similarities = []

        if is_json_folder:
            folder_path = "output/neg-caption/coco/subset"
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.json'):
                    file_path = os.path.join(folder_path, file_name)
                    
                    data = load_json(file_path)
                    similarities=cal_sim_rob_clip(data=data) #cal_sim_clip_llm()

                    all_similarities.extend(similarities)
                    all_file_path.append(file_path)
        else:
            all_file_path.append(file_path)
            file_path ="output/neg-caption/coco/coco_caption.json"
            data = load_json(file_path)
            similarities=cal_sim_rob_clip(data=data)
            all_similarities.extend(similarities)
        
        outputs = {
            "pos-neg-caption-pairs file": all_file_path,
            "result": all_similarities
        }

        # Save file
        current_dateTime = datetime.now()
        time = f'{current_dateTime.year}-{current_dateTime.month}-{current_dateTime.day}_{current_dateTime.hour}:{current_dateTime.minute}'

        output_file = (args.results_file_for_cap_pairs + time + ".json") if args.results_file_for_cap_pairs else f'output/_{args.model_name}_{time}.json'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        try:
            with open(output_file, 'w') as f:
                json.dump(outputs, f, indent=4) 
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")

if __name__ == "__main__":
    main()