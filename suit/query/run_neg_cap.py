from PIL import Image
import argparse
import torch
import json
import os
import sys

import numpy as np
from collections import defaultdict

from datetime import datetime
import random
from tinydb import TinyDB, Query
from tinydb.storages import JSONStorage
from filelock import FileLock
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from scripts.ImageCaptionDataset import CIRCODataLoader, JourneyDBDataLoader, COCODataLoader, CaptionDataset, Flickr30kDataLoader

from scripts.api_generate import NegCaptionGenerator

# argument parser
parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str, default="Llama-3-8B")
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--results_file", type=str, default=None,
                    help="JSON file to save results")
parser.add_argument("--db_path", type=str, default=None,
                    help="JSON file to save results")

parser.add_argument("--dataset", type=str, default="CIRCO")

parser.add_argument("--use_chatgpt", action="store_true", default=False)

# Trial arguments
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--random_id", type=int, default=0)
parser.add_argument("--num_samples", type=int, default=6)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--prompt_type", type=str, default=None, )
parser.add_argument("--is_samples", type=str, default='true', )
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for dataset shuffling")

parser.add_argument("--subset_id", type=int, default=0,
                    help="Specify which subset to process.")
parser.add_argument("--num_subsets", type=int, default=4)

# Positive Caption Data
parser.add_argument("--coco_pos_cap_json_path", type=str, default=None)

parser.add_argument("--flickr30k_annotations_csv_path", type=str, default=None)
parser.add_argument("--flickr30k_image_dir_path", type=str, default=None)

args = parser.parse_args()
model_name = args.model_name
device = args.device


def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch


def get_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=custom_collate_fn,
        num_workers=4,
        pin_memory=True
    )


def prepare_eval_samples(test_dataset, num_samples, seed):
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(test_dataset), num_samples, replace=False)
    return torch.utils.data.Subset(test_dataset, random_indices)


def split_dataset(dataset, num_splits):
    indices = np.arange(len(dataset))
    subsets = np.array_split(indices, num_splits)
    return [Subset(dataset, subset) for subset in subsets]


def process_subset_single_caption(subset_id, subset, prompt, neg_caption_generator, dataset_name):
    folder_path = args.db_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    db_file = os.path.join(
        folder_path, f'{dataset_name}_{args.is_samples}_subset_{args.subset_id}.json')
    db_lock = FileLock(f"{db_file}.lock")

    with db_lock:
        db = TinyDB(db_file)
        query = Query()
        processed_image_ids = {item['image_id'] for item in db.all()}

    # all_results = []
    save_threshold = 1
    batch_counter = 0
    batch_results = []

    dataloader = get_dataloader(subset, args.batch_size)

    for batch in tqdm(dataloader, desc=f"Processing subset {subset_id}", total=len(dataloader)):
        batch_counter += 1
        for i, image_id in enumerate(batch['image_id']):
            if image_id in processed_image_ids:
                print(f"Skipping {image_id}, already processed.")
                continue
            if dataset_name == "flickr30k":
                caption = batch.get('caption', [None])[i]
                negative_caption, refinements = neg_caption_generator.generate_random_neg_caption(
                    positive_caption=caption, prompt=prompt)

                batch_results.append({
                    'subset_id': subset_id,
                    'image_id': image_id,
                    'positive_caption': caption,
                    "negative_caption": negative_caption,
                    "refinements": refinements
                })

        if batch_counter >= save_threshold:
            # Save results to the subset-specific database
            with db_lock:
                db.insert_multiple(batch_results)
            batch_counter = 0
            batch_results = []

    if len(batch_results) > 0:
        with db_lock:
            db.insert_multiple(batch_results)

    db_length = len(db.all())
    print("db_length:", db_length)  # db.all() returns a list of all records

    # Close the database to ensure the file is properly handled
    db.close()

    return db_length


def process_subset(subset_id, subset, prompt, neg_caption_generator, dataset_name):

    folder_path = args.db_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    db_file = os.path.join(
        folder_path, f'{dataset_name}_{args.is_samples}_subset_{args.subset_id}.json')
    db_lock = FileLock(f"{db_file}.lock")

    with db_lock:
        db = TinyDB(db_file)
        query = Query()
        processed_image_ids = {item['image_id'] for item in db.all()}

    # all_results = []
    save_threshold = 1
    batch_counter = 0
    batch_results = []

    dataloader = get_dataloader(subset, args.batch_size)

    for batch in tqdm(dataloader, desc=f"Processing subset {subset_id}", total=len(dataloader)):
        batch_counter += 1
        for i, image_id in enumerate(batch['image_id']):
            if image_id in processed_image_ids:
                print(f"Skipping {image_id}, already processed.")
                continue
            if dataset_name == "flickr30k":
                captions = batch['caption']
                # for multiple captions
                # captions = [caption for sublist in batch['caption'] for caption in sublist]

                with db_lock:
                    existing_record = db.get(query.image_id == image_id)

                if existing_record:
                    for caption in captions:
                        negative_caption, refinements = neg_caption_generator.generate_random_neg_caption(
                            positive_caption=caption, prompt=prompt)

                        existing_record['results'].append({
                            "positive_caption": caption,
                            "negative_caption": negative_caption,
                            "refinements": refinements
                        })

                    with db_lock:
                        db.update(existing_record, query.image_id == image_id)
                else:
                    new_record = {
                        'subset_id': subset_id,
                        'image_id': image_id,
                        'positive_caption': [],
                        # 'negative_caption': [],
                        # 'refinements': []
                    }
                    for caption in captions:
                        negative_caption, refinements = neg_caption_generator.generate_random_neg_caption(
                            positive_caption=caption, prompt=prompt)

                        new_record['positive_caption'].append({
                            "positive_caption": caption,
                            "negative_caption": negative_caption,
                            "refinements": refinements
                        })

                    batch_results.append(new_record)

        if batch_counter >= save_threshold:
            # Save results to the subset-specific database
            with db_lock:
                db.insert_multiple(batch_results)
            batch_counter = 0
            batch_results = []

    if len(batch_results) > 0:
        with db_lock:
            db.insert_multiple(batch_results)

    db_length = len(db.all())
    print("db_length:", db_length)  # db.all() returns a list of all records

    # Close the database to ensure the file is properly handled
    db.close()

    return db_length


def flickr30k_NegCaption(
    args: argparse.Namespace,
    prompt: str = "",
    dataset_name: str = "flickr30k",
):
    model_name = args.model_name
    print(f"Loading Model: {args.model_name}")

    device = args.device
    use_chatgpt = args.use_chatgpt

    if dataset_name == "flickr30k":
        load_dataset = Flickr30kDataLoader(
            annotation_path=args.flickr30k_annotations_csv_path,
            image_folder=args.flickr30k_image_dir_path,
            is_testset=False
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset = prepare_eval_samples(load_dataset, args.num_samples,
                                   seed=args.seed) if args.is_samples == 'true' else load_dataset
    print(f"Dataset length before splitting: {len(dataset)}")

    subsets = split_dataset(dataset, args.num_subsets)
    print(
        f"Number of subsets: {len(subsets)}, Subset sizes: {[len(s) for s in subsets]}")

    neg_caption_generator = NegCaptionGenerator(
        model_name, device, use_chatgpt)

    with ThreadPoolExecutor() as executor:
        subset = subsets[args.subset_id]
        future = executor.submit(process_subset_single_caption, args.subset_id,
                                 subset, prompt, neg_caption_generator, dataset_name)
        for _ in tqdm(range(1), desc=f"Processing subset {args.subset_id}"):
            try:
                data_length = future.result()
            except Exception as e:
                print(f"Error processing future: {e}")

    outputs = {
        "dataset": dataset_name,
        "model": args.model_name,
        "prompt": prompt,
        "results": data_length
    }
    return outputs


def coco_Caption(
    args: argparse.Namespace,
    prompt: str = "",
    dataset_name: str = "coco",
):
    model_name = args.model_name
    print(f"Loading Model: {args.model_name}")

    device = args.device
    use_chatgpt = args.use_chatgpt

    if dataset_name == "coco":
        file_path = args.coco_pos_cap_json_path

    print('file_path:', file_path)

    folder_path = args.db_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    db_file = os.path.join(
        folder_path, f'{dataset_name}_{args.is_samples}_subset_{args.subset_id}.json')
    db_lock = FileLock(f"{db_file}.lock")

    with db_lock:
        db = TinyDB(db_file)
        query = Query()

        neg_caption_generator = NegCaptionGenerator(
            model_name, device, use_chatgpt)

        data = load_json(file_path)
        data = data.get('_default', {})

        print(f"Processing {len(data)} entries from the dataset.")

        for entry_index, (key, entry) in enumerate(data.items()):
            image_id = entry.get('image_id')
            positive_caption = entry.get('positive_caption', "")
            original_captions = entry.get('original_caption', [])

            print(f"Processing image {image_id}")

            existing_entry = db.get(query.image_id == image_id)

            if existing_entry:
                print(f"Skipping image {image_id}, already processed.")
                continue

            negative_captions, refinements = neg_caption_generator.generate_random_neg_caption(
                positive_caption, prompt)

            if not negative_captions:
                print(f"Skipping entry {entry_index} due to error.")
                continue
            entry_data = {
                'image_id': image_id,
                'original_captions': original_captions,
                'positive_caption': positive_caption,
                'negative_caption': negative_captions,
                'refinements': refinements
            }

            db.insert(entry_data)

    print(f"Finished processing dataset: {dataset_name}")

    db_length = len(db.all())
    print("db_length:", db_length)  # db.all() returns a list of all records

    # Close the database to ensure the file is properly handled
    db.close()

    return db_length


def coco_NegCaption(
    args: argparse.Namespace,
    prompt: str = "",
    dataset_name: str = "coco",
):
    model_name = args.model_name
    print(f"Loading Model: {args.model_name}")

    device = args.device
    use_chatgpt = args.use_chatgpt
    if dataset_name == "coco":
        file_path = args.coco_pos_cap_json_path

    device = args.device

    neg_caption_generator = NegCaptionGenerator(
        model_name, device, use_chatgpt)

    print(f'Processing file: {file_path}')
    processed_data = []
    data = load_json(file_path)
    data = data.get('_default', {})
    print(f"Processing {len(data)} entries from the dataset.")

    for entry_index, (key, entry) in enumerate(data.items()):
        print('entry:', entry)
        image_id = entry.get('image_id')
        positive_caption = entry.get('positive_caption', "")
        original_captions = entry.get('original_caption', [])
        print(
            f"Processing image {image_id} with {len(original_captions)} original captions.")

        for result_index, original_caption in enumerate(original_captions):
            negative_captions, refinements = neg_caption_generator.generate_neg_caption(
                positive_caption, prompt)
            if not negative_captions:
                print(
                    f"Skipping entry {entry_index}, result {result_index} due to error.")
                continue

            entry_data = {
                'image_id': image_id,
                'original_caption': original_caption,
                'positive_caption': positive_caption,
                'negative_caption': negative_captions,
                'refinements': refinements
            }

            processed_data.append(entry_data)

    outputs = {
        "information": {
            "pos-cap file": file_path,
            "model": model_name,
            "prompt for neg-cap": prompt,
        },
        "results": processed_data
    }

    return outputs


def main():
    args, leftovers = parser.parse_known_args()
    final_result = defaultdict(list)

    prompt_type = args.prompt_type
    print(f"Generating Negative Caption with: {prompt_type}")
    prompt = get_prompt_template(prompt_type)

    num_subsets = args.num_subsets

    if args.dataset == "flickr30k":
        if args.subset_id >= num_subsets or args.subset_id < 0:
            print(
                f"Error: Invalid subset_id {args.subset_id}. Valid range is 0 to {num_subsets-1}.")
            sys.exit(1)
        print('Processing Flickr30k dataset')
        file_path = args.flickr30k_annotations_csv_path
        final_result = flickr30k_NegCaption(
            args, prompt, dataset_name="flickr30k")

    elif args.dataset == "coco":
        print('Processing coco dataset')
        file_path = args.coco_pos_cap_json_path
        # coco_NegCaption(args, dataset_name="coco")
        final_result = coco_Caption(args, prompt, dataset_name="coco")

    # save results
    current_dateTime = datetime.now()
    time = current_dateTime.strftime('%Y-%m-%d-%H-%M')

    if args.subset_id:
        subset_id = args.subset_id

        save_output_path = (args.results_file + time +
                            f"_subset_{subset_id}.json") if args.results_file else f'output/_{args.model_name}_captions_{time}_subset_{subset_id}.json'
    else:
        save_output_path = (args.results_file + time +
                            ".json") if args.results_file else f'output/_{args.model_name}_captions_{time}.json'

    os.makedirs(os.path.dirname(save_output_path), exist_ok=True)

    output = {
        'file_path': file_path,
        'output_path': save_output_path,
        'prompt': prompt,
        "output length": final_result
    }

    try:
        with open(save_output_path, 'w') as f:
            json.dump(output, f, indent=4)
        print(f"Results saved to {save_output_path}")
    except Exception as e:
        print(f"Error saving results: {e}")


def get_prompt_template(prompt_type):
    if prompt_type == "neg-cap-prompt":
        context = "In this task, you are given an input sentence. "
        return {
            "Object": context+"Your job is to generate a sentence with a different meaning by changing only one of the main entities, specifically the subject (the one performing the action) or the object (the one affected by the action), in the sentence. Do not change any other parts of the sentence such as the verb or sentence structure. The modified sentence must still make sense and follow the same grammatical structure. Do not introduce any new entities. Only output the modified sentence, do not include explanations. Input sentence: '{}'. Output:",
            "Attributes": context+"Your job is to generate a sentence with a different meaning by changing only one of the attributes of the objects in the sentence. These attributes include adjectives such as color, size, shape, texture, or material, describing the subject or object. Do not change the subject, object, or verb in the sentence, and ensure that the modified sentence remains reasonable and grammatically correct. Do not introduce new attributes. Only output the modified sentence, do not include explanations. Input sentence: '{}'. Output:",
            "Actions": context+"Your job is to generate a sentence with a different meaning by changing only one of the action verbs (the predicate) in the sentence. The subject (who is performing the action) and object (who or what is affected by the action) should remain unchanged. Make sure that the new sentence keeps the same structure but changes the action being described. Only output the modified sentence, do not include explanations. Input sentence: '{}'. Output:",
            "Environment":
            context +
                "Your job is to generate a sentence with a different meaning by changing only one of the environmental details (the setting or background) in the sentence. This includes changing the where or when part of the sentence, such as location, background, or atmosphere. Do not change the subject, object, or action in the sentence. The new sentence should remain grammatically correct and reasonable. Only output the modified sentence, do not include explanations. Input sentence: '{}'. Output:",
            "Relations": context + "Your job is to generate a sentence with a different meaning by changing only one of the relationships between the subject, object, or other entities in the sentence. This includes altering spatial, temporal, or interaction-based connections between the entities. The subject, object, and action verbs should remain unchanged. The sentence should still follow the same structure and be grammatically correct. Only output the modified sentence, do not include explanations. Input sentence: '{}'. Output:"
        }

    else:
        print("Invalid prompt type")


if __name__ == "__main__":
    main()
