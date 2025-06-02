from PIL import Image
import requests
import argparse
import torch
import json
import os
import sys
import numpy as np
from collections import defaultdict

from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
from torch.nn import DataParallel

import uuid
from datetime import datetime

from tqdm import tqdm

from scripts.ImageCaptionDataset import CIRCODataLoader, JourneyDBDataLoader, COCODataLoader, CaptionDataset
from scripts.positive_caption import CaptionGenerator

from tinydb import TinyDB, Query
from tinydb.storages import JSONStorage
from filelock import FileLock
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# argument parser
parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str)

parser.add_argument("--tokenizer_path", type=str)
parser.add_argument("--lang_encoder_path", type=str)

parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--results_file", type=str, default=None,
                    help="JSON file to save results")
parser.add_argument("--db_path", type=str, default=None,
                    help="JSON file to save results")

# Trial arguments
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--num_samples", type=int, default=6)
parser.add_argument("--max_length", type=int, default=128)
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for dataset shuffling")

parser.add_argument("--prompt_type", type=str)
parser.add_argument("--is_samples", type=str, default='true')

parser.add_argument("--dataset", type=str, default="CIRCO")

parser.add_argument("--subset_id", type=int, required=True,
                    help="Specify which subset to process.")
parser.add_argument("--num_subsets", type=int, default=10)

# Per-dataset flags
parser.add_argument(
    "--circo_dataset",
    action="store_true",
    default=False,
    help="Whether to load CIRCO.",
)
parser.add_argument(
    "--JourneyDB_dataset",
    action="store_true",
    default=True,
    help="Whether to load JourneyDB.",
)
parser.add_argument(
    "--gqa_dataset",
    action="store_true",
    default=False,
    help="Whether to load gqa.",
)
parser.add_argument(
    "--coco_dataset",
    action="store_true",
    default=False,
    help="Whether to load coco.",
)

# CIRCO Dataset
parser.add_argument("--circo_image_folder_path", type=str)
parser.add_argument("--circo_train_annotations_json_path", type=str)
parser.add_argument("--circo_val_annotations_json_path", type=str)
parser.add_argument("--circo_test_annotations_json_path", type=str)

# JourneyDB Dataset
parser.add_argument("--JourneyDB_image_folder_path", type=str)
parser.add_argument("--JourneyDB_test_annotations_json_path", type=str)

# GQA Dataset
parser.add_argument("--gqa_image_folder_path", type=str)

# COCO Dataset
parser.add_argument("--coco_train_image_dir_path", type=str, default=None,)
parser.add_argument("--coco_val_image_dir_path", type=str, default=None,)
parser.add_argument("--coco_train_annotations_json_path",
                    type=str, default=None,)
parser.add_argument("--coco_val_annotations_json_path",
                    type=str, default=None,)

args = parser.parse_args()
model_name = args.model_name
device = args.device


def split_dataset(dataset, num_splits, seed=42):
    length = len(dataset)
    indices = np.arange(length)

    subsets = np.array_split(indices, num_splits)
    return [Subset(dataset, subset_indices) for subset_indices in subsets]


def process_subset(subset_id, subset, prompt, caption_generator, original_dataset):
    # Create a separate TinyDB instance for each subset
    folder_path = args.db_path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Create a separate TinyDB instance for each subset in the specified folder
    db_file = os.path.join(
        folder_path, f'{args.dataset}_captions_subset_{subset_id}.json')
    db_lock = FileLock(os.path.join(
        folder_path, f"{args.dataset}_captions_subset_{subset_id}.json.lock"))

    with db_lock:
        db = TinyDB(db_file)  # Use the separate database for each subset
        query = Query()
        processed_image_ids = {item['image_id'] for item in db.all()}

    # all_results = []
    save_threshold = 5
    batch_counter = 0
    batch_results = []

    dataloader = get_dataloader(subset, args.batch_size)

    for batch in tqdm(dataloader, desc=f"Processing subset {subset_id}", total=len(dataloader)):
        batch_counter += 1

        for i, image_id in enumerate(batch['image_id']):
            if image_id in processed_image_ids:
                print(f"Skipping {image_id}, already processed.")
                continue

            # Get the image path
            image_path = original_dataset.get_img_path(image_id)
            positive_caption = caption_generator.generate_general_image_caption(
                image_path, prompt)

            # Add results for this batch
            batch_results.append({
                'subset_id': subset_id,  # Add subset_id to the results
                'image_id': image_id,
                'positive_caption': positive_caption,
                'original_caption': batch.get('caption', [None])[i]
            })

        # all_results.extend(batch_results)#append
        if batch_counter >= save_threshold:
            # Save results to the subset-specific database
            with db_lock:
                db.insert_multiple(batch_results)
            # all_results = []
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


def coco_captioning(
    args: argparse.Namespace,
    dataset_name: str = "coco-val",
):
    image_train_dir_path = args.coco_train_image_dir_path
    image_val_dir_path = args.coco_val_image_dir_path
    annotations_train_json_path = args.coco_train_annotations_json_path
    annotations_val_json_path = args.coco_val_annotations_json_path
    subset_id = args.subset_id

    if dataset_name == "coco-train":
        load_dataset = COCODataLoader(
            annotation_path=annotations_train_json_path,
            image_folder=image_train_dir_path,
            is_testset=False
        )
    elif dataset_name == "coco-val":
        load_dataset = COCODataLoader(
            annotation_path=annotations_val_json_path,
            image_folder=image_val_dir_path,
            is_testset=False
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset = prepare_eval_samples(load_dataset, args.num_samples,
                                   seed=args.seed) if args.is_samples == 'true' else load_dataset

    subsets = split_dataset(dataset, args.num_subsets, seed=args.seed)
    print(f"Dataset length before splitting: {len(dataset)}")
    print(
        f"Number of subsets: {len(subsets)}, Subset sizes: {[len(s) for s in subsets]}")

    prompt = get_prompt_template(args.prompt_type)
    caption_generator = CaptionGenerator(
        model_name=args.model_name, device=args.device, dataloader=load_dataset)
    all_results = []
    with ThreadPoolExecutor() as executor:
        subset = subsets[args.subset_id]
        future = executor.submit(
            process_subset, args.subset_id, subset, prompt, caption_generator, load_dataset)
        for _ in tqdm(range(1), desc=f"Processing subset {args.subset_id}"):
            try:
                data_length = future.result()
                # if result is not None:
                #     all_results.extend(result)
            except Exception as e:
                print(f"Error processing future: {e}")

    outputs = {
        "dataset": dataset_name,
        "model": args.model_name,
        "prompt": prompt,
        "results": data_length
    }
    return outputs


def main():
    args, leftovers = parser.parse_known_args()
    final_result = defaultdict(list)

    num_subsets = args.num_subsets
    if args.subset_id >= num_subsets or args.subset_id < 0:
        print(
            f"Error: Invalid subset_id {args.subset_id}. Valid range is 0 to {num_subsets-1}.")
        sys.exit(1)

    if args.dataset == "CIRCO":
        print("CIRCO Captioning...")
        if args.is_samples == 'true':
            output = circo_captioning(
                args=args,
                dataset_name="circo-val",
            )
            final_result["image-caption"].append(output)
        else:
            val_output = circo_captioning(
                args=args,
                dataset_name="circo-val",
            )
            final_result["image-caption"].append(val_output)

            test_output = circo_captioning(
                args=args,
                dataset_name="circo-test",
            )
            final_result["image-caption"].append(test_output)

    elif args.dataset == "COCO":
        print("COCO Captioning...")
        if args.is_samples == 'true':
            output = coco_captioning(
                args=args,
                dataset_name="coco-val",
            )
            final_result["image-caption"].append(output)
        else:
            val_output = coco_captioning(
                args=args,
                dataset_name="coco-val",
            )
            final_result["image-caption"].append(val_output)

            # train_output=coco_captioning(
            #     args=args,
            #     dataset_name="coco-train",
            # )
            # final_result["image-caption"].append(train_output)

    # elif args.JourneyDB_dataset:
    #     print("JourneyDB Captioning...")
    #     if args.is_samples == 'true':
    #         output = JourneyDB_captioning(
    #             args=args,
    #             dataset_name="JourneyDB-test",
    #         )
    #         final_result["JourneyDB-sample"].append(output)
    #     else:
    #         output = JourneyDB_captioning(
    #             args=args,
    #             dataset_name="JourneyDB-val",
    #         )
    #         final_result["JourneyDB-val"].append(output)

    #         output = JourneyDB_captioning(
    #             args=args,
    #             dataset_name="JourneyDB-test",
    #         )
    #         final_result["JourneyDB-test"].append(output)

    # save resulte
    current_dateTime = datetime.now()
    time = current_dateTime.strftime('%Y-%m-%d-%H-%M')
    if args.subset_id:
        subset_id = args.subset_id

        output_file = (args.results_file + time +
                    f"_subset_{subset_id}.json") if args.results_file else f'output/_{args.model_name}_captions_{time}_subset_{subset_id}.json'
    else:
        output_file = (args.results_file + time +
                    ".json") if args.results_file else f'output/_{args.model_name}_captions_{time}.json'
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    try:
        with open(output_file, 'w') as f:
            json.dump(final_result, f, indent=4)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


def get_prompt_template(prompt_type):
    if prompt_type == "pos-cap-prompt":
        prompt = """In this task, you are given an input image. Your task is to generate a single, complete sentence that describes the image in detail, incorporating the following five semantic aspects, and avoid using vague or abstract words:
                    - [Object]: Who or what are the main entities (such as subject, object, people, animal, ...) in the image?
                    - [Attributes]: What are the key attributes or characteristics of these entities (e.g., color, size, shape, texture, emotion, appearance, age, ...)? Provide concrete descriptions, avoiding abstract terms.
                    - [Actions]: What are the main actions taking place in the image? Use specific verbs and clearly state what is happening.
                    - [Environment]: What is the specific environment (focus on 'where', e.g., location, atmosphere, weather, background, setting, scenario ...) of the image? Avoid general terms like "outdoors" and instead specify the actual location (e.g., "in a park," "on a city street").
                    - [Relations]: How are the entities in the image related to each other (focus on the relationship between different entities, such as spatial, temporal, interaction-based connections, ...)? Use clear relationships rather than abstract or vague connections.
                Ensure that your sentence combines all of these aspects smoothly into a single coherent sentence. Avoid generating multiple sentences or breaking the description into separate parts. Your output must be one well-structured sentence.
                Output:"""

    else:
        return "Invalid prompt type"
    return prompt


def circo_captioning(
    args: argparse.Namespace,
    seed: int = 42,
    # min_generation_length: int = 0,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = 0.0,
    num_shots: int = 8,
    dataset_name: str = "circo-val",
):

    image_path = args.circo_image_folder_path
    json_test_path = args.circo_test_annotations_json_path
    json_val_path = args.circo_val_annotations_json_path
    json_train_path = args.circo_train_annotations_json_path

    print(f"Load Dataset...{dataset_name}")

    if dataset_name == "circo-val":
        load_dataset = CIRCODataLoader(
            annotation_path=json_val_path,
            image_folder=image_path,
            is_testset=False
        )
    elif dataset_name == "circo-test":
        load_dataset = CIRCODataLoader(
            annotation_path=json_test_path,
            image_folder=image_path,
            is_testset=True
        )
    elif dataset_name == "circo-train":
        load_dataset = CIRCODataLoader(
            annotation_path=json_train_path,
            image_folder=image_path,
            is_testset=False
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    print("dataset length:", load_dataset.__len__())

    if args.is_samples == 'true':
        print("sampling....")
        dataset = prepare_eval_samples(load_dataset, args.num_samples, seed=40)
    else:
        print("default no sampling.")
        dataset = load_dataset

    dataloader = get_dataloader(dataset, args.batch_size)

    print(f"Loading Model: {args.model_name}")

    prompt_type = args.prompt_type  # "llava-prompt-1"
    print(f"Prompt Type: {prompt_type}")

    prompt = get_prompt_template(prompt_type)

    caption_generator = CaptionGenerator(
        model_name=args.model_name,
        device=args.device,
        dataloader=dataset
    )

    all_results = []

    for batch in dataloader:
        for i, query_id in enumerate(batch['query_id']):
            image_id = batch['reference_img_id'][i]
            original_caption = batch['shared_concept'][i]

            image_path = load_dataset.get_img_path(query_id)

            positive_caption = caption_generator.generate_general_image_caption(
                image_path, prompt)  # generate_image_caption(image_path, prompt)

            results = {
                'image_id': image_id,
                'positive_caption': positive_caption,
                'original_caption': original_caption,
            }

            all_results.append(results)

    outputs = {
        "dataset": dataset_name,
        "model": args.model_name,
        "prompt": prompt,
        "results": all_results
    }

    return outputs


def JourneyDB_captioning(
    args: argparse.Namespace,
    seed: int = 42,
    # min_generation_length: int = 0,
    max_generation_length: int = 5,
    num_beams: int = 3,
    length_penalty: float = 0.0,
    num_shots: int = 8,
    dataset_name: str = "JourneyDB-test",
):

    image_path = args.JourneyDB_image_folder_path
    json_test_path = args.JourneyDB_test_annotations_json_path
    # json_val_path = args.circo_val_annotations_json_path
    # json_train_path = args.circo_train_annotations_json_path

    print("Load Dataset...")

    if dataset_name == "JourneyDB-test":
        load_dataset = JourneyDBDataLoader(
            annotation_path=json_test_path,
            image_folder=image_path,
            is_testset=True
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    # print("dataset length:",load_dataset.__len__())

    if args.is_samples == 'true':
        print("sampling....")
        dataset = prepare_eval_samples(load_dataset, args.num_samples, seed=40)
    else:
        print("default no sampling.")
        dataset = load_dataset

    dataloader = get_dataloader(dataset, args.batch_size)

    print(f"Loading Model: {args.model_name}")
    # caption_generator = CaptionGenerator(model_name=args.model_name, device=args.device,dataloader=dataloader)
    caption_generator = CaptionGenerator(
        model_name=args.model_name,
        device=args.device,
        dataloader=dataset
    )

    prompt_type = args.prompt_type  # "llava-prompt-1"
    print(f"Prompt Type: {prompt_type}")

    if prompt_type == "pos-cap-prompt-2":
        SEMANTIC_QUESTIONS = {
            "Object": "What are the main objects present in the image?",
            "Attributes": "What are the key attributes or characteristics of the objects?",
            "Actions": "What actions or activities are occurring in the image?",
            "Environment": "What is the environment or setting of the image?",
            "Relations": "How are the objects or people in the image related to each other?"
        }
        prompt = SEMANTIC_QUESTIONS

    elif prompt_type == "llava-prompt-1":
        prompt = """Please analyze the image in the following five semantic aspects: [Object, Attributes, Actions, Environment, Relations].
                For each aspect, answer the corresponding question:
                - [Object]: What are the main objects present in the image?
                - [Attributes]: What are the key attributes or characteristics of the objects?
                - [Actions]: What actions or activities are occurring in the image?
                - [Environment]: What is the environment or setting of the image?
                - [Relations]: How are the objects or people in the image related to each other?
                Answer in the following format:
                - [Semantic Aspect]: Your answer"""
    all_results = []

    for batch in dataloader:
        print("Generating captions...")
        for i, image_id in enumerate(batch['image_id']):
            if dataset_name == "JourneyDB-test":
                Task3_Style_QA = batch['Task3_Style_QA'][i]
                Task3_Content_QA = batch['Task3_Content_QA'][i]
                image_path = load_dataset.get_img_path(image_id)
                positive_caption = caption_generator.generate_image_caption(
                    img_path=image_path, prompt=prompt)
                results = {
                    'image_id': image_id,
                    'positive_caption': positive_caption,
                    'Task3_Style_QA': Task3_Style_QA,
                    'Task3_Content_QA': Task3_Content_QA,
                }
                all_results.append(results)
            else:
                return ('Unsupported dataset')
        # all_results.extend(results)

    outputs = {
        "image-caption": {
            "model": args.model_name,
            "prompt": prompt,
            "results": all_results
        }
    }

    return outputs


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    main()
