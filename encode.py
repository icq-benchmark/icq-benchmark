from utils import ClipFeatureExtractor
import argparse
import numpy as np
import os
import glob
from utils import load_jsonl


def _get_qid_query(list_of_dicts):
    result_list = []
    for d in list_of_dicts:
        if "qid" in d and "query" in d:
            result_list.append((d["qid"], d["query"]))
    return result_list


def encode(data, new_folder, scr_type):
    clip_len = 2
    model_name = "ViT-B/32"
    device = "cpu"
    feature_extractor = ClipFeatureExtractor(framerate=1/clip_len, size=224,
                                             centercrop=True,model_name_or_path=model_name, device=device)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        print(f'{new_folder} is created.')

    if scr_type == "text":
        for qid_id, query in data:
            text_embeddings = feature_extractor.encode_text([query])
            text_embeddings = text_embeddings[0].numpy().astype(np.float16)

            np.savez(f'{new_folder}/qid{qid_id}.npz', last_hidden_state=text_embeddings)
            print(f'qid{qid_id}.npz is saved.')
    else:
        for path in data:
            image_embeddings = feature_extractor.encode_image(path)
            image_embeddings = image_embeddings[0].numpy().astype(np.float16)
            qid_id = path.split("/")[-1].split('_')[0].split('qid')[1]
            np.savez(f'{new_folder}/qid{qid_id}.npz',
                     last_hidden_state=image_embeddings)
            print(f'qid{qid_id}.npz is saved.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Feature Extractions")
    parser.add_argument('--text_file', type=str, required=False,
                        help="Path to the refined captions file.")
    parser.add_argument('--image_dir', type=str, required=False,
                        help="Path to the reference image folder.")
    parser.add_argument('--des_dir', type=str, required=True,
                        help="Path to the destination folder")
    parser.add_argument('--src_type', type=str, required=True,
                        help="Indicate the type of source: text or image")

    args = parser.parse_args()

    if args.src_type == "text":
        json_file = load_jsonl(args.text_file)
        text_lists = _get_qid_query(json_file)
        encode(text_lists, args.des_dir, args.src_type)
    else:
        image_paths = glob.glob(f"{args.image_dir}/*.jpg")
        encode(image_paths, args.des_dir, args.src_type)
    print("Done.")