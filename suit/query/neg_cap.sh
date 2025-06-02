#!/usr/bin/bash
#SBATCH --gres=gpu:1

python run_neg_cap.py \
--model_name "gpt-35-turbo-16k" \
--device "cpu" \
--results_file "output/neg-caption/coco/gpt_" \
--db_path "output/neg-caption/coco/" \
--dataset "coco" \
--use_chatgpt \
--random_id $$ \
--batch_size 10 \
--num_samples 20 \
--max_length 128 \
--seed 42 \
--prompt_type "neg-cap-prompt" \
--is_samples "false" \
--num_subsets 20 \
--coco_pos_cap_json_path "output/pos-caption/COCO/COCO_captions.json" \
--flickr30k_annotations_csv_path 'data/flickr30k/flickr_annotations_30k.csv' \
--flickr30k_image_dir_path 'data/flickr30k/flickr30k-images'
