#!/usr/bin/bash
#SBATCH --gres=gpu:1

python run_post_process.py \
--process_negative \
--model_name "roberta" \
--batch_size 10 \
--device "cpu" \
--num_samples 20 \
--is_samples "true" \
--results_file_for_cap_pairs "output/post-process/negative_" \
--results_file_for_pos_cap "output/post-process/positive_" \
--pos_neg_pairs_json_path "output/neg-caption/COCO/neg-cap-prompt.json" \
--pos_json_path "output/pos-caption/CIRCO/pos-cap-prompt.json" \
--image_folder_path ""