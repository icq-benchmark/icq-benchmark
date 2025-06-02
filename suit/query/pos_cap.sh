#!/usr/bin/bash
#SBATCH --gres=gpu:1

python run_pos_cap.py \
    --tokenizer_path "lmsys/vicuna-7b-v1.1" \
    --lang_encoder_path "lmsys/vicuna-7b-v1.1" \
    --max_length 128 \
    --seed 42 \
    --model_name "llava" \
    --dataset 'COCO' \
    --batch_size 10 \
    --device "cuda:0" \
    --num_samples 200 \
    --num_subsets 5 \
    --is_samples "false"\
    --prompt_type "pos-cap-prompt" \
    --coco_dataset \
    --results_file "output/pos-caption/COCO/save" \
    --db_path "output/pos-caption/COCO/subset/" \
    --circo_image_folder_path 'data/circo/COCO2017_unlabeled/unlabeled2017' \
    --circo_train_annotations_json_path 'data/circo/COCO2017_unlabeled/annotations/image_info_unlabeled2017.json' \
    --circo_val_annotations_json_path 'data/circo/annotations/val.json' \
    --circo_test_annotations_json_path 'data/circo/annotations/test.json' \
    --JourneyDB_image_folder_path 'data/JourneyDB/test/imgs' \
    --JourneyDB_test_annotations_json_path 'data/JourneyDB/test/test_questions.jsonl' \
    --gqa_image_folder_path "data/GQA/images" \
    --coco_train_image_dir_path "data/COCO/train2017" \
    --coco_val_image_dir_path "data/COCO/val2017" \
    --coco_train_annotations_json_path "data/COCO/annotations/captions_train2017.json" \
    --coco_val_annotations_json_path "data/COCO/annotations/captions_val2017.json" \