# Localizing Events in Videos with Multimodal Queries

This repository contains the code for the paper Localizing Events in Videos with Multimodal Queries. We introduce a new
benchmark, ICQ, for localizing events in videos with multimodal queries, along  with a new evaluation dataset ICQ-Highlight. Our new benchmark aims to evaluate
how well models can localize an event given a multimodal semantic query that
consists of a reference image, which depicts the event, and a refinement text to
adjust the images’ semantics. Concretely, we include 4 styles of reference images and 5 types of refinement texts, allowing us
to explore model performance across different domains. We propose 3 adaptation
methods that tailor existing models to our new setting and evaluate 10 SOTA
models, ranging from specialized to large-scale foundation models

## Installation
```python
# git clone
git clone https://github.com/icq-benchmark/icq-benchmark.git
cd icq-benchmark
# create conda environment (optional)
conda create -n icq-benchmark python=3.9
conda activate icq-benchmark
# install pytorch with CUDA 11.0
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
# install requirements
pip install -r requirements.txt
```

## Code Structure

```
icq-benchmark
├── scripts                      # update soon
├── clip                         # CLIP module for feature extraction borrowed from Moment-DETR
├── encode.py                    # extract either text or image features
├── img2caption.py               # perform image captioning on reference images
├── refine_text.py               # captioning adaptation method
├── sum_img_text.py              # summarization adaptation method
├── utils.py                     # utility functions used across various files
```

## Data Preparation
### Default Directory Structure
```
├── datasets
│   ├── images
│   │   ├── val_style_0 # cinematic
│   │   ├── val_style_1 # cartoon
│   │   ├── val_style_2 # realistic
│   │   └── val_style_3 # scribble
│   ├── data
│   │   ├── highlight_val_modified_style_0.jsonl # cinematic
│   │   ├── highlight_val_modified_style_1.jsonl # cartoon
│   │   ├── highlight_val_modified_style_2.jsonl # realistic
│   │   └── highlight_val_modified.jsonl # scribble
├── exps_use
│   ├── image_captioning # results of image captioning 
│   │   ├── caption_style_0.jsonl 
│   │   ├── ...
│   ├── captioning # results of captioning
│   │   ├── adjusted_caption_style_0_cap.jsonl
│   │   ├── ...
│   └── summarization # results of captioning
│   │   ├── adjusted_caption_style_0_sum.jsonl
│   │   ├── ...
├── features
│   ├── clip_features 
│   ├── slowfast_features 
│   ├── clip_text_features 
│   ├── pann_features
│   ├── clip_text_features_s0_cap # text features of style 0 for captioning
│   ├── clip_text_features_s0_sum # text features of style 0 for summarization
│   ├── clip_image_features_s0 # image features of style 0 for visual query
│   ├── ...
```


### Download Datasets
For clip_features, slowfast_features, clip_text_features, raw videos, 
please refer to the link provided in [Moment-DETR](https://github.com/jayleicn/moment_detr).
For pann_features, please find in [UMT](https://github.com/TencentARC/UMT).

### Captioning adaptation method
1. Use MLLM to perform image captioning

Follow the instruction in [LLaVA](https://github.com/haotian-liu/LLaVA) for the setup of LLaVA. 
```python
python3 img2caption.py --image_dir datasets/images/val_style_0/ --des_path exps_use/image_captioning/caption_style_0 --style cinematic
```
2. use LLM as a modifier to integrate refinement texts

We define the ```ref_img_cap_file``` as the image captioning file you got in the first step, 
```gt_file``` as the ground truth file which provides the refinement text types and details, 
```des_path``` as the path the file to be saved, lastly, ```is_scribble``` tells whether the reference image style is scribble or not.
```python
# for general reference image styles
python3 refine_text.py --ref_img_cap_file exps_use/image_captioning/caption_style_0.jsonl --gt_file data/highlight_val_modified_style_0.jsonl --des_path exps_use/LLM_adjusted/adjusted_caption_style_0_cap
# for scribble image style
python3 refine_text.py --ref_img_cap_file exps_use/image_captioning/caption_style_3.jsonl --gt_file data/highlight_val_modified.jsonl --des_path exps_use/LLM_adjusted/adjusted_caption_style_3_cap --is_scribble
```

### Summarization adaptation method
```python
# for general reference image styles
python3 sum_img_text.py --gt_file data/highlight_val_modified_style_0.jsonl --image_dir datasets/images/val_style_0/ --des_path exps_use/summarization/adjusted_caption_style_0_sum --style cinematic
# for scribble image style
python3 sum_img_text.py --gt_file data/highlight_val_modified.jsonl --image_dir datasets/images/val_style_3/ --des_path exps_use/summarization/adjusted_caption_style_3_sum --style scribble
```

### Feature Extraction on ICQ Dataset's queries
Encode refined texts to CLIP features.
```python
# encode text
python3 encode.py --text_file exps_use/LLM_adjusted/adjusted_caption_style_0_cap.jsonl --des_dir features/clip_text_features_s0_cap --src_type text
# encode image (as visual query)
python3 encode.py --image_dir data/qimages/val_style_0 --des_dir features/clip_image_features_s0 --src_type image
```

## Benchmarking Scripts
We will update soon.

## Acknowledgement
This repo is built based on the following repos:

## Citation
Please cite our paper if you find this repo useful in your research:

