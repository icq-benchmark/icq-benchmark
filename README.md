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
├── dataset
│   ├── images
│   │   ├── val_style_cartoon
│   │   ├── val_style_cinematic
│   │   ├── val_style_realistic
│   │   └── val_style_scribble
│   ├── data
│   │   └── icq_highlight_release.jsonl
├── exps_use
│   ├── image2caption # results of image captioning 
│   │   ├── caption_style_cinematic.jsonl 
│   │   ├── ...
│   ├── captioning # results of captioning
│   │   ├── adjusted_caption_style_cinematic_cap.jsonl
│   │   ├── ...
│   └── summarization # results of captioning
│   │   ├── adjusted_caption_style_cinematic_sum.jsonl
│   │   ├── ...
├── features
│   ├── clip_features 
│   ├── slowfast_features 
│   ├── clip_text_features 
│   ├── pann_features
│   ├── clip_text_features_cinematic_cap # text features of style cinematic for captioning
│   ├── clip_text_features_cinematic_sum # text features of style cinematic for summarization
│   ├── clip_image_features_cinematic # image features of style cinematic for visual query
│   ├── ...
```


### Download Datasets
Our dataset and annotation files can be found at [here](https://huggingface.co/datasets/gengyuanmax/ICQ-Highlight).
For clip_features, slowfast_features, clip_text_features, raw videos, 
please refer to the link provided in [Moment-DETR](https://github.com/jayleicn/moment_detr).
For pann_features, please find in [UMT](https://github.com/TencentARC/UMT).

### Captioning adaptation method
1. Use MLLM to perform image captioning

Follow the instruction in [LLaVA](https://github.com/haotian-liu/LLaVA) for the setup of LLaVA. 
```python
python3 img2caption.py --image_dir dataset/images/val_style_cinematic/ --des_path exps_use/image_captioning/caption_style_cinematic
```
2. use LLM as a modifier to integrate refinement texts

We define the ```ref_img_cap_file``` as the image captioning file you got in the first step, 
```gt_file``` as the ground truth file which provides the refinement text types and details, 
```des_path``` as the path the file to be saved, lastly, ```is_scribble``` tells whether the reference image style is scribble or not.
```python
# for general reference image styles
python3 refine_text.py --ref_img_cap_file exps_use/image2caption/caption_style_cinematic.jsonl --gt_file data/icq_highlight_release_style_cinematic.jsonl --des_path exps_use/captioning/adjusted_caption_style_cinematic_cap
# for scribble image style
python3 refine_text.py --ref_img_cap_file exps_use/image2caption/caption_style_scribble.jsonl --gt_file data/icq_highlight_release.jsonl --des_path exps_use/captioning/adjusted_caption_style_scribble_cap --is_scribble
```

### Summarization adaptation method
```python
# for general reference image styles
python3 sum_img_text.py --gt_file dataset/data/icq_highlight_release_style_cinematic.jsonl --image_dir dataset/images/val_style_cinematic/ --des_path exps_use/summarization/adjusted_caption_style_cinematic_sum --style cinematic
# for scribble image style
python3 sum_img_text.py --gt_file dataset/data/icq_highlight_release.jsonl --image_dir dataset/images/val_style_scribble/ --des_path exps_use/summarization/adjusted_caption_style_scribble_sum --style scribble
```

### Feature Extraction on ICQ Dataset's queries
Encode refined texts to CLIP features.
```python
# encode text
python3 encode.py --text_file exps_use/captioning/adjusted_caption_style_cinematic_cap.jsonl --des_dir features/clip_text_features_cinematic_cap --src_type text
# encode image (as visual query)
python3 encode.py --image_dir data/images/val_style_cinematic --des_dir features/clip_image_features_cinematic --src_type image
```

## Benchmarking
The inferences are evaluated on the methods below. Please follow the instructions in their repo. You will need to adapt our annotation file and newly encoded features. The following includes the checkpoints used as well.

| Methods                                               | Source Type |  Checkpoints                                                                                                                    |
|-------------------------------------------------------|-------------|---------------------------------------------------------------------------------------------------------------------------------|
| [Moment-DETR](https://github.com/jayleicn/moment_detr)| V           | Please check Moment-DETR/run_on_video/moment_detr_ckpt                                                              |
| [QD-DETR](https://github.com/wjun0830/QD-DETR)        | V           | QD-DETR (Video only) Checkpoint                                                                                                 |
| [QD-DETR](https://github.com/wjun0830/QD-DETR)        | V+A         | QD-DETR (Video+Audio) Checkpoint                                                                                                |
| [EaTR](https://github.com/jinhyunj/EaTR)              | V           | Please check EaTR/eatr_ckpt                                                                                                     |
| [CG-DETR](https://github.com/wjun0830/CGDETR)         | V           | Please check QVHighlights (under Model Zoo) in their README                                                                     |
| [TR-DETR](https://github.com/mingyao1120/TR-DETR)     | V           | Please check TR-DETR/checkpoint                                                                                                 |
| [UMT](https://github.com/TencentARC/UMT)              | V+A         | Please check QVHighlights UMT-B (under Model Zoo) in their README                                                               |
| [UniVTG](https://github.com/showlab/UniVTG)           | V           | Please check QVHL (Joint Moment Retrieval and Highlight Detection) [here](https://github.com/showlab/UniVTG/blob/main/model.md) |
| [UVCOM](https://github.com/EasonXiao-888/UVCOM)       | V           | Please check QVHighlights Val (under Main Results) in their README                                                              |
| [SeViLA](https://github.com/Yui010206/SeViLA)         | V           | Please follow their instructions (under Download Pretrained Models) in their README                                             |

## Acknowledgement
This repo is built based on [Moment-DETR](https://github.com/jayleicn/moment_detr). Benchmarking were made on [Moment-DETR](https://github.com/jayleicn/moment_detr),
[QD-DETR](https://github.com/wjun0830/QD-DETR), [EaTR](https://github.com/jinhyunj/EaTR),
[CG-DETR](https://github.com/wjun0830/CGDETR), [TR-DETR](https://github.com/mingyao1120/TR-DETR),
[UMT](https://github.com/TencentARC/UMT), [UniVTG](https://github.com/showlab/UniVTG),
[UVCOM](https://github.com/EasonXiao-888/UVCOM), [SeViLA](https://github.com/Yui010206/SeViLA). We thank all the authors for their great open-source contributions!

## Citation
Please cite our paper if you find this repo useful in your research:
```python
TBD
```

