# SUIT Strategy

## Summary
- We prepare pseudo multimodal query dataset
- We fine-tune LLaVA for adaption

## DATASET: Pseudo Multimodel Query Generation

### Introduction
To prepare the necessary dataset, we:
- generate semantic-enriched queries
- generate refinement text
- post-process queries

### Code Structure
```
query
├── scripts                      # store helper functions
│   ├── ImageCaptionDataset.py   # data loader for ImgCap Datasets: COCO, Flickr30k, CIRCO, ...
│   ├── models.py                # load models
│   ├── api_generate.py          # GPT for neg cap
│   ├── positive_caption.py      # LLaVA for pos cap
│   ├── post_processing.py       # ROBERTA & CLIP
├── pos_cap.sh
├── run_pos_cap.py               # generate semantic-enriched query
├── neg_cap.sh
├── run_neg_cap.py               # generate refinement text
├── post_process.sh       
├── run_post_process.py          # post-processing
```

## FINE-TUNING: VLM PEFT

Adopted from https://github.com/haotian-liu/LLaVA.

### Installation
1. Navigate to LLaVA folder
```bash
cd LLaVA
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

4. Training
```shell
sbatch scripts/lora/finetune_lora_suit.sh 
```