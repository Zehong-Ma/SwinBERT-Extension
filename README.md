# SwinBERT Extension

 

This is our research extension code for CVPR 2022 paper: [SwinBERT: End-to-End Transformers with Sparse Attention for Video Captioning](https://arxiv.org/abs/2111.13196). 

Video captioning is one of the hottest topics in multi-modal research today. Among the methods proposed in recent years, SWINBERT achieves the state of the art performance in video caption. In this project, we further examine the effectiveness of the original SWINBERT and propose the following approaches to improve it: 
1)Unlike static images, the additional temporal dimension of video delivers another important cue for semantic understanding, namely motion. In addition to the original input modality of RGB, Our method investigates another modality optical flow as the source of motion representation, and is proven to be effective. 
2)High computational complexity is the main obstacle for the practical application of the video transformer. We will investigate sparse masking strategies and propose neighborhood masked attention to reduce the redundancy in attention mechanism and improve the efficiency of the SWINBERT.
3)The attention mask is sparse enough, but the model still take as input as all dense sampled frames. So we introduce an adaptive frame sampling module for selecting sparse frames from dense samplied videos. 
4)There exists huge modal heterogeneity since the model is random initialized and trained from scratch. So it's necessary to take advantage of rich multi-modal information in vision-language pretraining model to narrow modal heterogeneity. And in practice, we introduce the popular CLIP model as the multimodal representation extractor and proposed the Video Caption CLIP(VC-CLIP) as a strong baseline. 

## Table of contents

* [Requirements](#Requirements)
* [Download](#Download)
* [Launch Docker Container](#before-running-code-launch-docker-container)
* [Quick Demo](#Quick-Demo)
* [Evaluation](#Evaluation)
* [Training](#Training)
* [License](#License)
* [Reference](#Reference)

## Requirements 
We provide a [Docker image](https://hub.docker.com/r/linjieli222/videocap_torch1.7/tags) for easier reproduction. Please install the following:
  - [nvidia driver](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#package-manager-installation) (418+), 
  - [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/) (19.03+), 
  - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-docker#quickstart).

We only support Linux with NVIDIA GPUs. We test on Ubuntu 18.04 and V100 cards.
We use mixed-precision training hence GPUs with Tensor Cores are recommended.
Our scripts require the user to have the [docker group membership](https://docs.docker.com/install/linux/linux-postinstall/)
so that docker commands can be run without sudo.

## Download

1. Create folders that store pretrained models, datasets, and predictions.
    ```bash
    export REPO_DIR=$PWD
    mkdir -p $REPO_DIR/models  # pre-trained models
    mkdir -p $REPO_DIR/datasets  # datasets
    mkdir -p $REPO_DIR/predictions  # prediction outputs
    ```

2. Download pretrained models.

    Our pre-trained models can be downloaded with the following command.
    ```bash
    cd $REPO_DIR
    bash scripts/download_models.sh
    ```
    The script will download our models that are trained for VATEX, MSRVTT, MSVD, TVC and YouCook2, respectively. It will also download our training logs and output predictions. 

    The resulting data structure should follow the hierarchy as below. 
    ```
    ${REPO_DIR}  
    |-- models  
    |   |-- table1
    |   |   |-- msvd
    |   |   |   |-- best-checkpoint
    |   |   |   |   |-- model.bin
    |   |   |   |   |-- optmizer_state.bin
    |   |   |   |   |-- pred.*
    |   |   |   |-- tokenizer
    |   |   |   |   |-- added_tokens.json
    |   |   |   |   |-- special_tokens_map.json
    |   |   |   |   |-- vocab.txt
    |   |   |   |-- log
    |   |   |   |   |-- log.txt
    |   |   |   |   |-- args.json
    |   |-- 32frm
    |   |   |-- msvd
    |   |   |   |-- best-checkpoint
    |   |   |   |   |-- model.bin
    |   |   |   |   |-- optmizer_state.bin
    |   |   |   |   |-- pred.*
    |   |   |   |-- tokenizer
    |   |   |   |   |-- added_tokens.json
    |   |   |   |   |-- special_tokens_map.json
    |   |   |   |   |-- vocab.txt
    |   |   |   |-- log
    |   |   |   |   |-- log.txt
    |   |   |   |   |-- args.json
    |-- docs 
    |-- src
    |-- scripts 
    |-- README.md 
    |-- ... 
    |-- ... 
    ```
    
3. Download pretrained Video Swin Transformers.

    To run our code smoothly, please visit [Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer) to download pre-trained weights models.

    Download `swin_base_patch244_window877_kinetics*_22k.pth`, 
    and place them under `${REPO_DIR}/models/video_swin_transformer` directory.
    The data structure should follow the hierarchy below.
    ```
    ${REPO_DIR}  
    |-- models  
    |   |-- video_swin_transformer
    |    |   |-- swin_base_patch244_window877_kinetics600_22k.pth
    |    |   |-- swin_base_patch244_window877_kinetics400_22k.pth
    |   |-- table1
    |   |-- 32frm
    |-- docs 
    |-- src
    |-- scripts 
    |-- README.md 
    |-- ... 
    |-- ... 
    ```

4. Download prediction files that were evaluated on [VALUE Leaderboard Evaluation Server](https://competitions.codalab.org/competitions/34470)

    The prediction files can be downloaded with the following command.
    ```bash
    cd $REPO_DIR
    bash scripts/download_value_preds.sh
    ```
    You could submit the prediction files to VALUE Leaderboard and reproduce our results.

5. Download datasets for training and evaluation

    In this project, we provide our pre-parsed annotation files in TSV format. To download the files, please use the following command.
    ```bash
    cd $REPO_DIR
    bash scripts/download_annotations.sh
    ```
    
    Following prior studies, we use the standard train/val/test splits for each dataset. Here, we just reorganize the data format in TSV files to better fit our codebase. 

    **Due to copyright issue, we could not release the raw videos.** We suggest downloading the orignal raw videos from the official dataset websites. Please place the downloaded videos under `raw_videos` or `videos` of each dataset folder. 

    The `datasets` directory structure should follow the below hierarchy.
    ```
    ${ROOT}  
    |-- datasets  
    |   |-- MSVD  
    |   |   |-- *.yaml 
    |   |   |-- *.tsv  
    |   |   |-- videos <<< please place the downloaded videos under this folder 
    |   |   |   |-- *.avi 
    folder 
    |   |   |   |-- *.mp4 
    |   |   |-- testing <<< please place the downloaded testing videos under this folder 
    |   |   |   |-- *.mp4 
    |-- docs
    |-- src
    |-- scripts
    |-- models 
    |-- README.md 
    |-- ... 
    |-- ... 
    
    ```
    
    We also provide example scripts to reproduce our annotation tsv files. You may find the examples below.
    ```
    ${ROOT}  
    |-- prepro  
    |   |-- tsv_preproc_vatex.py
    |   |-- tsv_preproc_msrvtt.py
    |   |-- tsv_preproc_msvd.py
    |   |-- tsv_preproc_tvc.py
    |   |-- tsv_preproc_youcook2.py
    |-- docs
    |-- src
    |-- scripts
    |-- README.md 
    |-- ... 
    |-- ... 
    
    ```


## Before Running Code: Launch Docker Container 

We provide a [Docker image](https://hub.docker.com/r/linjieli222/videocap_torch1.7/tags) for easier reproduction. Please launch the docker container before running our codes. 

```bash
export REPO_DIR=$PWD
DATASETS=$REPO_DIR'/datasets/'
MODELS=$REPO_DIR'/models/'
OUTPUT_DIR=$REPO_DIR'/output/'
source launch_container.sh $DATASETS $MODELS $OUTPUT_DIR
```

Our latest docker image `linjieli222/videocap_torch1.7:fairscale` supports the following mixed precision training
- [x] Torch.amp (with limited GPU memory optimization, deprecated from this codebase)
- [x] Nvidia Apex O2
- [x] deepspeed (Best setting on VATEX, deepspeed fp16 with zero_opt_stage=1)
- [x] fairscale


## Quick Demo
We provide a demo to run end-to-end inference on the test video.

Our inference code will take a video as input, and generate video caption.

```bash
# After launching the docker container 
EVAL_DIR='./models/table1/vatex/best-checkpoint/'
CHECKPOINT='./models/table1/vatex/best-checkpoint/model.bin'
VIDEO='./docs/G0mjFqytJt4_000152_000162.mp4'
CUDA_VISIBLE_DEVICES=0 python src/tasks/run_caption_VidSwinBert_inference.py \
       --resume_checkpoint $CHECKPOINT  \
       --eval_model_dir $EVAL_DIR \
       --test_video_fname $VIDEO \
       --do_lower_case \
       --do_test 
```

The prediction should look like

```bash
Prediction: a young boy is showing how to make a paper airplane.
```

## Evaluation

We provide example scripts to evaluate pre-trained checkpoints

### MSVD

```bash
# Assume in the docker container 
EVAL_DIR='./models/table1/msvd/best-checkpoint/'
CUDA_VISIBLE_DEVICES=0 python src/tasks/run_caption_VidSwinBert.py \
       --val_yaml MSVD/val_32frames.yaml  \
       --do_eval true \
       --do_train false \
       --eval_model_dir $EVAL_DIR
```
For online decoding, please use `MSVD/val.yaml`
For offline decoding, please use  `MSVD/val_32frames.yaml`

## Training

We provide example scripts to train our model (with 32-frame inputs, soft sparse attention)

### MSVD
```bash
# Assume in the docker container 
python src/tasks/run_caption_VidSwinBert.py
        --config src/configs/VidSwinBert/msvd_8frm_default.json
        --train_yaml MSVD/train_32frames.yaml
        --val_yaml MSVD/val_32frames.yaml
        --per_gpu_train_batch_size 6
        --per_gpu_eval_batch_size 6
        --num_train_epochs 15
        --learning_rate 0.0003
        --max_num_frames 32
        --pretrained_2d 0
        --backbone_coef_lr 0.05
        --mask_prob 0.5
        --max_masked_token 45
        --zero_opt_stage 1
        --mixed_precision_method deepspeed
        --deepspeed_fp16
        --gradient_accumulation_steps 1
        --learn_mask_enabled
        --loss_sparse_w 0.5
        --output_dir ./output
```
## License

Our research code is released under MIT license.

## Reference

[SwinBERT](https://github.com/microsoft/SwinBERT)

