# Assume in the docker container 
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9


# python -m torch.distributed.launch --master_port 21203 --nproc_per_node=8 --nnodes=1 src/tasks/run_caption_VidSwinBert_clip.py \
#   --config src/configs/VidSwinBert/msvd_8frm_clip.json \
#   --train_yaml MSVD/train_8frames.yaml \
#   --val_yaml MSVD/val_8frames.yaml \
#   --per_gpu_train_batch_size 1 \
#   --per_gpu_eval_batch_size 1 \
#   --num_train_epochs 15 \
#   --learning_rate 0.0003 \
#   --max_num_frames 8 \
#   --max_candidate_frames 8 \
#   --max_seq_a_length 77 \
#   --max_seq_length 77 \
#   --pretrained_2d 0 \
#   --backbone_coef_lr 0.05 \
#   --mask_prob 0.5 \
#   --max_masked_token 45 \
#   --zero_opt_stage 1 \
#   --mixed_precision_method deepspeed \
#   --deepspeed_fp16 \
#   --gradient_accumulation_steps 1 \
#   --img_feature_dim 512 \
#   --output_dir ./output


python -m torch.distributed.launch --master_port 24203 --nproc_per_node=8 --nnodes=1 src/tasks/run_caption_VidSwinBert_clip.py \
  --config src/configs/VidSwinBert/msvd_8frm_clip.json \
  --train_yaml MSVD/train_8frames.yaml \
  --val_yaml MSVD/val_8frames.yaml \
  --per_gpu_train_batch_size 5 \
  --per_gpu_eval_batch_size 5 \
  --num_train_epochs 15 \
  --learning_rate 0.0003 \
  --max_num_frames 8 \
  --max_candidate_frames 8 \
  --max_seq_a_length 77 \
  --max_seq_length 77 \
  --pretrained_2d 0 \
  --backbone_coef_lr 0.333 \
  --mask_prob 0.5 \
  --max_masked_token 45 \
  --zero_opt_stage 1 \
  --mixed_precision_method deepspeed \
  --deepspeed_fp16 \
  --gradient_accumulation_steps 1 \
  --img_feature_dim 512 \
  --img_res 224 \
  --output_dir ./output \
  --freeze_backbone
