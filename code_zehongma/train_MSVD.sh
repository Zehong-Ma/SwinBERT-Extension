# Assume in the docker container 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9


# python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 src/tasks/run_caption_VidSwinBert.py \
#   --config src/configs/VidSwinBert/msvd_8frm_default.json \
#   --train_yaml MSVD/train_64frames.yaml \
#   --val_yaml MSVD/val_64frames.yaml \
#   --per_gpu_train_batch_size 1 \
#   --per_gpu_eval_batch_size 1 \
#   --num_train_epochs 15 \
#   --learning_rate 0.0003 \
#   --max_num_frames 32 \
#   --max_candidate_frames 64 \
#   --pretrained_2d 0 \
#   --backbone_coef_lr 0.05 \
#   --mask_prob 0.5 \
#   --max_masked_token 45 \
#   --zero_opt_stage 1 \
#   --mixed_precision_method deepspeed \
#   --deepspeed_fp16 \
#   --gradient_accumulation_steps 1 \
#   --learn_mask_enabled \
#   --loss_sparse_w 0.5 \
#   --output_dir ./output \
#   --transfer_method 0 \
#   --pretrained_checkpoint ./models/32frm/MSVD/best-checkpoint/ \
#   --use_sample_net

python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 src/tasks/run_caption_VidSwinBert.py \
  --config src/configs/VidSwinBert/msvd_8frm_default.json \
  --train_yaml MSVD/train_32frames.yaml \
  --val_yaml MSVD/val_32frames.yaml \
  --per_gpu_train_batch_size 2 \
  --per_gpu_eval_batch_size 2 \
  --num_train_epochs 5 \
  --learning_rate 0.00001 \
  --max_num_frames 8 \
  --max_candidate_frames 32 \
  --pretrained_2d 0 \
  --backbone_coef_lr 0.05 \
  --mask_prob 0.5 \
  --max_masked_token 45 \
  --zero_opt_stage 1 \
  --mixed_precision_method deepspeed \
  --deepspeed_fp16 \
  --gradient_accumulation_steps 1 \
  --learn_mask_enabled \
  --loss_sparse_w 0.5 \
  --output_dir ./output \
  --use_sample_net \
  --transfer_method 0 \
  --pretrained_checkpoint ./output/checkpoint-15-13050-baseline/ \
