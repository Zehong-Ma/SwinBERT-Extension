# EVAL_DIR='./output/checkpoint-15-13050-sample/'
# EVAL_DIR='./models/32frm/MSVD/best-checkpoint/'
# EVAL_DIR='./models/table1/msvd/best-checkpoint/'
EVAL_DIR="./output/checkpoint-5-3810/"
CUDA_VISIBLE_DEVICES=0 python src/tasks/run_caption_VidSwinBert.py \
       --val_yaml MSVD/test_32frames.yaml  \
       --do_eval true \
       --do_train false \
       --max_num_frames 8 \
       --max_candidate_frames 32 \
       --sparse_mask_soft2hard \
       --eval_model_dir $EVAL_DIR