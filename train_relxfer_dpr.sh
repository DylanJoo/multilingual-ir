export CUDA_VISIBLE_DEVICES=0,2
# Training using the parallel msmarco of few languages

python3 dpr/train_rel_xfer.py \
  --model_name_or_path castorini/mdpr-tied-pft-msmarco \
  --freeze_document_encoder \
  --tokenizer_name castorini/mdpr-tied-pft-msmarco \
  --config_name castorini/mdpr-tied-pft-msmarco \
  --output_dir ./checkpoints/mdpr-tied-pft-msmarco-rel-xfer \
  --train_file dataset/mmarco.400k.english.json \
  --max_q_length 64 \
  --max_d_length 256 \
  --per_device_train_batch_size 64 \
  --evaluation_strategy 'steps'\
  --max_steps 50000 \
  --save_steps 5000 \
  --eval_steps 5000 \
  --do_train \
  --do_eval

