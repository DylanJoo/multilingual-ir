export CUDA_VISIBLE_DEVICES=1,2
# Training using the parallel msmarco of few languages

python3 dpr/train_rel_xfer.py \
  --model_name_or_path castorini/mdpr-tied-pft-msmarco \
  --freeze_document_encoder \
  --tokenizer_name castorini/mdpr-tied-pft-msmarco \
  --config_name castorini/mdpr-tied-pft-msmarco \
  --output_dir ./checkpoints/mdpr-tied-pft-msmarco-rel-xfer-distill \
  --train_file dataset/mmarco.400k.english.json \
  --language_relxfer 'distill' \
  --max_q_length 64 \
  --max_d_length 256 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --evaluation_strategy 'steps'\
  --max_steps 10000 \
  --save_steps 1000 \
  --eval_steps 1000 \
  --do_train \
  --do_eval

