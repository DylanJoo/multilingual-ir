# mdpr + monot5 rerank 
for lang in zh ar bn en es fa fi fr hi id ja ko ru sw te th;do
    rm runs/run.miracl.bm25-rerank.$lang.dev.txt 
    python3 reranking.py \
        --run runs/run.miracl.mdpr.$lang.dev.txt \
        --lang $lang \
        --topk 100 \
        --batch_size 4 \
        --max_p_length 256 \
        --max_q_length 36 \
        --output runs/run.miracl.mdpr.mt5-rerank.$lang.dev.txt \
        --model_name_or_path unicamp-dl/mt5-base-mmarco-v2 \
        --reranker_type cross-encoder \
        --gpu 2 \
        --postfix mdpr.mt5-rerank
done

# bm25 + monot5 rerank 
        # --run runs/run.miracl.bm25.$lang.dev.txt \
        # --output runs/run.miracl.bm25.mt5-rerank.$lang.dev.txt \
