for lang in zh ar bn en es fa fi fr hi id ja ko ru sw te th;do
    rm runs/run.miracl.bm25-rerank.$lang.dev.txt 
    python3 zeroshot_rerank.py \
        --run runs/run.miracl.bm25.$lang.dev.txt \
        --lang $lang \
        --topk 100 \
        --batch_size 8 \
        --output runs/run.miracl.bm25-rerank.$lang.dev.txt \
        --model_name_or_path bert-base-multilingual-cased \
        --gpu 2 \
        --postfix bm25.mbert-rerank
done
