# this is CLS pooling as same as castorini/mdpr-tied-pft-msmarco
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# see detail parameter settings in spr/search.py
if [[ "$1" == *"dev"* ]]; then
    setting=run.miracl.mdprxfer-op.lang.dev
    mkdir -p runs/$setting
    for lang in ar bn en es fa fi fr hi zh id ja ko ru sw te th;do
        python3 -m pyserini.search.faiss \
          --encoder-class auto \
          --device cuda \
          --tokenizer castorini/mdpr-tied-pft-msmarco \
          --encoder checkpoints/mdpr-tied-pft-msmarco-rel-xfer/checkpoint-3000 \
          --topics miracl-v1.0-${lang}-dev \
          --index miracl-v1.0-${lang}-mdpr-tied-pft-msmarco \
          --output runs/$setting/${setting/lang/${lang}}.txt \
          --batch 128 --threads 10 --hits 100
    done
fi

if [[ "$1" == *"test-a"* ]]; then
    setting=run.miracl.mdprxfer-op.lang.test-a
    mkdir -p runs/$setting
    for lang in ar bn en fi id ja ko ru sw te th;do
        python3 -m pyserini.search.faiss \
          --encoder-class auto \
          --device cuda \
          --tokenizer castorini/mdpr-tied-pft-msmarco \
          --encoder checkpoints/mdpr-tied-pft-msmarco-rel-xfer/checkpoint-3000 \
          --topics ./dataset/topics.miracl-v1.0-${lang}-test-a.tsv \
          --index miracl-v1.0-${lang}-mdpr-tied-pft-msmarco \
          --output runs/$setting/${setting/lang/${lang}}.txt \
          --batch 128 --threads 10 --hits 100
    done
fi
