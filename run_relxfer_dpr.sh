# this is CLS pooling as same as castorini/mdpr-tied-pft-msmarco
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# see detail parameter settings in spr/search.py
for lang in ar bn en es fa fi fr hi zh id ja ko ru sw te th;do
    python3 -m pyserini.search.faiss \
      --encoder-class auto \
      --device cuda:1 \
      --tokenizer castorini/mdpr-tied-pft-msmarco \
      --encoder checkpoints/mdpr-tied-pft-msmarco-rel-xfer-vanilla/checkpoint-1000 \
      --topics miracl-v1.0-${lang}-dev \
      --index miracl-v1.0-${lang}-mdpr-tied-pft-msmarco \
      --output runs/run.miracl.mdprxfer.${lang}.dev.txt \
      --batch 128 --threads 4 --hits 100
done
