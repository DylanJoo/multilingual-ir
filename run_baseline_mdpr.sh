export CUDA_VISIBLE_DEVICES=2
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# see detail parameter settings in spr/search.py
for lang in ar bn en es fa fi fr hi zh id ja ko ru sw te th;do
    python3 -m pyserini.search.faiss \
      --encoder-class auto \
      --encoder castorini/mdpr-tied-pft-msmarco \
      --topics miracl-v1.0-${lang}-dev \
      --index miracl-v1.0-${lang}-mdpr-tied-pft-msmarco \
      --output runs/run.miracl.mdpr.${lang}.dev.txt \
      --batch 128 --threads 16 --hits 100
done
