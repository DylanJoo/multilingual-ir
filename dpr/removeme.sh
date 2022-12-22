export CUDA_VISIBLE_DEVICES=2
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

python3 -m pyserini.search.faiss \
  --encoder-class auto \
  --encoder castorini/mdpr-tied-pft-msmarco \
  --topics miracl-v1.0-zh-dev \
  --index miracl-v1.0-zh-mdpr-tied-pft-msmarco \
  --output run.miracl.mdpr-tied-pft-msmarco.zh.dev.txt \
  --batch 128 --threads 16 --hits 100

