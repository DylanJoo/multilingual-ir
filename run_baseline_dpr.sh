export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

if [[ "$1" == "dev" ]]; then
    # see detail parameter settings in spr/search.py
    for lang in ar bn en es fa fi fr hi zh id ja ko ru sw te th;do
        python3 -m pyserini.search.faiss \
          --encoder-class auto \
          --device cuda:1 \
          --encoder castorini/mdpr-tied-pft-msmarco \
          --topics miracl-v1.0-${lang}-dev \
          --index miracl-v1.0-${lang}-mdpr-tied-pft-msmarco \
          --output runs/run.miracl.mdpr.${lang}.dev.txt \
          --batch 128 --threads 16 --hits 100
    done
fi

if [[ "$1" == "test-a" ]]; then
    for lang in ar bn en fi id ja ko ru sw te th;do
        python3 -m pyserini.search.faiss \
          --encoder-class auto \
          --device cuda:1 \
          --encoder castorini/mdpr-tied-pft-msmarco \
          --topics ./dataset/topics.miracl-v1.0-${lang}-test-a.tsv \
          --index miracl-v1.0-${lang}-mdpr-tied-pft-msmarco \
          --output runs/run.miracl.mdpr.${lang}.test-a.txt \
          --batch 128 --threads 4 --hits 100
    done
fi
