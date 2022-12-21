export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# see detail parameter settings in spr/search.py
for lang in zh ar bn en es fa fi fr hi id ja ko ru sw te th;do
    python3 spr/search.py \
        --index miracl-v1.0-${lang} \
        --output runs/run.miracl.bm25.${lang}.dev.txt \
        --lang $lang 
done
