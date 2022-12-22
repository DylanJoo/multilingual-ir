export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

for lang in ar bn en es fa fi fr hi id ja ko ru sw te th zh;do
    run=runs/run.miracl.bm25.$lang.dev.txt
    qrel=/home/jhju/.cache/pyserini/topics-and-qrels/qrels.miracl-v1.0-${lang}-dev.tsv
    ./trec_eval-9.0.7/trec_eval -c \
        -m recall.100 \
        -m ndcg_cut.10 $qrel $run | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'

    run=runs/run.miracl.mdpr.$lang.dev.txt
    qrel=/home/jhju/.cache/pyserini/topics-and-qrels/qrels.miracl-v1.0-${lang}-dev.tsv
    ./trec_eval-9.0.7/trec_eval -c \
        -m recall.100 \
        -m ndcg_cut.10 $qrel $run | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'
done
