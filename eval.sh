export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

echo '| Language | Ranking | R@100  | nDCG@10 |'
echo '|----------|---------|--------|---------|'

for lang in ar bn en es fa fi fr hi id ja ko ru sw te th zh;do
# for lang in zh bn en;do
    # echo -n $lang '| BM25 |';
    # run=runs/run.miracl.bm25.lang.dev/run.miracl.bm25.$lang.dev.txt
    # qrel=/home/jhju/.cache/pyserini/topics-and-qrels/qrels.miracl-v1.0-${lang}-dev.tsv
    # ./trec_eval-9.0.7/trec_eval -c \
    #     -m recall.100 \
    #     -m ndcg_cut.10 $qrel $run | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'

    echo -n $lang '| mDPR |';
    run=runs/run.miracl.mdpr.lang.dev/run.miracl.mdpr.$lang.dev.txt
    qrel=/home/jhju/.cache/pyserini/topics-and-qrels/qrels.miracl-v1.0-${lang}-dev.tsv
    ./trec_eval-9.0.7/trec_eval -c \
        -m recall.100 \
        -m ndcg_cut.10 $qrel $run | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'

    echo -n $lang '| mDPR-xfer-vanilla (1K) |';
    run=runs/run.miracl.mdprxfer-few.lang.dev/run.miracl.mdprxfer-few.$lang.dev.txt
    qrel=/home/jhju/.cache/pyserini/topics-and-qrels/qrels.miracl-v1.0-${lang}-dev.tsv
    ./trec_eval-9.0.7/trec_eval -c \
        -m recall.100 \
        -m ndcg_cut.10 $qrel $run | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'

    echo -n $lang '| mDPR-xfer-vanilla (3K)|';
    run=runs/run.miracl.mdprxfer-med.lang.dev/run.miracl.mdprxfer-med.$lang.dev.txt
    qrel=/home/jhju/.cache/pyserini/topics-and-qrels/qrels.miracl-v1.0-${lang}-dev.tsv
    ./trec_eval-9.0.7/trec_eval -c \
        -m recall.100 \
        -m ndcg_cut.10 $qrel $run | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'

    echo -n $lang '| mDPR-xfer-vanilla (10K) |';
    run=runs/run.miracl.mdprxfer-all.lang.dev/run.miracl.mdprxfer-all.$lang.dev.txt
    qrel=/home/jhju/.cache/pyserini/topics-and-qrels/qrels.miracl-v1.0-${lang}-dev.tsv
    ./trec_eval-9.0.7/trec_eval -c \
        -m recall.100 \
        -m ndcg_cut.10 $qrel $run | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'

    echo -n $lang '| mDPR-xfer-op (1K) |';
    run=runs/run.miracl.mdprxfer-op-few.lang.dev/run.miracl.mdprxfer-op-few.$lang.dev.txt
    qrel=/home/jhju/.cache/pyserini/topics-and-qrels/qrels.miracl-v1.0-${lang}-dev.tsv
    ./trec_eval-9.0.7/trec_eval -c \
        -m recall.100 \
        -m ndcg_cut.10 $qrel $run | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'

    echo -n $lang '| mDPR-xfer-op (3K) |';
    run=runs/run.miracl.mdprxfer-op-med.lang.dev/run.miracl.mdprxfer-op-med.$lang.dev.txt
    qrel=/home/jhju/.cache/pyserini/topics-and-qrels/qrels.miracl-v1.0-${lang}-dev.tsv
    ./trec_eval-9.0.7/trec_eval -c \
        -m recall.100 \
        -m ndcg_cut.10 $qrel $run | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'

    echo -n $lang '| mDPR-xfer-op (10K) |';
    run=runs/run.miracl.mdprxfer-op.lang.dev/run.miracl.mdprxfer-op.$lang.dev.txt
    qrel=/home/jhju/.cache/pyserini/topics-and-qrels/qrels.miracl-v1.0-${lang}-dev.tsv
    ./trec_eval-9.0.7/trec_eval -c \
        -m recall.100 \
        -m ndcg_cut.10 $qrel $run | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'

    # echo  $lang '| mDPR-xfer-med |'
    # run=runs/run.miracl.mdprxfer-med.lang.dev/run.miracl.mdprxfer-med.$lang.dev.txt
    # qrel=/home/jhju/.cache/pyserini/topics-and-qrels/qrels.miracl-v1.0-${lang}-dev.tsv
    # ./trec_eval-9.0.7/trec_eval -c \
    #     -m recall.100 \
    #     -m ndcg_cut.10 $qrel $run | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'
    #
    # echo  $lang '| mDPR + monoMT5 |'
    # run=runs/run.miracl.mdpr.mt5-rerank.lang.dev/run.miracl.mdpr.mt5-rerank.$lang.dev.txt
    # qrel=/home/jhju/.cache/pyserini/topics-and-qrels/qrels.miracl-v1.0-${lang}-dev.tsv
    # ./trec_eval-9.0.7/trec_eval -c \
    #     -m recall.100 \
    #     -m ndcg_cut.10 $qrel $run | cut -f3 | sed ':a; N; $!ba; s/\n/|/g'

done
