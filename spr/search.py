import os
import json
from tqdm import tqdm
import argparse
from pyserini.search.lucene import LuceneSearcher
from datasets import load_dataset

def searching(args):
    # Lucuene initialization
    searcher = LuceneSearcher.from_prebuilt_index(args.index)
    searcher.set_bm25(k1=args.k1, b=args.b)
    searcher.set_language(args.lang)

    # Load query 
    dataset=load_dataset('miracl/miracl', args.lang)[args.subset]
    qid=dataset['query_id']
    qtext=dataset['query']

    # Prepare a trec output 
    output = open(args.output, 'w')

    # search for each q
    for iq in tqdm(range(len(qid))):
        index=qid[iq]
        text=qtext[iq].strip()
        hits = searcher.search(text, k=args.k)
        for i in range(len(hits)):
            output.write(f'{index} Q0 {hits[i].docid:4} {i+1} {hits[i].score:.5f} lucene.k.82.b1.68\n')

    # trec ouptut done
    output.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--k", default=1000, type=int)
    parser.add_argument("-k1", "--k1", type=float, default=0.82)
    parser.add_argument("-b", "--b", type=float, default=0.68)
    parser.add_argument("-index", "--index", type=str, default='miracl-v1.0-zh')
    parser.add_argument("-output", "--output", default='runs/run.miracl.bm25.zh.dev.txt', type=str)
    # special args
    parser.add_argument("-lang", "--lang", type=str, default='zh')
    parser.add_argument("-subset", "--subset", type=str, default='dev')
    args = parser.parse_args()

    os.makedirs('runs', exist_ok=True)
    searching(args)
