import torch
from tqdm import tqdm
import argparse
import collections
from torch.utils.data import DataLoader
from datasets import Dataset
from tools.loader import get_miracl_query, get_miracl_corpus, load_runs
from tools.utils import chunk_list
from datacollator import DataCollatorForDPR
from rerank.encoder import BertEncoder

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Load triplet
    parser.add_argument("--run", type=str, required=True,)
    parser.add_argument("--lang", type=str, required=True,)
    # Reranking conditions
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default='bert-base-uncased')
    parser.add_argument("--gpu", type=str, default='2')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--postfix", type=str, default='monoMT5')
    args = parser.parse_args()

    fout = open(args.output, 'w')

    # load model
    enc = BertEncoder(args.model_name_or_path, device=f'cuda:{args.gpu}')

    # load triplet
    queries = get_miracl_query(args.lang, 'dev')
    corpus = get_miracl_corpus(args.lang)
    runs = load_runs(args.run, False)

    # prepare dataset 
    data = collections.defaultdict(list)
    for qid in runs:
        for did in runs[qid]:
            data['qid'].append(qid)
            data['did'].append(did)
            data['query'].append(queries[qid])
            data['passage'].append(corpus[did])

    dataset = Dataset.from_dict(data)

    # data loader
    datacollator = DataCollatorForDPR(
            tokenizer=enc.tokenizer,
            padding=True,
            max_length=512,
            max_p_length=512,
            max_q_length=36,
            truncation=True,
            return_tensors="pt"
    )
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=datacollator
    )

    # prediction
    ranking_list = collections.defaultdict(list)
    for batch in tqdm(dataloader):
        batch_q_inputs, batch_p_inputs, ids = batch
        Q = enc.encode(inputs=batch_q_inputs, max_length=36, return_tensors=True)
        D = enc.encode(inputs=batch_p_inputs, max_length=512, return_tensors=True)
        scores = (Q*D).sum(-1).detach().cpu().numpy()

        for score, (qid, docid) in zip(scores, ids):
            ranking_list[qid].append((docid, score))

    # output
    for qid, candidate_passage_list in tqdm(ranking_list.items()):
        candidate_passage_list = sorted(candidate_passage_list, key=lambda x: x[1], reverse=True)

        for docid, t_prob in candidate_passage_list[:args.topk]:
            example = f'{qid} Q0 {docid} {str(idx+1)} {t_prob} {args.postfix}\n'
            fout.write(example)

    fout.close()
