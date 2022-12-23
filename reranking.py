import torch
from tqdm import tqdm
import argparse
import collections
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def prepare_dataset(args):
    # load raw triplet
    from tools.loader import get_miracl_query, get_miracl_corpus, load_runs
    queries = get_miracl_query(args.lang, 'dev')
    corpus = get_miracl_corpus(args.lang)
    runs = load_runs(args.run, False)

    # prepare data 
    data = collections.defaultdict(list)
    for qid in runs:
        for did in runs[qid]:
            data['qid'].append(qid)
            data['did'].append(did)
            data['query'].append(queries[qid])
            data['passage'].append(corpus[did])

    from datasets import Dataset
    dataset = Dataset.from_dict(data)
    return dataset 

def biencoder_rerank(args):
    # load model
    from rerank.biencoder import BertEncoder
    model = BertEncoder(args.model_name_or_path, device=f'cuda:{args.gpu}')

    # get dataset
    dataset = prepare_dataset(args)

    from datacollator import DataCollatorForDPR
    datacollator = DataCollatorForDPR(
            tokenizer=model.tokenizer,
            padding=True,
            max_p_length=args.max_p_length,
            max_q_length=args.max_q_length,
            truncation=True,
            return_tensors="pt"
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
            collate_fn=datacollator
    )

    # prediction using biencoder
    ranking_list = collections.defaultdict(list)
    for batch in tqdm(dataloader):
        batch_q_inputs, batch_p_inputs, ids = batch
        Q = model.encode(inputs=batch_q_inputs, max_length=args.max_q_length, return_tensors=True)
        D = model.encode(inputs=batch_p_inputs, max_length=args.max_p_length, return_tensors=True)
        scores = (Q*D).sum(-1).detach().cpu().numpy()

        for score, (qid, did) in zip(scores, ids):
            ranking_list[qid].append((did, score))

    # output
    with open(args.output, 'w') as fout:
        for qid, candidate_passage_list in tqdm(ranking_list.items()):
            candidate_passage_list = sorted(candidate_passage_list, key=lambda x: x[1], reverse=True)

            for idx, (did, score) in enumerate(candidate_passage_list[:args.topk]):
                example = f'{qid} Q0 {did} {str(idx+1)} {score} {args.postfix}\n'
                fout.write(example)

def crossencoder_rerank(args):
    # load model
    from rerank.crossencoder import monoMT5
    model = monoMT5.from_pretrained(args.model_name_or_path)
    model.set(
            device=f'cuda:{args.gpu}', 
            tokenizer_name=args.model_name_or_path,
            targeted_tokens=['yes', 'no']
    )

    # get dataset
    dataset = prepare_dataset(args)

    from datacollator import DataCollatorFormonoT5
    datacollator = DataCollatorFormonoT5(
            tokenizer=model.tokenizer,
            padding=True,
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt"
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
            collate_fn=datacollator
    )

    # prediction
    ranking_list = collections.defaultdict(list)
    for batch in tqdm(dataloader):
        batch_inputs, ids = batch
        output = model.predict(batch_inputs)

        true_prob = output[:, 0]
        false_prob = output[:, 1]

        for t_prob, (qid, docid) in zip(true_prob, ids):
            ranking_list[qid].append((docid, t_prob))

    # output
    with open(args.output, 'w') as fout:
        for qid, candidate_passage_list in tqdm(ranking_list.items()):
            candidate_passage_list = sorted(candidate_passage_list, key=lambda x: x[1], reverse=True)

            for idx, (did, score) in enumerate(candidate_passage_list[:args.topk]):
                example = f'{qid} Q0 {did} {str(idx+1)} {score} {args.postfix}\n'
                fout.write(example)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Load triplet
    parser.add_argument("--run", type=str, required=True,)
    parser.add_argument("--lang", type=str, required=True,)
    # Reranking conditions
    parser.add_argument("--topk", type=int, default=1000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default='bert-base-uncased')
    parser.add_argument("--reranker_type", type=str, default='cross-encoder')
    parser.add_argument("--gpu", type=str, default='2')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_p_length", type=int, default=256)
    parser.add_argument("--max_q_length", type=int, default=36)
    parser.add_argument("--postfix", type=str, default='monoMT5')
    args = parser.parse_args()

    if args.reranker_type == 'cross-encoder':
        crossencoder_rerank(args)
    else:
        biencoder_rerank(args)

