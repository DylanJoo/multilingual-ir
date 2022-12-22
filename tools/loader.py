import collections 
from datasets import load_dataset

def get_miracl_query(lang=None, subset='dev'):
    data = load_dataset('miracl/miracl', lang)[subset]
    print(data)
    data_dict = dict(zip(data['query_id'], data['query']))
    return data_dict

def get_miracl_corpus(lang=None, append_title=False):
    data = load_dataset('miracl/miracl-corpus', lang)['train']
    print(data)
    if append_title:
        data_dict = dict(zip(data['docid'], 
            [f"{ti} {te}" for ti, te in zip(data['title'], data['text'])]))
    else:
        data_dict = dict(zip(data['docid'], data['text']))
    return data_dict

def load_runs(path=None, output_score=False): # support .trec file only
    run_dict = collections.defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            qid, _, docid, rank, score, _ = line.strip().split()
            run_dict[qid] += [(docid, float(rank), float(score))]

    sorted_run_dict = collections.OrderedDict()
    for qid, docid_ranks in run_dict.items():
        sorted_docid_ranks = sorted(docid_ranks, key=lambda x: x[1], reverse=False) 
        if output_score:
            sorted_run_dict[qid] = [(docid, rel_score) for docid, rel_rank, rel_score in sorted_docid_ranks]
        else:
            sorted_run_dict[qid] = [docid for docid, _, _ in sorted_docid_ranks]

    return sorted_run_dict

