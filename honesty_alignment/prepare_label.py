import sys
sys.path.append('../')
from post_process.utils import has_answer, deal_judge_new
from tqdm import tqdm
from itertools import combinations
from collections import Counter, defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import orjson
from pathos.multiprocessing import ProcessingPool as Pool

def read_json(path):
    qa_data = []
    with open(path, 'rb') as f:  # 二进制模式，orjson需要bytes
        for line in f:
            qa_data.append(orjson.loads(line))
    return qa_data

def write_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f'write jsonl to: {path}')
    f.close()

def construct_right_answer_conf(sample_qa_path, num_workers=8):
    """
    construct correctness label for short_qa
    """
    def process_item(item):
        responses = item['response']
        temp_acc = [has_answer(item['answer'], response) for response in responses]
        confidence = sum(temp_acc) / len(temp_acc)
        return item['prompt'], confidence
    
    data = read_json(sample_qa_path)
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_item, data, chunksize=100),
            total=len(data),
            desc="Processing items"
        ))
    
    inputs, labels = zip(*results)
    return list(inputs), list(labels)

def calculate_single_group_conf(group):
    return sum([1 - int(deal_judge_new(item['response'][0])) for item in group]) / len(group)

def calculate_consistency_conf_per_n_elements_parallel(lst, n, num_workers=8):
    groups = [lst[i:i+n] for i in range(0, len(lst), n)]
    
    with Pool(processes=num_workers) as pool:
        averages = list(tqdm(
            pool.imap(calculate_single_group_conf, groups, chunksize=10),
            total=len(groups),
            desc="Calculating consistency conf"
        ))
    
    return averages

def construct_greedy_answer_conf(sample_qa_path, consis_qa_path, num_workers=8):
    """
    construct self-consistency / correctness label for long_qa
    """
    qa_data = read_json(sample_qa_path)
    consis_data = read_json(consis_qa_path)

    # ✅ 用并行的版本替代, 数据处理一定要并行, 因为大文件特别慢
    consis_conf = calculate_consistency_conf_per_n_elements_parallel(consis_data, 20, num_workers)
    assert len(qa_data) == len(consis_conf)

    # 不再需要多进程做这一步（因为它很快）
    prompts = [item['prompt'] for item in qa_data]
    return prompts, consis_conf