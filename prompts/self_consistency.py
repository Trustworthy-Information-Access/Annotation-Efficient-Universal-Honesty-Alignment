from convert import read_json, write_jsonl
import sys
sys.path.append('../')
from post_process.utils import has_answer, deal_judge_new
from tqdm import tqdm
from itertools import combinations
from collections import Counter, defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse

def construct_question_for_consistency_check(qa_path, sample_qa_path):
    """
    将 greedy answer 和 sampled answer 整合成一个问题，用于一致性判断。
    
    输入:
        - qa_path: greedy answer 对应的文件路径
        - sample_qa_path: sampled answer 对应的文件路径
    """
    qa_data = read_json(qa_path)
    sample_data = read_json(sample_qa_path)
    new_data = []

    for idx in range(len(qa_data)):
        for sample_ans in sample_data[idx]['response']:
            question = qa_data[idx]['question']
            base_template = (
                "Are the following two answers to the same question semantically equivalent?\n"
                "If the two answers are semantically equivalent, answer \"certain\". Otherwise, answer \"uncertain\". Given ONLY the judgement (\"certain\" or \"uncertain\"), no other words or explanation."
            )
            
            template1 = (
                f"Question: {question}\n"
                f"Answer: {qa_data[idx]['response'][0]}"
            )
            
            template2 = (
                f"Question: {question}\n"
                f"Answer: {sample_ans}"
            )
            
            full_template = f"{base_template}\n{template1}\n\n{template2}"
            
            new_data.append({
                'question': full_template,
                'answer': 'no answer',
                'original_question': question
            })

    
    output_path = sample_qa_path.replace('.jsonl', '_for_greedy_consistency_check.jsonl')
    write_jsonl(new_data, output_path)

def get_args():
    parser = argparse.ArgumentParser(description='Self consistency data construction')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_type', type=str, required=True)
    parser.add_argument('--qa_type', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--res_path', type=str, required=True)

    return parser


if __name__ == '__main__':
    parser = get_args()
    args = parser.parse_args()
    dataset=args.dataset
    data_type=args.data_type # train, test
    qa_type=args.qa_type # short, long
    model_name=args.model_name
    res_path=args.res_path
    qa_path=f'{res_path}/{model_name}/{dataset}/{data_type}_data/{qa_type}/{dataset}_{data_type}_{model_name}_{qa_type}_0.0_0.95_50_sample_1.jsonl'
    sample_path=f'{res_path}/{model_name}/{dataset}/{data_type}_data/{qa_type}/{dataset}_{data_type}_{model_name}_{qa_type}_1.0_0.95_50_sample_20.jsonl'
    construct_question_for_consistency_check(qa_path, sample_path)
    
