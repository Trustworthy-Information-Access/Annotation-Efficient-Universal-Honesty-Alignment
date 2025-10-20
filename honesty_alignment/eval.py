import torch
from torch import nn
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from trl import SFTTrainer
from peft import PeftModel, LoraConfig
from train import LMWithVectorHead, DataCollatorWithVectorLabel, VectorTrainer
import argparse
import os
from transformers import set_seed
import sys
sys.path.append('../')
from post_process.utils import *
import math
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd
from prepare_label import calculate_consistency_conf_per_n_elements_parallel
from openpyxl import Workbook
from rouge import Rouge
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

def write_to_excel(data, path):
    """
    将二维列表写入 Excel 文件 (.xlsx)

    参数:
        data: 二维列表，如 [["Name", "Age"], ["Alice", 25]]
        path: 文件保存路径，例如 "output.xlsx"
    """
    wb = Workbook()
    ws = wb.active

    for row in data:
        ws.append(row)

    wb.save(path)

def compute_auroc(conf_data, acc_data):
    auroc = roc_auc_score(acc_data, conf_data)
    return auroc

def compute_ece(conf_data, acc_data):
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    # 转换为 NumPy 数组
    conf_data = np.array(conf_data)
    acc_data = np.array(acc_data)
    
    bin_indices = np.digitize(conf_data, bin_edges) - 1
    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            avg_pred = np.mean(conf_data[mask])
            avg_true = np.mean(acc_data[mask])
            ece += np.abs(avg_pred - avg_true) * np.sum(mask) / len(acc_data)
    return ece

def calculate_consistency_conf_per_n_elements(lst, n):
    """
    得到对一个问题多次采样的平均confidence
    """
    averages = []
    for i in range(0, len(lst), n):
        # 获取当前组的10个元素（如果不足10个，则取剩余的所有元素）
        group = lst[i:i+n]
        group = [1 - int(deal_judge_new(item['response'][0])) for item in group]
        # 计算当前组的平均值
        if group:  # 确保组不为空
            average = sum(group) / len(group)
            averages.append(average)
    return averages

 
def extract_score(text):
    # for verbalized conf
    # 匹配 0 到 1 之间的浮点数（包括 0.0, 1.0, .5 等形式）
    pattern = r'\b(0?\.\d+|1(?:\.0+)?|0)\b'
    matches = re.findall(pattern, text)
    
    if matches:
        # 取最后一个匹配（通常是最新的分数）
        return float(matches[-1])
    return 0.0

def compute_rouge_for_group(args):
    # for Sem-Lex
    idx, qa_data, sample_data = args
    rouge = Rouge()
    greedy_answer = qa_data[idx]['response'][0]
    sample_answers = sample_data[idx]['response']
    if not sample_answers:
        return 0
    refs = [greedy_answer] * len(sample_answers)
    scores = rouge.get_scores(sample_answers, refs)
    rouge_l_scores = [s['rouge-l']['f'] for s in scores]
    return sum(rouge_l_scores) / len(rouge_l_scores)

def compute_best_threshold(val_conf, val_acc):
    """
    在验证集上找最佳阈值
    要求：不能把所有样本都分成0或1
    """
    best_thres, best_align = 0.5, 0
    for thres in [i/100 for i in range(1, 100)]:  # 从0.01到0.99搜索
        preds = [1 if c >= thres else 0 for c in val_conf]
        if all(p == 0 for p in preds) or all(p == 1 for p in preds):
            continue  # 跳过全0或全1
        align = sum(int(p == a) for p, a in zip(preds, val_acc)) / len(val_acc)
        if align > best_align:
            best_align, best_thres = align, thres
    return best_thres, best_align

def compute_alignment_for_one_conf(test_qa_path, test_conf_path, test_consis_path='', 
                                   dev_qa_path='', dev_conf_path='', dev_consis_path='', 
                                   sample_cnt=0, conf_type='greedy'):
    """
    test_qa_path: greedy answer & 
    test_conf_path: 预测得到的信心
    test_consis_path: 多次采样+一致性检验得到的信心 or 多次采样回复正确的概率
    """
    qa_data = read_json(test_qa_path)
    conf_data = read_json(test_conf_path)
    test_consis_data = read_json(test_consis_path)
    if 'long_qa' in test_qa_path:
        acc_judge_path = test_qa_path.replace('.jsonl', '_long_qa_judge.jsonl')
        acc_judge_data = read_json(acc_judge_path)
        acc = [int(1 - deal_judge_new(item['response'][0])) for item in acc_judge_data]
    else:
        acc = [has_answer(item['answer'], item['response'][0]) for item in qa_data]
    prob_conf = [math.exp(item['cumulative_logprobs'][0] / len(item['tokens'][0])) for item in qa_data]
    test_consis_conf = calculate_consistency_conf_per_n_elements_parallel(test_consis_data, 20, 4)
    pred_conf = [item['predicted'] for item in conf_data]

    sample_cnt = len(acc) if sample_cnt == 0 else sample_cnt
    print(f'conf type: {conf_type}')
    print(f'sample_cnt: {sample_cnt}')
    print(f'avg acc: {sum(acc)/len(acc)}')

    print(f'auroc: prob_conf, test_consis_conf, pred_conf')
    prob_auroc = compute_auroc(prob_conf, acc)
    consis_auroc = compute_auroc(test_consis_conf, acc)
    pred_auroc = compute_auroc(pred_conf, acc)
    print(f'AUROC: {prob_auroc}')
    print(f'AUROC: {consis_auroc}')
    print(f'AUROC: {pred_auroc}')

    print(f'ece: prob_conf, test_consis_conf, pred_conf')
    prob_ece = compute_ece(prob_conf, acc)
    consis_ece = compute_ece(test_consis_conf, acc)
    pred_ece = compute_ece(pred_conf, acc)
    print(f'ECE: {prob_ece}')
    print(f'ECE: {consis_ece}')
    print(f'ECE: {pred_ece}')

    # === 计算 alignment ===
    def get_alignment(conf_list, acc_list):
        half = int(len(conf_list) * 0.2)
        val_conf, val_acc = conf_list[:half], acc_list[:half]
        test_conf, test_acc = conf_list[half:], acc_list[half:]
        thres, _ = compute_best_threshold(val_conf, val_acc)
        preds = [1 if c >= thres else 0 for c in test_conf]
        align = sum(int(p == a) for p, a in zip(preds, test_acc)) / len(test_acc)
        return round(align * 100, 2)

    prob_align = get_alignment(prob_conf, acc)
    consis_align = get_alignment(test_consis_conf, acc)
    pred_align = get_alignment(pred_conf, acc)
    print(f'Alignment: prob_conf={prob_align}, consis_conf={consis_align}, pred_conf={pred_align}')

    return (round(prob_auroc*100, 2), round(consis_auroc*100, 2), round(pred_auroc*100, 2), 
            round(prob_ece, 2), round(consis_ece, 2), round(pred_ece, 2),
            prob_align, consis_align, pred_align)


def get_res_on_each_dataset_for_every_method(args):
    """
    Output:
    [[prob, self-consis, elicitation, calibration-only(1k), calibration(1k)],
     [prob, self-consis, elicitation, calibration-only(2k), calibration(2k)],
     [prob, self-consis, elicitation, calibration-only(4k), calibration(4k)],
      ...,
     [prob, self-consis, elicitation, calibration-only(560k), calibration(560k)]]
    """
    base_path = args.base_path
    qa_type=args.qa_type # long/short
    test_qa_type = qa_type
    eval_dataset=args.eval_dataset
    if eval_dataset == 'mmlu':
        test_qa_type = test_qa_type + '_mc'
    model_name=args.model_name
    test_qa_path = f'{base_path}/{model_name}/{eval_dataset}/test_data/{test_qa_type}_qa/{eval_dataset}_test_{model_name}_{test_qa_type}_qa_0.0_0.95_50_sample_1.jsonl'
    test_consis_path=f'{base_path}/{model_name}/{eval_dataset}/test_data/{test_qa_type}_qa/{eval_dataset}_test_{model_name}_{test_qa_type}_qa_1.0_0.95_50_sample_20_for_greedy_consistency_res.jsonl'
    train_dataset = args.train_dataset
    all_sample_cnts = args.training_samples.split(',')
    conf_types = ['greedy', 'right', 'hybrid']

    greedy_training_samples=int(args.greedy_samples)
    if greedy_training_samples == 0:
        greedy_epochs=10
        greedy_tail_name = ''
    elif greedy_training_samples <= 10000:
        greedy_epochs=50
        k=int(greedy_training_samples/1000)
        greedy_tail_name=f'_{k}k_training_samples'
    else:
        greedy_epochs=15
        k=int(greedy_training_samples/1000)
        greedy_tail_name=f'_{k}k_training_samples'

    res_for_all_sample_cnt_auroc = []
    res_for_all_sample_cnt_ece = []
    res_for_all_sample_cnt_align = []   # 新增 alignment 保存列表

    for training_data_cnt in all_sample_cnts:
        training_data_cnt=int(training_data_cnt)
        run_cnt = 0
        res_for_one_sample_cnt_auroc = []
        res_for_one_sample_cnt_ece = []
        res_for_one_sample_cnt_align = []  # 新增：每个 sample_cnt 的 alignment

        for conf_type in conf_types:
            if training_data_cnt == 0:
                epochs=10
                tail_name = ''
            elif training_data_cnt <= 10000:
                epochs=50
                k=int(training_data_cnt/1000)
                tail_name=f'_{k}k_training_samples'
            else:
                epochs=15
                k=int(training_data_cnt/1000)
                tail_name=f'_{k}k_training_samples'
            if conf_type == 'greedy':
                if run_cnt != 0:
                    continue
                epochs=10

            # evaluation for mlp
            # if conf_type == 'greedy':
            #      test_conf_path=f'{base_path}/{model_name}/eval/{train_dataset}/mlp/{eval_dataset}/{conf_type}_answer_conf/{qa_type}_qa/batchsize{args.batch_size}_accumulation{args.accumulation_steps}_epochs{greedy_epochs}_weightdecay0.1_mlp_1_layer{greedy_tail_name}/evaluation_results_test.jsonl'
            # elif conf_type == 'hybrid':
            #     test_conf_path=f'{base_path}/{model_name}/eval/{train_dataset}/mlp/{eval_dataset}/{conf_type}_answer_conf{greedy_tail_name}/{qa_type}_qa/batchsize{args.batch_size}_accumulation{args.accumulation_steps}_epochs{epochs}_weightdecay0.1_mlp_1_layer{tail_name}/evaluation_results_test.jsonl'
            # else:
            #     test_conf_path=f'{base_path}/{model_name}/eval/{train_dataset}/mlp/{eval_dataset}/{conf_type}_answer_conf/{qa_type}_qa/batchsize{args.batch_size}_accumulation{args.accumulation_steps}_epochs{epochs}_weightdecay0.1_mlp_1_layer{tail_name}/evaluation_results_test.jsonl'
            
            # evaluation for LoRA
            if conf_type == 'greedy':
                 test_conf_path=f'{base_path}/{model_name}/eval/{train_dataset}/{eval_dataset}/{conf_type}_answer_conf/{qa_type}_qa/batchsize{args.batch_size}_accumulation{args.accumulation_steps}_epochs{greedy_epochs}_weightdecay0.1_r8_alpha16_loradrpout0.0{greedy_tail_name}/evaluation_results_test.jsonl'
            elif conf_type == 'hybrid':
                test_conf_path=f'{base_path}/{model_name}/eval/{train_dataset}/{eval_dataset}/{conf_type}_answer_conf{greedy_tail_name}/{qa_type}_qa/batchsize{args.batch_size}_accumulation{args.accumulation_steps}_epochs{epochs}_weightdecay0.1_r8_alpha16_loradrpout0.0{tail_name}/evaluation_results_test.jsonl'
            else:
                test_conf_path=f'{base_path}/{model_name}/eval/{train_dataset}/{eval_dataset}/{conf_type}_answer_conf/{qa_type}_qa/batchsize{args.batch_size}_accumulation{args.accumulation_steps}_epochs{epochs}_weightdecay0.1_r8_alpha16_loradrpout0.0{tail_name}/evaluation_results_test.jsonl'

            prob_auroc, consis_auroc, pred_auroc, prob_ece, consis_ece, pred_ece, prob_align, consis_align, pred_align = compute_alignment_for_one_conf(
                test_qa_path, test_conf_path, test_consis_path, '', '', '', training_data_cnt, conf_type
            )
            if run_cnt == 0:
                res_for_one_sample_cnt_auroc.extend([prob_auroc, consis_auroc, pred_auroc])
                res_for_one_sample_cnt_ece.extend([prob_ece, consis_ece, pred_ece])
                res_for_one_sample_cnt_align.extend([prob_align, consis_align, pred_align])
            else:
                res_for_one_sample_cnt_auroc.append(pred_auroc)
                res_for_one_sample_cnt_ece.append(pred_ece)
                res_for_one_sample_cnt_align.append(pred_align)
            run_cnt += 1

        res_for_all_sample_cnt_auroc.append(res_for_one_sample_cnt_auroc)
        res_for_all_sample_cnt_ece.append(res_for_one_sample_cnt_ece)
        res_for_all_sample_cnt_align.append(res_for_one_sample_cnt_align)

    save_dir = f'./res/{model_name}/{train_dataset}/{greedy_tail_name}/{eval_dataset}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    write_to_excel(res_for_all_sample_cnt_auroc, f'{save_dir}/{qa_type}_alignment_auroc.xlsx')
    write_to_excel(res_for_all_sample_cnt_ece, f'{save_dir}/{qa_type}_alignment_ece.xlsx')
    write_to_excel(res_for_all_sample_cnt_align, f'{save_dir}/{qa_type}_alignment_align.xlsx')  # 新增保存 alignment




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--train_dataset', type=str, required=True)
    parser.add_argument('--eval_dataset', type=str, required=True)
    parser.add_argument('--qa_type', type=str, required=True)
    parser.add_argument('--training_samples', type=str, required=True)
    parser.add_argument('--greedy_samples', type=str, required=True)
    parser.add_argument('--base_path', type=str, default='/data/users/nishiyu/res/honesty_alignment/res', required=True)
    parser.add_argument('--batch_size', type=int, required=True, default=16)
    parser.add_argument('--accumulation_steps', type=int, required=True, default=8)

    return parser

if __name__ == "__main__":

    parser = get_args()
    args = parser.parse_args()
    get_res_on_each_dataset_for_every_method(args)

