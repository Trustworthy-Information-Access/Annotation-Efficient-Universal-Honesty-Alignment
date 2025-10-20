import torch
from torch import nn
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from trl import SFTTrainer
from peft import PeftModel, LoraConfig
from prepare_label import construct_right_answer_conf, construct_greedy_answer_conf
from train_mlp import LMWithVectorHead, DataCollatorWithVectorLabel, VectorTrainer
import argparse
import os
from transformers import set_seed
import sys
sys.path.append('../')
from post_process.utils import *
import math
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
import torch.distributed as dist

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, nargs='+', required=True)  # 改为支持多个路径
    parser.add_argument("--test_data_path", type=str, nargs='+', required=True)  # 改为支持多个路径
    parser.add_argument("--train_data_path", type=str, nargs='+', required=True)  # 改为支持多个路径
    parser.add_argument("--test_consis_path", type=str, nargs='*', default=[])  # 改为可选的多路径
    parser.add_argument("--train_consis_path", type=str, nargs='*', default=[])  # 改为可选的多路径
    parser.add_argument("--vector_head_path", type=str, required=True)
    parser.add_argument("--per_device_eval_batch_size", default=16, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--type", default='test', type=str)
    args = parser.parse_args()
    return args

def evaluate():
    args = get_args()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据
    total_test_dataset = []
    total_train_dataset = []
    for idx in range(len(args.test_data_path)):
        test_data, test_labels = construct_right_answer_conf(args.test_data_path[idx])
        train_data, train_labels = construct_right_answer_conf(args.train_data_path[idx])
        test_dataset = Dataset.from_dict({"text": test_data, "label": test_labels})
        train_dataset = Dataset.from_dict({"text": train_data, "label": train_labels})
        total_test_dataset.append(test_dataset)
        total_train_dataset.append(train_dataset)
    collator = DataCollatorWithVectorLabel(tokenizer)

    model = LMWithVectorHead(args.model_path)

    # 5. 加载分类头权重
    state_dict = torch.load(args.vector_head_path, map_location=device)
    model.vector_head.load_state_dict(state_dict)
    model = model.to(device)
    
    # 评估模式
    model.eval()

    # 执行预测
    for idx in range(len(total_test_dataset)):
        # 创建TrainingArguments (每次循环都创建新的)
        training_args = TrainingArguments(
            output_dir=args.output_path[idx],
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            remove_unused_columns=False,
            seed=args.seed,
            eval_strategy="no",
            save_strategy="no",
            logging_steps=10,
            report_to="none",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            dataloader_drop_last=False,  # 不丢弃最后一个批次
        )
        
        # 创建Trainer (每次循环都创建新的)
        trainer = VectorTrainer(
            model=model,
            args=training_args,
            data_collator=collator,
            train_dataset=total_train_dataset[idx],
            eval_dataset=total_test_dataset[idx]
        )
        
        # 记录内存使用情况
        if dist.is_initialized() and dist.get_rank() == 0:
            print(f"\n=== Processing dataset {idx+1}/{len(total_test_dataset)} ===")
            print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
        
        # 执行预测
        predictions = trainer.predict(total_test_dataset[idx] if args.type == 'test' else total_train_dataset[idx])
        
        pred_logits = predictions.predictions
        pred_probs = pred_logits.astype(float).tolist()
        
        # 结果展示
        true_labels = predictions.label_ids.astype(float).tolist()
        if dist.is_initialized() and dist.get_rank() == 0:
            print(type(true_labels))
            for i in range(5):
                print(f'idx={i}--------------------------------------')
                print("Predicted:", pred_probs[i])
                print("True label:", true_labels[i])

            # 保存结果到JSONL文件
            output_file = os.path.join(args.output_path[idx], f"evaluation_results_{args.type}.jsonl")
            print(f'output_file={output_file}')
            with open(output_file, 'w') as f:
                for i in range(len(pred_probs)):
                    # 处理pred_probs
                    if torch.is_tensor(pred_probs[i]):
                        pred_list = pred_probs[i].cpu().numpy().astype(float).tolist()
                    elif isinstance(pred_probs[i], np.ndarray):
                        pred_list = pred_probs[i].astype(float).tolist()
                    else:
                        pred_list = pred_probs[i]
                    
                    # 处理true_labels
                    if torch.is_tensor(true_labels[i]):
                        true_list = true_labels[i].cpu().numpy().astype(float).tolist()
                    elif isinstance(true_labels[i], np.ndarray):
                        true_list = true_labels[i].astype(float).tolist()
                    else:
                        true_list = true_labels[i]
                    
                    record = {
                        "index": i,
                        "predicted": pred_list,
                        "true_label": true_list
                    }
                    f.write(json.dumps(record) + '\n')
        
        # 清理Trainer和CUDA缓存
        del trainer
        torch.cuda.empty_cache()
        
        # 强制垃圾回收
        import gc
        gc.collect()

if __name__ == "__main__":
    evaluate()