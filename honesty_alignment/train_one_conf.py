import torch
from torch import nn
# from torch.utils.data import Dataset
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig, PeftModel
import argparse
import os
from prepare_label import construct_right_answer_conf, construct_greedy_answer_conf
import wandb
import random
import numpy as np
import json
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
import torch.distributed as dist

### 2. Data Collator: tokenize text + attach soft label
# 默认的collator是按照language modeling的方式构造labels, labels=input_ids。然后对一个batch的数据做padding
class DataCollatorWithVectorLabel:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # 进来的batch是按照trainer内部逻辑分词后的, 每个分词最后都加了一个eos_token
        # 这里按自己的逻辑重新处理一下
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        tokenized["labels"] = torch.tensor(labels, dtype=torch.float32)
        return tokenized


### 3. Model with classification head (output 10-dim vector)
class LMWithVectorHead(nn.Module):
    def __init__(self, model_name, lora_config, output_dim=1):
        super().__init__()
        backbone = AutoModel.from_pretrained(model_name, device_map='cpu')
        # backbone.config.use_cache = False
        self.peft_model = get_peft_model(backbone, lora_config)
        self.config = backbone.config
        hidden_size = backbone.config.hidden_size
        self.vector_head = nn.Linear(hidden_size, output_dim)  # 输出维度为 1

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """启用梯度检查点，并处理可能的额外参数"""
        self.peft_model.enable_input_require_grads()
        if gradient_checkpointing_kwargs is not None:
            self.peft_model.gradient_checkpointing_enable(**gradient_checkpointing_kwargs)
        else:
            self.peft_model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # if hasattr(self.peft_model, "gradient_checkpointing"):
        #     print(f"✅ 梯度检查点已启用 - 当前模式: {self.peft_model.is_gradient_checkpointing}")
        # else:
        #     print("❌ 梯度检查点未正确初始化")
        outputs = self.peft_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        # 获取最后一个 token 的隐藏状态
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        cls_hidden = last_hidden[:, -1, :]       # [B, H]
        logits = self.vector_head(cls_hidden)    # [B, 1]
        logits = torch.sigmoid(logits).squeeze(-1)  # 添加 sigmoid 并压缩至 [B]

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()  # 使用 MSE 损失
            loss = loss_fct(logits, labels)  # 计算 logits 和 labels 的 MSE

        return CausalLMOutput(
            loss=loss,
            logits=logits
        )


class VectorTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_losses = []  # 存储每个epoch的test loss
        self.output_dir = self.args.output_dir
        self.best_test_loss = float("inf")  # 当前最优的 test loss
        self.early_stop_counter = 0  # 早停计数器
        self.early_stop_patience = 4  # 如果test loss超过N个epoch没有下降，就early stop

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False, epoch_tag=None):
        """保存 LoRA 和 vector head。
        - epoch_tag: 用于文件命名，支持 'epoch{int}' 或 'epoch_best'
        """
        if not self.is_world_process_zero():  # 确保仅主进程保存
            return
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 设置 epoch 标签
        if epoch_tag is None:
            if hasattr(self.state, "epoch") and self.state.epoch is not None:
                epoch_tag = f"epoch{int(self.state.epoch)}"
            else:
                epoch_tag = "epoch_unknown"

        # 保存LoRA
        lora_output_dir = os.path.join(output_dir, f"lora_{epoch_tag}")
        self.model.peft_model.save_pretrained(lora_output_dir)

        # 保存vector head
        vector_head_path = os.path.join(output_dir, f"vector_head_{epoch_tag}.pt")
        torch.save(self.model.vector_head.state_dict(), vector_head_path)

        print(f"✅ Saved LoRA and vector head for {epoch_tag} to {output_dir}")

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        """重写以在每个epoch结束时记录test loss，并保存最佳模型"""
        result = super()._maybe_log_save_evaluate(*args, **kwargs)

        if self.state.epoch is not None and "eval_loss" in self.state.log_history[-1]:
            eval_loss = self.state.log_history[-1]["eval_loss"]
            current_epoch = int(self.state.epoch)

            # === 新增：同步各进程的eval_loss ===
            eval_loss_tensor = torch.tensor(eval_loss).to(self.args.device)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(eval_loss_tensor, op=torch.distributed.ReduceOp.AVG)
            avg_eval_loss = eval_loss_tensor.item()

            # 记录当前test loss
            if self.is_world_process_zero():
                self.test_losses.append({
                    "epoch": current_epoch,
                    "test_loss": eval_loss
                })

                # 增量写入文件
                loss_file = os.path.join(self.output_dir, "test_losses.jsonl")
                with open(loss_file, "a") as f:
                    f.write(json.dumps({
                        "epoch": current_epoch,
                        "test_loss": eval_loss
                    }) + "\n")

                # 保存当前最优模型
                if eval_loss < self.best_test_loss:
                    self.best_test_loss = eval_loss
                    self.early_stop_counter = 0  # 重置计数器
                    best_output_dir = os.path.join(self.output_dir, "best-checkpoint")
                    os.makedirs(best_output_dir, exist_ok=True)
                    self.save_model(output_dir=best_output_dir, epoch_tag="epoch_best")
                    print(f"🌟 New best model at epoch {current_epoch} with test_loss = {eval_loss:.4f}")
                else:
                    self.early_stop_counter += 1
                    print(f"⚠️ No improvement in test loss for {self.early_stop_counter} epoch(s)")

            # === 新增：分布式同步early_stop_counter ===
            if torch.distributed.is_initialized():
                # 将计数器转换为tensor进行同步
                counter_tensor = torch.tensor([self.early_stop_counter]).to(self.args.device)
                torch.distributed.broadcast(counter_tensor, 0)
                self.early_stop_counter = counter_tensor.item()
            
            # 所有进程检查早停条件
            if self.early_stop_counter >= self.early_stop_patience:
                if self.is_world_process_zero():
                    print(f"⛔ Early stopping triggered at epoch {current_epoch}")
                self.control.should_training_stop = True  # 核心修改：仅设置停止标志

        return result
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/mnt/bn/motor-nlp-team/models/LLM/base_models/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--train_data_path", default="", type=str)
    parser.add_argument("--test_data_path", default="", type=str)
    parser.add_argument("--train_consis_path", default="nothing", type=str)
    parser.add_argument("--test_consis_path", default="nothing", type=str)
    parser.add_argument("--train_acc_path", default="", type=str)
    parser.add_argument("--test_acc_path", default="", type=str)
    parser.add_argument("--output_path", default="./train/save_model")
    parser.add_argument("--logger_path", default="./train/save_model")
    parser.add_argument("--per_device_train_batch_size", default=16, type=int)
    parser.add_argument("--per_device_eval_batch_size", default=128, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--r", default=8, type=int)
    parser.add_argument("--alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.0, type=float)
    parser.add_argument("--train_data_count", default=0, type=int)
    parser.add_argument("--resume_from", action="store_true", help="Whether to resume from saved LoRA + vector head")
    parser.add_argument("--lora_path", type=str, default="", help="Path to saved LoRA")
    parser.add_argument("--vector_head_path", type=str, default="", help="Path to saved vector head")
    parser.add_argument("--dataset", type=str, default="nq", help="Dataset to process")
    parser.add_argument('--shuffle', action='store_true', help='Enable shuffling')

    args = parser.parse_args()

    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 固定 CUDA 卷积算法
    torch.backends.cudnn.benchmark = False    # 关闭自动优化

def print_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable}/{total} ({100*trainable/total:.2f}%)")

def main():
    args = get_args()
    set_seed(args.seed)
    accelerator = Accelerator()
    accelerate_set_seed(args.seed) # 设置随机种子（使用accelerate的版本以确保所有进程同步）
    
    if accelerator.is_main_process:
        print(args)
        output_path = args.output_path
        # check output_path
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"Folder {output_path} Created")
        else:
            print(f"Folder {output_path} Existed")

            # 初始化一个空的test_losses.jsonl文件
        loss_file = os.path.join(output_path, "test_losses.jsonl")
        if os.path.exists(loss_file):
            print(f"Warning: {loss_file} already exists, will append new data")
        else:
            with open(loss_file, "w") as f:
                pass  # 创建空文件
    accelerator.wait_for_everyone()
    ### Load tokenizer + dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_datasets = args.dataset.split('-')
    train_data, train_labels = [], []
    test_data, test_labels = [], []

    if len(all_datasets) == 1:
        if args.train_consis_path != "nothing":
            train_data, train_labels = construct_greedy_answer_conf(args.train_data_path, args.train_consis_path)
            test_data, test_labels = construct_greedy_answer_conf(args.test_data_path, args.test_consis_path)
        else:
            train_data, train_labels = construct_right_answer_conf(args.train_data_path)
            test_data, test_labels = construct_right_answer_conf(args.test_data_path)
    else:
        for dataset in all_datasets:
            print(f'construct data for {dataset}')
            # 为每个数据集构造输入数据路径
            train_data_path = args.train_data_path.replace(args.dataset, dataset)
            test_data_path = args.test_data_path.replace(args.dataset, dataset)
            train_consis_path = args.train_consis_path.replace(args.dataset, dataset)
            test_consis_path = args.test_consis_path.replace(args.dataset, dataset)

            if args.train_consis_path != "nothing":
                tmp_train_data, tmp_train_labels = construct_greedy_answer_conf(train_data_path, train_consis_path)
                tmp_test_data, tmp_test_labels = construct_greedy_answer_conf(test_data_path, test_consis_path)
            else:
                tmp_train_data, tmp_train_labels = construct_right_answer_conf(train_data_path)
                tmp_test_data, tmp_test_labels = construct_right_answer_conf(test_data_path)

            train_data.extend(tmp_train_data)
            train_labels.extend(tmp_train_labels)
            test_data.extend(tmp_test_data)
            test_labels.extend(tmp_test_labels)

        # shuffle train_data and train_labels
        combined = list(zip(train_data, train_labels))
        if args.shuffle:
            print('shuffle train data')
            random.shuffle(combined)
        train_data, train_labels = zip(*combined)
        train_data, train_labels = list(train_data), list(train_labels)
            
    train_data_cnt = len(train_data) if args.train_data_count == 0 else args.train_data_count
    if accelerator.is_main_process:
        print(f'train data count: {train_data_cnt}')

    if args.train_data_count == 0:
        # 如果 train_data_count 为 0，取全部数据
        train_dataset = Dataset.from_dict({
            "text": train_data,
            "label": train_labels
        })
    else:
        # 生成随机索引
        random_indices = random.sample(range(len(train_data)), train_data_cnt)
        # 根据随机索引选择数据
        sampled_train_data = [train_data[i] for i in random_indices]
        sampled_train_labels = [train_labels[i] for i in random_indices]
        
        train_dataset = Dataset.from_dict({
            "text": sampled_train_data,
            "label": sampled_train_labels
        })

    test_dataset = Dataset.from_dict({
    "text": test_data,
    "label": test_labels
    })
    collator = DataCollatorWithVectorLabel(tokenizer)
    
    ### 5. Training arguments
    # 看了代码发现, SFTTrainer里面的默认操作是做casual language modeling任务而不是SFT, 比如默认的collator不会把prompt部分的labels设置为-100
    # 忽略prompt部分的loss需要用DataCollatorForCompletionOnlyLM
    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.1,
        num_train_epochs=args.num_train_epochs,
        report_to="tensorboard",
        logging_steps=1,
        logging_dir=args.logger_path,
        remove_unused_columns=False,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        seed=args.seed,
        eval_strategy="epoch",
        save_strategy="no",
        optim="adamw_torch",
        weight_decay=args.weight_decay,
        gradient_checkpointing=False,
        ddp_find_unused_parameters=False,  # 重要DDP参数
        fsdp="",  # 明确不使用FSDP
    )
    if accelerator.is_main_process:
        print(training_args)

    ### 6. Model and trainer
    lora_config = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = LMWithVectorHead(args.model_path, lora_config)

    if args.resume_from:
        base_model = AutoModel.from_pretrained(args.model_path, device_map='cpu')
        peft_model = PeftModel.from_pretrained(
            base_model,
            args.lora_path,
            adapter_name="default",
            device_map='cpu'
        )
        model.peft_model = peft_model

        state_dict = torch.load(args.vector_head_path, map_location="cpu")
        model.vector_head.load_state_dict(state_dict)
        model.peft_model.set_adapter("default")

    if accelerator.is_main_process:
        print(f'use cache: {model.config.use_cache}')  # 应该输出False
        print_trainable_params(model)

    trainer = VectorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collator
    )

    ### 7. Train
    trainer.train()
    # 保存 LoRA 参数
    # model.peft_model.save_pretrained(output_path)  # 仅 LoRA

    # # 保存 vector_head 参数
    # torch.save(model.vector_head.state_dict(), os.path.join(output_path, "vector_head.pt"))

if __name__ == "__main__":
    main()