import torch
from torch import nn
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments
from transformers.modeling_outputs import CausalLMOutput
from trl import SFTTrainer
import argparse
import os
from prepare_label import construct_right_answer_conf, construct_greedy_answer_conf
import random
import numpy as np
import json
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed

# === Data Collator ===
class DataCollatorWithVectorLabel:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
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

# === 只训练分类头版本 ===
class LMWithVectorHead(nn.Module):
    def __init__(self, model_name, output_dim=1):
        super().__init__()
        backbone = AutoModel.from_pretrained(model_name)
        self.config = backbone.config
        hidden_size = backbone.config.hidden_size

        # 冻结主干参数
        for p in backbone.parameters():
            p.requires_grad = False

        self.backbone = backbone
        
        # 三层 MLP 作为 head
        self.vector_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_dim)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        cls_hidden = last_hidden[:, -1, :]       # [B, H]

        logits = self.vector_head(cls_hidden)    # [B, output_dim]
        logits = torch.sigmoid(logits).squeeze(-1)

        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits, labels)

        return CausalLMOutput(
            loss=loss,
            logits=logits
        )

# === Trainer ===
class VectorTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_losses = []
        self.output_dir = self.args.output_dir
        self.best_test_loss = float("inf")
        self.early_stop_counter = 0
        self.early_stop_patience = 4

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False, epoch_tag=None):
        if not self.is_world_process_zero():
            return
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if epoch_tag is None:
            if hasattr(self.state, "epoch") and self.state.epoch is not None:
                epoch_tag = f"epoch{int(self.state.epoch)}"
            else:
                epoch_tag = "epoch_unknown"

        # 仅保存分类头
        vector_head_path = os.path.join(output_dir, f"vector_head_{epoch_tag}.pt")
        torch.save(self.model.vector_head.state_dict(), vector_head_path)
        print(f"✅ Saved vector head for {epoch_tag} to {output_dir}")

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        result = super()._maybe_log_save_evaluate(*args, **kwargs)
        if self.state.epoch is not None and "eval_loss" in self.state.log_history[-1]:
            eval_loss = self.state.log_history[-1]["eval_loss"]
            current_epoch = int(self.state.epoch)

            eval_loss_tensor = torch.tensor(eval_loss).to(self.args.device)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(eval_loss_tensor, op=torch.distributed.ReduceOp.AVG)
            avg_eval_loss = eval_loss_tensor.item()

            if self.is_world_process_zero():
                self.test_losses.append({"epoch": current_epoch, "test_loss": eval_loss})
                loss_file = os.path.join(self.output_dir, "test_losses.jsonl")
                with open(loss_file, "a") as f:
                    f.write(json.dumps({"epoch": current_epoch, "test_loss": eval_loss}) + "\n")

                if eval_loss < self.best_test_loss:
                    self.best_test_loss = eval_loss
                    self.early_stop_counter = 0
                    best_output_dir = os.path.join(self.output_dir, "best-checkpoint")
                    os.makedirs(best_output_dir, exist_ok=True)
                    self.save_model(output_dir=best_output_dir, epoch_tag="epoch_best")
                    print(f"🌟 New best model at epoch {current_epoch} with test_loss = {eval_loss:.4f}")
                else:
                    self.early_stop_counter += 1
                    print(f"⚠️ No improvement in test loss for {self.early_stop_counter} epoch(s)")

            if torch.distributed.is_initialized():
                counter_tensor = torch.tensor([self.early_stop_counter]).to(self.args.device)
                torch.distributed.broadcast(counter_tensor, 0)
                self.early_stop_counter = counter_tensor.item()

            if self.early_stop_counter >= self.early_stop_patience:
                if self.is_world_process_zero():
                    print(f"⛔ Early stopping triggered at epoch {current_epoch}")
                self.control.should_training_stop = True
        return result

# === 其他函数保留不变 ===
def get_args():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/mnt/bn/motor-nlp-team/models/LLM/base_models/Qwen2.5-7B-Instruct", type=str)
    parser.add_argument("--train_data_path", default="", type=str)
    parser.add_argument("--test_data_path", default="", type=str)
    parser.add_argument("--train_consis_path", default="nothing", type=str)
    parser.add_argument("--test_consis_path", default="nothing", type=str)
    parser.add_argument("--output_path", default="./train/save_model")
    parser.add_argument("--logger_path", default="./train/save_model")
    parser.add_argument("--per_device_train_batch_size", default=16, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--train_data_count", default=0, type=int)
    parser.add_argument("--resume_from", action="store_true", help="Whether to resume from saved LoRA + vector head")
    parser.add_argument("--vector_head_path", type=str, default="", help="Path to saved vector head")
    parser.add_argument("--dataset", type=str, default="nq", help="Dataset to process")
    parser.add_argument('--shuffle', action='store_true', help='Enable shuffling')
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable}/{total} ({100*trainable/total:.2f}%)")

def main():
    args = get_args()
    set_seed(args.seed)
    accelerator = Accelerator()
    accelerate_set_seed(args.seed)

    if accelerator.is_main_process:
        print(args)
        os.makedirs(args.output_path, exist_ok=True)
        loss_file = os.path.join(args.output_path, "test_losses.jsonl")
        if not os.path.exists(loss_file):
            open(loss_file, "w").close()

    accelerator.wait_for_everyone()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_datasets = args.dataset.split('-')
    train_data, train_labels, test_data, test_labels = [], [], [], []

    if len(all_datasets) == 1:
        if args.train_consis_path != "nothing":
            train_data, train_labels = construct_greedy_answer_conf(args.train_data_path, args.train_consis_path)
            test_data, test_labels = construct_greedy_answer_conf(args.test_data_path, args.test_consis_path)
        else:
            train_data, train_labels = construct_right_answer_conf(args.train_data_path)
            test_data, test_labels = construct_right_answer_conf(args.test_data_path)
    else:
        for dataset in all_datasets:
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
        if args.shuffle:
            combined = list(zip(train_data, train_labels))
            random.shuffle(combined)
            train_data, train_labels = zip(*combined)

    train_data_cnt = len(train_data) if args.train_data_count == 0 else args.train_data_count
    if accelerator.is_main_process:
        print(f'train data count: {train_data_cnt}')

    train_dataset = Dataset.from_dict({"text": train_data[:train_data_cnt], "label": train_labels[:train_data_cnt]})
    test_dataset = Dataset.from_dict({"text": test_data, "label": test_labels})
    collator = DataCollatorWithVectorLabel(tokenizer)

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
        ddp_find_unused_parameters=False,
        fsdp="",
    )

    model = LMWithVectorHead(args.model_path)
    if args.resume_from:
        state_dict = torch.load(args.vector_head_path, map_location="cpu")
        model.vector_head.load_state_dict(state_dict)
    if accelerator.is_main_process:
        print_trainable_params(model)

    trainer = VectorTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collator
    )
    trainer.train()

if __name__ == "__main__":
    main()
