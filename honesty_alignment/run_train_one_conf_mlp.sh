#!/bin/bash
set -e
# 安装依赖
pip install peft trl nltk matplotlib tensorboardX scikit-learn orjson pathos wandb
batch_size=16
accumulation_steps=8
weight_decay=0.1
lora_dropout=0.0
model_name=Qwen2.5-7B-Instruct
base_path="/data/users/nishiyu/res/honesty_alignment/res/${model_name}"
MAIN_PORT=2223

# pretrain greedy
conf_type=greedy
dataset="pararel_patterns-nq-tq-hq-2wikimultihopqa"
for training_samples in 0
do
    if [ "$training_samples" -eq 0 ]; then
        tail_name=""
        epochs=10
    elif [ "$training_samples" -le 10000 ]; then
        k_val=$((training_samples / 1000))
        tail_name="_${k_val}k_training_samples"
        epochs=50
    else
        k_val=$((training_samples / 1000))
        tail_name="_${k_val}k_training_samples"
        epochs=15
    fi

    for qa_type in long
    do
        outdir="batchsize${batch_size}_accumulation${accumulation_steps}_epochs${epochs}_weightdecay${weight_decay}_mlp_1_layer${tail_name}"
        accelerate launch --main_process_port ${MAIN_PORT} --config_file ./ddp.yaml \
        train_mlp.py \
            --model_path "/data/models/${model_name}" \
            --train_data_path "${base_path}/${dataset}/train_data/${qa_type}_qa/${dataset}_train_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20.jsonl" \
            --test_data_path "${base_path}/${dataset}/test_data/${qa_type}_qa/${dataset}_test_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20.jsonl" \
            --train_consis_path "${base_path}/${dataset}/train_data/${qa_type}_qa/${dataset}_train_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20_for_greedy_consistency_res.jsonl" \
            --test_consis_path "${base_path}/${dataset}/test_data/${qa_type}_qa/${dataset}_test_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20_for_greedy_consistency_res.jsonl" \
            --output_path "${base_path}/models/${dataset}/mlp/${conf_type}_answer_conf/${qa_type}_qa/${outdir}/" \
            --logger_path "${base_path}/logs/${dataset}/mlp/${conf_type}_answer_conf/${qa_type}_qa/${outdir}/" \
            --per_device_train_batch_size $batch_size \
            --gradient_accumulation_steps $accumulation_steps \
            --num_train_epochs $epochs \
            --weight_decay $weight_decay \
            --train_data_count $training_samples \
            --dataset $dataset
    done
done


# -----------------------------------------------------------------------------
# hybrid

conf_type=hybrid
dataset="pararel_patterns-nq-tq-hq-2wikimultihopqa"
for greedy_samples in 0
do
    if [ "$greedy_samples" -eq 0 ]; then
        greedy_tail_name=""
        greedy_epochs=10
    elif [ "$greedy_samples" -le 10000 ]; then
        k_val=$((greedy_samples / 1000))
        greedy_tail_name="_${k_val}k_training_samples"
        greedy_epochs=50
    else
        k_val=$((greedy_samples / 1000))
        greedy_tail_name="_${k_val}k_training_samples"
        greedy_epochs=15
    fi

    for training_samples in 1000 2000 4000 6000 8000 10000 20000 30000 50000 80000 200000 0
    do
        if [ "$training_samples" -eq 0 ]; then
            tail_name=""
            epochs=10
        elif [ "$training_samples" -le 10000 ]; then
            k_val=$((training_samples / 1000))
            tail_name="_${k_val}k_training_samples"
            epochs=50
        else
            k_val=$((training_samples / 1000))
            tail_name="_${k_val}k_training_samples"
            epochs=15
        fi

        for qa_type in long
        do
            if [ "$qa_type" == "long" ]; then
                train_consis_path="${base_path}/${dataset}/train_data/${qa_type}_qa/${dataset}_train_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20_long_qa_judge.jsonl"
                test_consis_path="${base_path}/${dataset}/test_data/${qa_type}_qa/${dataset}_test_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20_long_qa_judge.jsonl"
            else
                train_consis_path="nothing"
                test_consis_path="nothing"
            fi

            outdir="batchsize${batch_size}_accumulation${accumulation_steps}_epochs${epochs}_weightdecay${weight_decay}_mlp_1_layer${tail_name}"
            accelerate launch --main_process_port ${MAIN_PORT} --config_file ./ddp.yaml \
            train_mlp.py \
                --model_path "/data/models/${model_name}" \
                --train_data_path "${base_path}/${dataset}/train_data/${qa_type}_qa/${dataset}_train_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20.jsonl" \
                --test_data_path "${base_path}/${dataset}/test_data/${qa_type}_qa/${dataset}_test_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20.jsonl" \
                --train_consis_path $train_consis_path \
                --test_consis_path $test_consis_path \
                --output_path "${base_path}/models/${dataset}/mlp/${conf_type}_answer_conf${greedy_tail_name}/${qa_type}_qa/${outdir}/" \
                --logger_path "${base_path}/logs/${dataset}/mlp/${conf_type}_answer_conf${greedy_tail_name}/${qa_type}_qa/${outdir}/" \
                --per_device_train_batch_size $batch_size \
                --gradient_accumulation_steps $accumulation_steps \
                --num_train_epochs $epochs \
                --weight_decay $weight_decay \
                --train_data_count $training_samples \
                --resume_from \
                --vector_head_path "${base_path}/models/${dataset}/mlp/greedy_answer_conf/${qa_type}_qa/batchsize${batch_size}_accumulation${accumulation_steps}_epochs${greedy_epochs}_weightdecay${weight_decay}_mlp_1_layer${greedy_tail_name}/best-checkpoint/vector_head_epoch_best.pt" \
                --dataset $dataset \
                --shuffle
        done
    done
done

# right sft
conf_type=right
dataset="pararel_patterns-nq-tq-hq-2wikimultihopqa"
for greedy_samples in 0
do
    if [ "$greedy_samples" -eq 0 ]; then
        greedy_tail_name=""
        greedy_epochs=10
    elif [ "$greedy_samples" -le 10000 ]; then
        k_val=$((greedy_samples / 1000))
        greedy_tail_name="_${k_val}k_training_samples"
        greedy_epochs=50
    else
        k_val=$((greedy_samples / 1000))
        greedy_tail_name="_${k_val}k_training_samples"
        greedy_epochs=15
    fi

    for training_samples in 1000 2000 4000 6000 8000 10000 20000 30000 50000 80000 200000 0
    do
        if [ "$training_samples" -eq 0 ]; then
            tail_name=""
            epochs=10
        elif [ "$training_samples" -le 10000 ]; then
            k_val=$((training_samples / 1000))
            tail_name="_${k_val}k_training_samples"
            epochs=50
        else
            k_val=$((training_samples / 1000))
            tail_name="_${k_val}k_training_samples"
            epochs=15
        fi

        for qa_type in long
        do
            if [ "$qa_type" == "long" ]; then
                train_consis_path="${base_path}/${dataset}/train_data/${qa_type}_qa/${dataset}_train_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20_long_qa_judge.jsonl"
                test_consis_path="${base_path}/${dataset}/test_data/${qa_type}_qa/${dataset}_test_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20_long_qa_judge.jsonl"
            else
                train_consis_path="nothing"
                test_consis_path="nothing"
            fi

            outdir="batchsize${batch_size}_accumulation${accumulation_steps}_epochs${epochs}_weightdecay${weight_decay}_mlp_1_layer${tail_name}"
            accelerate launch --main_process_port ${MAIN_PORT} --config_file ./ddp.yaml \
            train_mlp.py \
                --model_path "/data/models/${model_name}" \
                --train_data_path "${base_path}/${dataset}/train_data/${qa_type}_qa/${dataset}_train_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20.jsonl" \
                --test_data_path "${base_path}/${dataset}/test_data/${qa_type}_qa/${dataset}_test_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20.jsonl" \
                --train_consis_path $train_consis_path \
                --test_consis_path $test_consis_path \
                --output_path "${base_path}/models/${dataset}/mlp/${conf_type}_answer_conf${greedy_tail_name}/${qa_type}_qa/${outdir}/" \
                --logger_path "${base_path}/logs/${dataset}/mlp/${conf_type}_answer_conf${greedy_tail_name}/${qa_type}_qa/${outdir}/" \
                --per_device_train_batch_size $batch_size \
                --gradient_accumulation_steps $accumulation_steps \
                --num_train_epochs $epochs \
                --weight_decay $weight_decay \
                --train_data_count $training_samples \
                --dataset $dataset \
                --shuffle
        done
    done
done

# eval---------------------------------------------------------
# Elicitation
for greedy_samples in 0; do
    if [ "$greedy_samples" -eq 0 ]; then
        greedy_tail_name=""
        greedy_epochs=10
    elif [ "$greedy_samples" -le 10000 ]; then
        k_val=$((greedy_samples / 1000))
        greedy_tail_name="_${k_val}k_training_samples"
        greedy_epochs=50
    else
        k_val=$((greedy_samples / 1000))
        greedy_tail_name="_${k_val}k_training_samples"
        greedy_epochs=15
    fi
    base_parameters="batchsize${batch_size}_accumulation${accumulation_steps}_epochs${greedy_epochs}_weightdecay${weight_decay}_mlp_1_layer"

    for dataset in pararel_patterns-nq-tq-hq-2wikimultihopqa; do
        for conf_type in greedy; do
            for qa_type in long; do
            # 初始化数组（每次外层循环开始时清空）
            test_data_paths=()
            train_data_paths=()
            output_paths=()

            # 遍历所有 test_dataset，收集它们的路径
            for test_dataset in nq hq tq 2wikimultihopqa pararel_patterns complex_web_questions musique web_questions popqa squad; do
                test_data_paths+=("${base_path}/${test_dataset}/test_data/${qa_type}_qa/${test_dataset}_test_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20.jsonl")
                train_data_paths+=("${base_path}/${test_dataset}/test_data/${qa_type}_qa/${test_dataset}_test_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20.jsonl")
                output_paths+=("${base_path}/eval/${dataset}/mlp/${test_dataset}/${conf_type}_answer_conf/${qa_type}_qa/${base_parameters}/")
            done

            # 一次性传入所有 test_dataset 的路径
            accelerate launch --main_process_port ${MAIN_PORT} --config_file ./ddp_eval.yaml \
            eval_one_conf_mlp.py \
                --model_path "/data/models/${model_name}" \
                --test_data_path "${test_data_paths[@]}" \
                --train_data_path "${train_data_paths[@]}" \
                --output_path "${output_paths[@]}" \
                --vector_head_path "${base_path}/models/${dataset}/mlp/${conf_type}_answer_conf/${qa_type}_qa/${base_parameters}/best-checkpoint/vector_head_epoch_best.pt" \
                --per_device_eval_batch_size $eval_batchsize
            done
        done
    done
done

# EliCal
for greedy_samples in 0; do
    if [ "$greedy_samples" -eq 0 ]; then
        greedy_tail_name=""
        greedy_epochs=10
    elif [ "$greedy_samples" -le 10000 ]; then
        k_val=$((greedy_samples / 1000))
        greedy_tail_name="_${k_val}k_training_samples"
        greedy_epochs=50
    else
        k_val=$((greedy_samples / 1000))
        greedy_tail_name="_${k_val}k_training_samples"
        greedy_epochs=15
    fi

    for dataset in pararel_patterns-nq-tq-hq-2wikimultihopqa; do
        for conf_type in hybrid; do
            for training_samples in 1000 2000 4000 6000 8000 10000 20000 30000 50000 80000 200000 0; do
                if [ "$training_samples" -eq 0 ]; then
                    tail_name=""
                    epochs=10
                elif [ "$training_samples" -le 10000 ]; then
                    k_val=$((training_samples / 1000))
                    tail_name="_${k_val}k_training_samples"
                    epochs=50
                else
                    k_val=$((training_samples / 1000))
                    tail_name="_${k_val}k_training_samples"
                    epochs=15
                fi
                base_parameters="batchsize${batch_size}_accumulation${accumulation_steps}_epochs${epochs}_weightdecay${weight_decay}_mlp_1_layer${tail_name}"
                for qa_type in long; do
                    # 初始化数组
                    test_data_paths=()
                    train_data_paths=()
                    output_paths=()

                    # 收集所有 test_dataset 的路径
                    for test_dataset in nq hq tq 2wikimultihopqa pararel_patterns complex_web_questions musique web_questions popqa squad; do
                        test_data_paths+=("${base_path}/${test_dataset}/test_data/${qa_type}_qa/${test_dataset}_test_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20.jsonl")
                        train_data_paths+=("${base_path}/${test_dataset}/test_data/${qa_type}_qa/${test_dataset}_test_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20.jsonl")
                        output_paths+=("${base_path}/eval/${dataset}/mlp/${test_dataset}/${conf_type}_answer_conf${greedy_tail_name}/${qa_type}_qa/${base_parameters}/")
                    done

                    # 一次性调用 eval_one_conf.py
                        accelerate launch --main_process_port ${MAIN_PORT} --config_file ./ddp_eval.yaml \
                        eval_one_conf_mlp.py \
                        --model_path "/data/models/${model_name}" \
                        --test_data_path "${test_data_paths[@]}" \
                        --train_data_path "${train_data_paths[@]}" \
                        --output_path "${output_paths[@]}" \
                        --vector_head_path "${base_path}/models/${dataset}/mlp/${conf_type}_answer_conf${greedy_tail_name}/${qa_type}_qa/${base_parameters}/best-checkpoint/vector_head_epoch_best.pt" \
                        --per_device_eval_batch_size $eval_batchsize
                done
            done
        done
    done
done


# Calibration-only
for dataset in pararel_patterns-nq-tq-hq-2wikimultihopqa; do
    for conf_type in right; do
        for training_samples in 1000 2000 4000 6000 8000 10000 20000 30000 50000 80000 200000 0; do
            if [ "$training_samples" -eq 0 ]; then
                tail_name=""
                epochs=10
            elif [ "$training_samples" -le 10000 ]; then
                k_val=$((training_samples / 1000))
                tail_name="_${k_val}k_training_samples"
                epochs=50
            else
                k_val=$((training_samples / 1000))
                tail_name="_${k_val}k_training_samples"
                epochs=15
            fi

            for qa_type in long; do
                base_parameters="batchsize${batch_size}_accumulation${accumulation_steps}_epochs${epochs}_weightdecay${weight_decay}_mlp_1_layer${tail_name}"
                
                # 初始化数组
                test_data_paths=()
                train_data_paths=()
                output_paths=()

                # 收集所有 test_dataset 的路径
                for test_dataset in nq hq tq 2wikimultihopqa pararel_patterns complex_web_questions musique web_questions popqa squad; do
                    test_data_paths+=("${base_path}/${test_dataset}/test_data/${qa_type}_qa/${test_dataset}_test_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20.jsonl")
                    train_data_paths+=("${base_path}/${test_dataset}/test_data/${qa_type}_qa/${test_dataset}_test_${model_name}_${qa_type}_qa_1.0_0.95_50_sample_20.jsonl")
                    output_paths+=("${base_path}/eval/${dataset}/mlp/${test_dataset}/${conf_type}_answer_conf/${qa_type}_qa/${base_parameters}/")
                done

                # 一次性调用 eval_one_conf.py
                    accelerate launch --main_process_port ${MAIN_PORT} --config_file ./ddp_eval.yaml \
                    eval_one_conf_mlp.py \
                    --model_path "/data/models/${model_name}" \
                    --test_data_path "${test_data_paths[@]}" \
                    --train_data_path "${train_data_paths[@]}" \
                    --output_path "${output_paths[@]}" \
                    --vector_head_path "${base_path}/models/${dataset}/mlp/${conf_type}_answer_conf${greedy_tail_name}/${qa_type}_qa/${base_parameters}/best-checkpoint/vector_head_epoch_best.pt" \
                    --per_device_eval_batch_size $eval_batchsize
            done
        done
    done
done
