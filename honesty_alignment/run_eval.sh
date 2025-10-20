#!/bin/bash
set -e

# 安装依赖
pip install peft trl nltk matplotlib tensorboardX orjson pathos openpyxl rouge

# 设置基础路径
base_path="/data/users/nishiyu/res/honesty_alignment/res"

for model_name in Qwen2.5-7B-Instruct
do
    if [ "$model_name" == "Qwen2.5-14B-Instruct" ]; then
        batch_size=4
        accumulation_steps=32
    else
        batch_size=16
        accumulation_steps=8
    fi

    for greedy_samples in 0
    do
        for eval_dataset in nq hq tq 2wikimultihopqa pararel_patterns popqa musique squad web_questions complex_web_questions
        do
            for qa_type in long
            do
                train_dataset="pararel_patterns-nq-tq-hq-2wikimultihopqa"
                training_samples="1000,2000,4000,6000,8000,10000,20000,30000,50000,80000,200000,0"

                python3 eval.py \
                    --model_name ${model_name}  \
                    --train_dataset ${train_dataset} \
                    --eval_dataset ${eval_dataset} \
                    --training_samples ${training_samples} \
                    --greedy_samples ${greedy_samples} \
                    --qa_type ${qa_type} \
                    --batch_size ${batch_size} \
                    --accumulation_steps ${accumulation_steps} \
                    --base_path ${base_path}
            done
        done
    done
done



