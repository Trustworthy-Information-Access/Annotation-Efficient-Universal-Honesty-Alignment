#!/bin/bash
#inference
pip install datasets
pip install setuptools==69.5.1
pip install -U ray
pip install peft trl nltk matplotlib tensorboardX orjson pathos openpyxl
pip install transformers==4.53.1

# --------------------------------spefify your base_path and model_name---------------------------------
GPU_NUMS=4 # specify how many gpus will be used
BASE_PATH=/data/users/nishiyu # specify your base_path for the models/datasets
RES_PATH=${BASE_PATH}/res/honesty_alignment/res # specify your out_dir

model_name=Qwen2.5-7B-Instruct # Specify your model_name, used for QA
model_path=/data/models/${model_name}

# the model useds for consistency checking and correctness judgement
check_model_name=Qwen2.5-32B-Instruct # used for consistency_checking & correctness judgement
check_model_path=/data/models/${model_name}

batchsize=1024



# --------------------------------run qa_inference---------------------------------
if [[ "$model_name" == *"Qwen2.5"* ]]; then # used for the model-specific chat template
    template=qwen2 # used for get prompts
elif [[ "$model_name" == *"Llama-3"* ]]; then
    template=llama3
fi
# greedy answer & sampled answer
for decoding_type in greedy sampling
do
    for prompt_type in long_qa # get prompt for answering the question (also support short_qa)
    do
        if [[ "$decoding_type" == "greedy" ]]; then
            temperature=0.0
            sample_num=1 # sampling count
            logprobs=1 # get generated probability for tokens in the greedy response
            get_tokens=1
        else
            temperature=1.0
            sample_num=20
            logprobs=0
            get_tokens=0
        fi

        if [[ "$prompt_type" == "long_qa" ]]; then
            max_tokens=256
        else
            max_tokens=64
        fi

        topp=0.95
        topk=50
        repetition_penalty=1.05
        tp_size=${GPU_NUMS}
        # for dataset in nq hq tq 2wikimultihopqa pararel_patterns popqa musique squad web_questions complex_web_questions
        for dataset in nq hq tq 2wikimultihopqa pararel_patterns popqa musique squad web_questions complex_web_questions # specify your datasets
        do 
            for data_type in test train
            do
                if [[ "$dataset" =~ ^(popqa|musique|squad|web_questions|complex_web_questions)$ && "$data_type" == "train" ]]; then
                    continue
                fi
                
                if [[ "$data_type" == "test" && "$dataset" =~ ^(2wikimultihopqa|hq|tq|musique|squad|complex_web_questions)$ ]]; then
                    input_path="${BASE_PATH}/data/${dataset}/dev.jsonl"
                else
                    input_path="${BASE_PATH}/data/${dataset}/${data_type}.jsonl"
                fi
                
                output_path="${RES_PATH}/${model_name}/${dataset}/${data_type}_data/${prompt_type}/${dataset}_${data_type}_${model_name}_${prompt_type}_${temperature}_${topp}_${topk}_sample_${sample_num}.jsonl"

                echo "Task $prompt_type"
                echo "Processing $dataset..."
                echo "Data_type $data_type"
                echo "Input: $input_path"
                echo "Output: $output_path"

                # --------------------------------以下不需要使用者修改---------------------------------

                python3 ../tools/vllm_infer_distributed_no_ray.py --template $template \
                                            --model_path $model_path \
                                            --tp_size $tp_size \
                                            --input_path $input_path \
                                            --output_path $output_path \
                                            --temperature $temperature \
                                            --topp $topp \
                                            --topk $topk \
                                            --max_tokens $max_tokens \
                                            --repetition_penalty $repetition_penalty \
                                            --sample_num $sample_num \
                                            --batchsize $batchsize \
                                            --prompt_type $prompt_type \
                                            --logprobs $logprobs \
                                            --get_tokens $get_tokens \
                                            --dataset_name $dataset
            done
        done
    done
done

# prepare data for consistency checking
cd ..
cd prompts
for dataset in nq hq tq 2wikimultihopqa pararel_patterns popqa musique squad web_questions complex_web_questions
do
    for data_type in test train
    do
        if [[ "$dataset" =~ ^(popqa|musique|squad|web_questions|complex_web_questions)$ && "$data_type" == "train" ]]; then
            continue
        fi

        for qa_type in long_qa 
        do  
            python3 self_consistency.py --dataset $dataset \
                                        --data_type $data_type \
                                        --qa_type $qa_type \
                                        --model_name $model_name \
                                        --res_path $RES_PATH
        done
    done
done

# run_consistency_checking
cd ..
cd infer_scripts
template=qwen2
prompt_type=consistency
temperature=0.0
topp=0.95
topk=50
max_tokens=64
repetition_penalty=1.05
sample_num=1 # 采样次数
logprobs=0
get_tokens=0
tp_size=${GPU_NUMS}
qa_model=${model_name}

for dataset in nq hq tq 2wikimultihopqa pararel_patterns popqa musique squad web_questions complex_web_questions
do 
    for data_type in test train
    do
        if [[ "$dataset" =~ ^(popqa|musique|squad|web_questions|complex_web_questions)$ && "$data_type" == "train" ]]; then
            continue
        fi

        for qa_type in long_qa
        do
            input_path="${RES_PATH}/${qa_model}/${dataset}/${data_type}_data/${qa_type}/${dataset}_${data_type}_${qa_model}_${qa_type}_1.0_0.95_50_sample_20_for_greedy_consistency_check.jsonl"
            output_path="${RES_PATH}/${qa_model}/${dataset}/${data_type}_data/${qa_type}/${dataset}_${data_type}_${qa_model}_${qa_type}_1.0_0.95_50_sample_20_for_greedy_consistency_res.jsonl"

            echo "Prompt_type $prompt_type"
            echo "Processing $dataset..."
            echo "Input: $input_path"
            echo "Output: $output_path"
            echo "Data_type $data_type"
            echo "Qa_type $qa_type"

            python3 ../tools/vllm_infer_distributed_no_ray.py --template $template \
                                        --model_path $check_model_path \
                                        --tp_size $tp_size \
                                        --input_path $input_path \
                                        --output_path $output_path \
                                        --temperature $temperature \
                                        --topp $topp \
                                        --topk $topk \
                                        --max_tokens $max_tokens \
                                        --repetition_penalty $repetition_penalty \
                                        --sample_num $sample_num \
                                        --batchsize $batchsize \
                                        --prompt_type $prompt_type \
                                        --logprobs $logprobs \
                                        --get_tokens $get_tokens \
                                        --dataset_name $dataset
        done
    done
done


# # correctness judgement
template=qwen2
prompt_type=llm_judge_gold
temperature=0.0
topp=0.95
topk=50
max_tokens=64
repetition_penalty=1.05
sample_num=1 # 采样次数
logprobs=0
get_tokens=0

tp_size=${GPU_NUMS}
qa_model=${model_name}
for dataset in nq hq tq 2wikimultihopqa pararel_patterns popqa musique squad web_questions complex_web_questions
do 
    for data_type in test train
    do
        if [[ "$dataset" =~ ^(popqa|musique|squad|web_questions|complex_web_questions)$ && "$data_type" == "train" ]]; then
            continue
        fi

        for qa_type in long_qa
        do
            for sample_type in greedy sample
            do
                if [ "$sample_type" = 'sample' ]; then
                    file_sample_num=20
                    sample_temperature=1.0
                else
                    file_sample_num=1
                    sample_temperature=0.0
                fi
                input_path="/data/users/nishiyu/res/honesty_alignment/res/${qa_model}/${dataset}/${data_type}_data/${qa_type}/${dataset}_${data_type}_${qa_model}_${qa_type}_${sample_temperature}_0.95_50_sample_${file_sample_num}.jsonl"
                output_path="/data/users/nishiyu/res/honesty_alignment/res/${qa_model}/${dataset}/${data_type}_data/${qa_type}/${dataset}_${data_type}_${qa_model}_${qa_type}_${sample_temperature}_0.95_50_sample_${file_sample_num}_long_qa_judge.jsonl"

                echo "Prompt_type $prompt_type"
                echo "Processing $dataset..."
                echo "Input: $input_path"
                echo "Output: $output_path"
                echo "Data_type $data_type"
                echo "Qa_type $qa_type"
                echo "Sample_type $sample_type"

                python3 ../tools/vllm_infer_distributed_no_ray.py --template $template \
                                            --model_path $check_model_path \
                                            --tp_size $tp_size \
                                            --input_path $input_path \
                                            --output_path $output_path \
                                            --temperature $temperature \
                                            --topp $topp \
                                            --topk $topk \
                                            --max_tokens $max_tokens \
                                            --repetition_penalty $repetition_penalty \
                                            --sample_num $sample_num \
                                            --batchsize $batchsize \
                                            --prompt_type $prompt_type \
                                            --logprobs $logprobs \
                                            --get_tokens $get_tokens \
                                            --dataset_name $dataset 
            done
        done
    done
done