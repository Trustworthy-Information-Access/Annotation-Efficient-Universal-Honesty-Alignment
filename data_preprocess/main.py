# prepare data from original datasets
import json
import sys
sys.path.append('../')
from post_process.utils import write_jsonl, read_json
import pandas as pd
import os
import random
import csv
random.seed(0)

def prepare_hotpotqa_for_perception(path, out_path):
    data = json.loads(open(path).read())
    print(len(data))
    print(data[0].keys())
    print(f'question: {data[0]["question"]}, answer: {data[0]["answer"]}')
    new_data = []
    for item in data:
        new_data.append({
            'question': item['question'],
            'answer': [item['answer']]
        })
    write_jsonl(new_data, out_path)

def prepare_2wikimultihop_for_perception(path, out_path=''):
    # test集合没答案, 用dev当test
    data = json.loads(open(path).read())
    print(len(data))
    print(data[0].keys())
    for id in range(5):
        print(f'question: {data[id]["question"]}, answer: {data[id]["answer"]}')
    new_data = []
    for item in data:
        new_data.append({
            'question': item['question'],
            'answer': [item['answer']]
        })
    write_jsonl(new_data, out_path)

def prepare_trivialqa_for_perception(path, out_path=''):
    # test集合没答案, 用dev当test
    data = json.loads(open(path).read())
    print(len(data))
    print(data.keys())
    print(len(data['Data']))
    print(data['Data'][0].keys())
    
    for id in range(5):
        print(f"question: {data['Data'][id]['Question']}, aliases: {data['Data'][id]['Answer']['Aliases']}, answer: {data['Data'][id]['Answer']['Value']}")
    new_data = []
    for item in data['Data']:
        item['Answer']['Aliases'].append(item['Answer']['Value'])
        
        new_data.append({
            'question': item['Question'],
            'answer': list(set(item['Answer']['Aliases']))
        })
    write_jsonl(new_data, out_path)

def prepare_squad_for_perception(path, out_path):
    data = pd.read_parquet(path)
    print(len(data))
    new_data = []
    cnt = 0
    for _, row in data.iterrows():
        new_data.append({
            'question': row['question'],
            'answer': list(set(row['answers']['text'].tolist()))
        })

    write_jsonl(new_data, out_path)

# Add question field to each template
def convert_template_to_question(data):
    def template_to_question(template):
        if template.startswith('[X] is a '):
            return f'What is [X]?'
        elif template.startswith('[X] is the '):
            return f'What is [X]?'
        elif template.startswith('[X] is '):
            return f'What is [X]?'
        elif template.startswith('[X] was '):
            return f'What was [X]?'
        elif template.startswith('[X] has '):
            return f'What position does [X] have?'
        elif template.startswith('[X] plays '):
            return f'What does [X] play?'
        elif template.startswith('[X] works '):
            return f'Where does [X] work?'
        elif template.startswith('[X] used to '):
            return f'What did [X] used to do?'
        elif template.startswith('The '):
            subject = template.split(' of ')[0][4:]
            return f'What is the {subject} of [X]?'
        else:
            # Default question format
            return f'What is the relationship between [X] and [Y]?'

    template = data['template']
    if '[X] is located in [Y]' in template:
        data['question'] = 'Where is [X] located?'
    elif '[X] is a legal term in [Y]' in template:
        data['question'] = 'In what legal system is [X] a term?'
    elif '[X] works in the field of [Y]' in template:
        data['question'] = 'What field does [X] work in?'
    elif 'The native language of [X] is [Y]' in template:
        data['question'] = 'What is the native language of [X]?'
    elif '[X] is a [Y] by profession' in template:
        data['question'] = 'What is [X]\'s profession?'
    elif '[X] works for [Y]' in template:
        data['question'] = 'Who does [X] work for?'
    elif '[X] is owned by [Y]' in template:
        data['question'] = 'Who owns [X]?'
    elif '[X] plays [Y]' in template:
        data['question'] = 'What does [X] play?'
    elif '[X] plays [Y] music' in template:
        data['question'] = 'What type of music does [X] play?'
    elif '[X] is the capital of [Y]' in template:
        data['question'] = 'Of what is [X] the capital?'
    elif '[X] is named after [Y]' in template:
        data['question'] = 'Who/what is [X] named after?'
    elif '[X] is affiliated with the [Y] religion' in template:
        data['question'] = 'What religion is [X] affiliated with?'
    elif '[X] used to communicate in [Y]' in template:
        data['question'] = 'What language did [X] used to communicate in?'
    elif 'The headquarter of [X] is in [Y]' in template:
        data['question'] = 'Where is the headquarters of [X] located?'
    elif '[X] is produced by [Y]' in template:
        data['question'] = 'Who produces [X]?'
    elif '[X] is developed by [Y]' in template:
        data['question'] = 'Who developed [X]?'
    elif '[X] was born in [Y]' in template:
        data['question'] = 'Where was [X] born?'
    elif '[X] and [Y] are twin cities' in template:
        data['question'] = 'What is [X]\'s twin city?'
    elif '[X] died in [Y]' in template:
        data['question'] = 'Where did [X] die?'
    elif '[X] is represented by music label [Y]' in template:
        data['question'] = 'What music label represents [X]?'
    elif '[X] is [Y] citizen' in template:
        data['question'] = 'What is [X]\'s citizenship?'
    elif '[X] is a subclass of [Y]' in template:
        data['question'] = 'What is [X] a subclass of?'
    elif '[X] is part of [Y]' in template:
        data['question'] = 'What is [X] part of?'
    elif 'The original language of [X] is [Y]' in template:
        data['question'] = 'What was the original language of [X]?'
    elif 'The official language of [X] is [Y]' in template:
        data['question'] = 'What is the official language of [X]?'
    elif '[X] has the position of [Y]' in template:
        data['question'] = 'What position does [X] hold?'
    elif '[X] was written in [Y]' in template:
        data['question'] = 'In what language was [X] written?'
    elif '[X] plays in [Y] position' in template:
        data['question'] = 'What position does [X] play in?'
    elif '[X] was originally aired on [Y]' in template:
        data['question'] = 'Where was [X] originally aired?'
    elif '[X] is a member of [Y]' in template:
        data['question'] = 'What is [X] a member of?'
    elif '[X] shares border with [Y]' in template:
        data['question'] = 'What does [X] share a border with?'
    elif '[X] was created in [Y]' in template:
        data['question'] = 'Where was [X] created?'
    elif '[X] maintains diplomatic relations with [Y]' in template:
        data['question'] = 'With whom does [X] maintain diplomatic relations?'
    elif '[X] was founded in [Y]' in template:
        data['question'] = 'Where was [X] founded?'
    elif '[X] used to work in [Y]' in template:
        data['question'] = 'Where did [X] used to work?'
    else:
        data['question'] = template_to_question(template)

def prepare_pararel_for_perception(path):
    data = pd.read_parquet(path)
    base_path=path.replace('train.parquet', '')
    folder_path = base_path + 'graphs_json/'
    file_names = os.listdir(folder_path)
    template_dict = {}
    for name in file_names:
        template_dict[name] = {}
        template_dict[name]['template'] = read_json(folder_path + name)[0]['pattern']
    
    for key, value in template_dict.items():
        print(template_dict[key])
        convert_template_to_question(template_dict[key])
    print(template_dict)

    new_data = []
    cnt = 0
    for _, row in data.iterrows():
        if row['relation'] not in template_dict:
            continue
        question_template = template_dict[row['relation']]['question']
        subject = row['subject']
        answers = [row['object']]
        question = question_template.replace('[X]', subject)

        new_data.append({
            'question': question,
            'answer': answers
        })

    test_data = random.sample(new_data, 3000)
    # the remaining data in new_data
    train_data = [item for item in new_data if item not in test_data]

    write_jsonl(test_data, path.replace('train.parquet', 'test.jsonl'))
    write_jsonl(train_data, path.replace('train.parquet', 'train.jsonl'))

def convert_web_questions_for_perception(path, out_path):
    data = pd.read_parquet(path)
    cnt = 0
    new_data = []
    for _, row in data.iterrows():
        new_data.append({
            'question': row['question'],
            'answer': row['answers'].tolist()
        })
    write_jsonl(new_data, out_path)

def convert_complex_web_questions_for_perception(path, out_path):
    data = json.loads(open(path).read())
    new_data = []
    for item in data:
        question = item['question']
        answers = item['answers'][0]['aliases']
        answers.append(item['answers'][0]['answer'])

        new_data.append({
            'question': question,
            'answer': answers
        })
    write_jsonl(new_data, out_path)

def convert_musique_for_perception(path, out_path):
    data = read_json(path)
    new_data = []
    for item in data:
        question = item['question']
        answers = item['answer_aliases']
        answers.append(item['answer'])
        new_data.append({
            'question': question,
            'answer': answers
        })

    write_jsonl(new_data, out_path)

def conver_popqa_for_perception(path, out_path):
    data = []
    with open(path, 'r', encoding='utf-8') as tsvfile:
        # 创建 reader 对象，指定分隔符为制表符 '\t'
        reader = csv.reader(tsvfile, delimiter='\t')
        # 逐行读取数据
        for row in reader:
            data.append(row)
    print(data[0])
    print(data[1])
    print(data[-1])
    new_data = []
    for item in data[1:]:
        question = item[15]
        answers = json.loads(item[8])
        answers.append(item[3])
        answers.extend(json.loads(item[-1]))
        new_data.append({
            'question': question,
            'answer': list(set(answers))
        })
    write_jsonl(new_data, out_path)

def convert_stategyqa_for_perception(path, out_path):
    data = json.loads(open(path).read())
    new_data = []
    for item in data:
        question = item['question']
        answers = ['Yes'] if item['answer'] == True else ['No']
        new_data.append({
            'question': question,
            'answer': answers
        })
    write_jsonl(new_data, out_path)

def convert_gsm8k_for_perception(path, out_path):
    data = read_json(path)
    new_data = []
    for item in data:
        question = item['question']
        solution = item['answer'].split('####')[0].strip('\n')
        answers = item['answer'].split('####')[1].strip()
        new_data.append({
            'question': question,
            'solution': solution,
            'answer': [answers]
        })
    write_jsonl(new_data, out_path)

def convert_math500_for_perception(path, out_path):
    data = read_json(path)
    new_data = []
    for item in data:
        question = item['problem']
        solution = item['solution']
        answers = item['answer']
        new_data.append({
            'question': question,
            'solution': solution,
            'answer': [answers]
        })
    write_jsonl(new_data, out_path)
        
def convert_aime24_for_perception(path, out_path):
    data = pd.read_parquet(path)
    cnt = 0
    new_data = []
    for _, row in data.iterrows():
        new_data.append({
            'question': row['Problem'],
            'solution': row['Solution'],
            'answer': [row['Answer']]
        })
    write_jsonl(new_data, out_path)

def convert_mmlu_for_perception(path, out_path):
    data = pd.read_parquet(path)
    cnt = 0
    new_data = []
    for _, row in data.iterrows():
        all_choices = ['A', 'B', 'C', 'D']
        question = row['question'] + '\n'
        for idx in range(len(all_choices)):
            question += all_choices[idx] + '.' + row['choices'][idx] + '\n'
        answer = all_choices[row['answer']]
        # print(answer)
        new_data.append({
            'question': question,
            'answer': [answer],
            'subject': row['subject']
        })
    write_jsonl(new_data, out_path)



if __name__ == '__main__':
    base_path = '/data/users/nishiyu/data/hq/'
    in_path = base_path + f'dev.json'
    out_path = base_path + f'dev.jsonl'
    prepare_hotpotqa_for_perception(in_path, out_path)
