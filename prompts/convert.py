import json
import random
import os
import sys
sys.path.append('../')
from post_process.utils import deal_judge_new

random.seed(42)

prompt_dict = {
    'long_mc_qa': {
        'none': 'Select the correct answer to the following question based on your internal knowledge.\nQuestion: {question}',
        'tail': '\nAnswer: ',
    },
    'long_qa': {
        'none': 'Answer the following question.\nQuestion: {question}',
        'tail': '\nAnswer: ',
    },
    'prior_conf':{
        'none': 'Judge whether you can provide a correct answer for the given question. Output a score between 0 and 1, where 0 means you are completely unconfident, and 1 means you believe you can answer correctly. Just give your score without any other words.\nQuestion: {question}',
        'tail': '\nAnswer: ',
    },
    'consistency':{
        'none': '{question}.',
        'tail': '\nAnswer: ',
    },
    'llm_judge_gold':{
        'none': 'We are assessing the quality of answers to the following question: {question}.\nThe expected answers to this question are: {answer}.\nWithin the context of the question, dose the generated answer mean the same as any of the expected answers?\nThe generated answer is: {prediction}.\nIf the answer is correct, say \"certain\". If not, please say \"uncertain\". Just give your judgement without any other words.',
        'tail': '\nAnswer: ',
    }
}

def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

def write_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f'write jsonl to: {path}')
    f.close()

def get_prompt(prompt_type, input, response="", answer=""):
    prompt = prompt_dict[prompt_type]['none']
    if 'judge_gold' in prompt_type:
        prompt=prompt.format(question=input, prediction=response, answer=answer)
    else:
        prompt = prompt.format(question=input)
    return prompt

class MyDataset:
    def __init__(self, path, dataset_name, n_shot=0) -> None:
        self.dataset_name = dataset_name
        self.n_shot = n_shot
        self.path = path
        print(f'{self.dataset_name}')
        print(f'data path: {path}')
        if '.jsonl' in path:
            self.data = read_json(path)
        elif '.json' in path:
            self.data = json.loads(open(path).read())
        else:
            raise ValueError('need .jsonl or .json file')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> Any:
        return self.data[index]

    def prepare_qa_data(self):
        """
        将每个数据的原始数据转换为qa可用数据
        """
        qa_data = []
        for item in self.data:
            if 'answer' not in item:
                if 'reference' in item:
                    item['answer'] = item['reference']
                else:
                    item['answer'] = ''
            qa_data.append(item)
            
        return qa_data
            
    def prepare_prompts(self, prompt_type):
        prompt_data = self.prepare_qa_data()
        examples = self.prepare_few_shot_examples(prompt_type, prompt_data) if self.n_shot != 0 else "" # prepare few_shot examples (optional) 
        final_data = []            
        for item in prompt_data:
            if 'judge_gold' in prompt_type:
                # prepare prompt for every response
                for tmp_response in item['response']:
                    new_item = item.copy()
                    new_item['instruction'] = get_prompt(prompt_type, item['question'], tmp_response, item['answer'])
                    final_data.append(new_item)
            else:
                item['instruction'] = get_prompt(prompt_type, item['question'])
                item['instruction'] = examples + item['instruction'] # insert few-shot examples (optional)
                final_data.append(item)
        return final_data

    def prepare_few_shot_examples(self, prompt_type, prompt_data):
        # randomly sampling examples from the test set
        if prompt_type != 'prior_conf':
            raise ValueError('Only support few-shot for prior_conf')
        indices = random.sample(range(len(prompt_data)), self.n_shot)
        sampled_data = [prompt_data[i] for i in indices]
        examples = ''
        for idx, item in enumerate(sampled_data):
            question_promt = get_prompt(prompt_type, item['question'])
            long_judge_data = read_json(self.path.replace('.jsonl', '_long_qa_judge.jsonl'))
            group = long_judge_data[indices[idx]*20: indices[idx]*20 + 20]
            confidence = sum([1 - int(deal_judge_new(t['response'][0])) for t in group]) / len(group)
            question_promt += f'\n{confidence}'
            examples += question_promt
            examples += '\n'
        return examples


if __name__ == '__main__':
    pass
            
