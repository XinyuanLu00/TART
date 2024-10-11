import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from table_utils import linearize_table

class Train_Data_Generator:
    def __init__(self, original_data_path, code_data_path, dataset_name):
        self.code_data_path = code_data_path
        self.data_path = original_data_path
        self.dataset_name = dataset_name
        self.valid_ids = self.load_valid_ids()
        self.raw_data = self.load_data()

    def load_valid_ids(self):
        valid_ids = set()
        with open(os.path.join(self.code_data_path, f'{self.dataset_name}_train.jsonl'), 'r') as f:
            for line in f:
                data = json.loads(line)
                valid_ids.add(data['id'])
        return valid_ids

    def load_data(self):
        with open(os.path.join(self.data_path, self.dataset_name, 'train.json'), 'r') as f:
            data = json.load(f)
        return [sample for sample in data if sample['id'] in self.valid_ids]

    def format_data_sample(self, sample):
        #TabMWP
        # formatted_data = {"id": sample["id"],"text": f"### INSTRUCTION:\nGiven the following table and question, generate a step-by-step reasoning explanation and the final answer.\n\n"f"{linearize_table(sample)}\n"f"### RESPONSE:\nAnswer: "f"{sample['gt_explanations']} Therefore, the answer is "f"{sample['answer']}\n\n### END"}
        #FINQA:
        formatted_data = {"id": sample["id"],"text": f"### INSTRUCTION:\nGiven a table, context and a question, generate a python program to answer the question.\n\nContext: "f"{sample['context']}"f"{linearize_table(sample)}\n"f"### RESPONSE:\nAnswer: "f"{sample['gt_explanations']}\n\n### END"}

        return formatted_data

    def generate_train_data(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        training_data = []
        for sample in self.raw_data:
            formatted_sample = self.format_data_sample(sample)
            training_data.append(formatted_sample)

        output_file = os.path.join(save_path, f'{self.dataset_name}_train_pot.jsonl')
        with open(output_file, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Generated {len(training_data)} training samples.")

if __name__ == '__main__':
    original_data_path = '../data/datasets/'
    code_data_path = '../tool_maker/data/' 
    dataset_name = 'finqa'
    save_path = './data'
    generator = Train_Data_Generator(original_data_path, code_data_path, dataset_name)
    generator.generate_train_data(save_path)