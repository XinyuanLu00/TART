import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from table_utils import linearize_table

class Train_Data_Generator:
    def __init__(self, original_data_path, code_data_path, dataset_name, exp_data_path):
        self.code_data_path = code_data_path
        self.data_path = original_data_path
        self.exp_data_path = exp_data_path
        self.dataset_name = dataset_name
        self.valid_ids = self.load_valid_ids()
        self.raw_data = self.load_data()
        self.exp_data = self.load_exp_data()

    def load_valid_ids(self):
        valid_ids = set()
        with open(os.path.join(self.code_data_path, f'{self.dataset_name}_train.jsonl'), 'r') as f:
            for line in f:
                data = json.loads(line)
                valid_ids.add(data['id'])
        print(f"Loaded {len(valid_ids)} valid IDs.")
        return valid_ids

    def load_data(self):
        with open(os.path.join(self.data_path, self.dataset_name, 'train.json'), 'r') as f:
            data = json.load(f)
        filtered_data = [sample for sample in data if sample['id'] in self.valid_ids]
        print(f"Loaded {len(filtered_data)} samples from dataset.")
        return filtered_data

    def load_exp_data(self):
        with open(self.exp_data_path, 'r') as f:
            exp_data = json.load(f)
        id_to_exp = {item['id']: item['explanation'] for item in exp_data if 'explanation' in item}
        print(f"Loaded {len(id_to_exp)} explanations from results.")
        return id_to_exp

    def format_data_sample(self, sample):
        explanation = self.exp_data.get(sample['id'], "No explanation available")
        #FINQA:
        # formatted_data = {"id": sample["id"], "text": f"### INSTRUCTION:\nGiven the following table, context and question, generate a step-by-step reasoning explanation and the final answer.\n\nContext: {sample['context']}{linearize_table(sample)}\n### RESPONSE:\nAnswer: {explanation} Therefore, the answer is {sample['answer']}\n\n### END"}
        formatted_data = {"id": sample["id"], "text": f"### INSTRUCTION:\nGiven the following table, and question, generate a step-by-step reasoning explanation and the final answer.\n\n{linearize_table(sample)}\n### RESPONSE:\nAnswer: {explanation}\n\n### END"}

        return formatted_data

    def generate_train_data(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        training_data = [self.format_data_sample(sample) for sample in self.raw_data]
        output_file = os.path.join(save_path, f'{self.dataset_name}_train.jsonl')
        with open(output_file, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Generated {len(training_data)} training samples.")

if __name__ == '__main__':
    original_data_path = '../data/datasets/'
    code_data_path = '../tool_maker/data/'
    exp_data_path = './cot_creation/results/scitab_gpt-4.json'
    dataset_name = 'scitab'
    save_path = './data'
    generator = Train_Data_Generator(original_data_path, code_data_path, dataset_name, exp_data_path)
    generator.generate_train_data(save_path)
