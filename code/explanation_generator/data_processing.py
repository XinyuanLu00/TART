import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from table_utils import linearize_table, extract_table_data_line, extract_solution_code_blocks

class Train_Data_Generator:
    def __init__(self, data_path, dataset_name):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.raw_data = self.load_data()

    def load_data(self):
        with open(os.path.join(self.data_path, f'{self.dataset_name}_gpt-4.json'), 'r') as f:
            data = json.load(f)
        return data

    # Format the data sample
    def format_data_sample(self, input_table, function_calls, output_explanations):
        # get each part of the prompt
        task_str = f"Given the following table, question, and python code solution, generate the explanations in natural language embedded with function calls.\n\n{input_table}\n\n{function_calls}"
        response_str = output_explanations

        # format the prompt
        # start = "Read the Instruction below and provide an answer.\n"
        task = f"### INSTRUCTION:\n{task_str.strip()}\n\n"
        response = f"### RESPONSE:\n{response_str.strip()}\n\n"
        end = "### END"

        parts = [part for part in [task, response, end] if part]
        # parts = [part for part in [start, task, response, end] if part]
        formatted_prompt = "".join(parts)
        formatted_prompt = formatted_prompt.replace('\\n', '\n')
        return formatted_prompt

    # def generate_train_data(self, save_path):
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
        
    #     training_data = []
    #     for sample in self.raw_data:
    #         input_table = linearize_table(sample)
    #         output_explanations = sample['exp_with_tools']
    #         #output_explanations = sample['prediction']
    #         function_calls = extract_solution_code_blocks(sample['python_code'])
    #         if function_calls is not None:
    #             formatted_sample = self.format_data_sample(input_table, function_calls, output_explanations)
    #             training_data.append({'id': sample['id'], 'text': formatted_sample})
        
    #     print(f"Number of training samples: {len(training_data)}")
    #     # print(training_data[0]['text'])
    #     with open(os.path.join(save_path, f'{self.dataset_name}_train.jsonl'), 'w') as f:
    #         for sample in training_data:
    #             f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    def generate_train_data(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        training_data = []
        for sample in self.raw_data:
            try:
                input_table = linearize_table(sample)
                # Using 'prediction' as output explanations since 'exp_with_tools' does not exist
                output_explanations = sample['prediction']
                function_calls = extract_solution_code_blocks(sample['python_code']) if 'python_code' in sample else "No function calls provided."
                if function_calls is not None:
                    formatted_sample = self.format_data_sample(input_table, function_calls, output_explanations)
                    training_data.append({'id': sample['id'], 'text': formatted_sample})
            except KeyError as e:
                print(f"Key error for sample ID {sample.get('id', 'Unknown')}: {str(e)} - skipping this sample.")

        print(f"Number of training samples: {len(training_data)}")
        with open(os.path.join(save_path, f'{self.dataset_name}_train.jsonl'), 'w') as f:
            for sample in training_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    data_path = './data_generation/results'
    dataset_name = 'scitab'
    save_path = './data'
    data_generator = Train_Data_Generator(data_path, dataset_name)
    data_generator.generate_train_data(save_path)

