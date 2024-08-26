import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from table_utils import linearize_table, extract_table_data_line

class Train_Data_Generator:
    def __init__(self, original_data_path, code_data_path, dataset_name):
        self.code_data_path = code_data_path
        self.data_path = original_data_path
        self.dataset_name = dataset_name
        self.raw_data_dict = self.load_original_data()

    def load_original_data(self):
        with open(os.path.join(self.data_path, self.dataset_name, 'train.json'), 'r') as f:
            data = json.load(f)
        data_dict = {sample['id']: sample for sample in data}
        return data_dict

    # Format the data sample
    def format_data_sample(self, input_table, table_data):
        # get each part of the prompt
        task_str = f"Given the following table and question, format the table into a python array.\n\n{input_table}"
        response_str = table_data

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

    def get_table_and_code(self):
        dataset = []
        python_files = [f for f in os.listdir(os.path.join(self.code_data_path, self.dataset_name)) if f.endswith('.py') and not f.startswith('tool_library')]
        for file_name in python_files:
            file_path = os.path.join(self.code_data_path, self.dataset_name, file_name)

            with open(file_path, 'r') as file:
                file_content = file.read()  # Read the entire file content

                # Extract table data from the file content
                ID = file_name.replace('.py', '').replace('sample', '').strip()

                if ID not in self.raw_data_dict:
                    print(f"Warning: No data for ID {ID}")
                    continue  # Skip this file if ID not found

                data_sample = self.raw_data_dict[ID]
                input_table = linearize_table(data_sample)

                table_data = extract_table_data_line(file_content)
                if table_data is None:
                    print(f"Warning: No table data extracted for ID {ID}")
                    continue  # Skip this file if no table data extracted

                table_data = table_data.strip()
                dataset.append({'id': ID, 'input_table': input_table, 'table_data': table_data})

        # Debugging output to check how many samples were loaded
        print(f'{self.dataset_name}: Loaded {len(dataset)} samples for training table formatter.')
        return dataset


    def generate_train_data(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        training_data = []
        table_and_codes = self.get_table_and_code()
        for sample in table_and_codes:
            formatted_sample = self.format_data_sample(sample['input_table'], sample['table_data'])
            training_data.append({'id': sample['id'], 'text': formatted_sample})
        
        print(f"Number of training samples: {len(training_data)}")
        print(training_data[0]['text'])
        with open(os.path.join(save_path, f'{self.dataset_name}_train.jsonl'), 'w') as f:
            for sample in training_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    original_data_path = '../data/datasets/'
    code_data_path = '../data/data_sample_code/'
    dataset_name = 'scitab'
    save_path = './data'
    data_generator = Train_Data_Generator(original_data_path, code_data_path, dataset_name)
    data_generator.generate_train_data(save_path)

