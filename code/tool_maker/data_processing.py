import json
import os
import sys
import re
import ast
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from table_utils import extract_table_data_line
from reasoning_executor.metrics import compare_results
import subprocess

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

    def extract_functions(self, code_string):
        functions = []
        lines = code_string.split('\n')
        function_str = ""
        in_function = False

        for line in lines:
            # omit comments
            if line.startswith('#'):
                continue
            # omit print statements
            if line.strip() == 'print(solution(table_data))':
                continue
            if line.startswith('def'):
                if in_function:
                    functions.append(function_str.strip())
                function_str = line + '\n'
                in_function = True
            else:
                function_str += line + '\n'

        # Append the last function
        if in_function:
            functions.append(function_str.strip())

        # Remove the 'def solution' function if present
        # functions = [func for func in functions if not func.startswith('def solution')]

        func_string = '\n\n'.join(functions)
        return func_string

    def get_table_and_code(self):
        dataset = []
        python_files = [f for f in os.listdir(os.path.join(self.code_data_path, self.dataset_name)) if f.endswith('.py') and not f.startswith('tool_library')]
        for file_name in python_files:
            file_path = os.path.join(self.code_data_path, self.dataset_name, file_name)

            with open(file_path, 'r') as file:
                file_content = file.read()  # Read the entire file content
                # extract table data from the file content
                ID = file_name.replace('.py', '').replace('sample', '').strip()
                data_sample = self.raw_data_dict[ID]
                # input_table = linearize_table(data_sample)
                #table_data = extract_table_data_line(file_content).strip()
                table_data = extract_table_data_line(file_content)
                if table_data is None:
                    continue  # Skip this file or handle the absence of table data appropriately
                else:
                    table_data = table_data.strip()

                code_solution = self.extract_functions(file_content).strip()

                # execute the code
                # answer = self.safe_execute_code(code_solution, table_data)
                answer = self.safe_execute_code(file_content.strip())
                if answer is not None:
                    # print(f'Pred: {answer}, GT: {data_sample["answer"]}')
                    if compare_results(str(answer), data_sample['answer']) == True:
                        dataset.append({'sample': data_sample, 'table_data': table_data, 'code_solution': code_solution})

        print(f'{self.dataset_name}: Loaded {len(dataset)} samples for training tool maker.')
        return dataset

    def safe_execute_code(self, code):
        try:
            code = code.replace('print(solution(table_data))', 'answer = solution(table_data)')
            exec(code, globals())
            return globals()['answer']
        except Exception as e:
            return None

    # Format the data sample
    def format_data_sample(self, input_table, full_code):
        # get each part of the prompt
        task_str = f"Given the following table and question, generate the python code to solve it.\n\n{input_table}"
        response_str = full_code

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

    def generate_train_data(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        training_data = []
        table_and_codes = self.get_table_and_code()
        for info in table_and_codes:
            question = info['sample']['question']
            table_data = info['table_data']
            caption = f"Caption = {info['sample']['table_caption']}" if info['sample']['table_caption'] is not None else 'Caption = None'
            input_table = f'{caption}\n{table_data}\nQuestion = {question}'
            full_code = info['code_solution'].strip()

            formatted_sample = self.format_data_sample(input_table, full_code)
            training_data.append({'id': info['sample']['id'], 'text': formatted_sample})
        
        print(f"Number of training samples: {len(training_data)}")
        print(training_data[0]['text'])
        
        with open(os.path.join(save_path, f'{self.dataset_name}_train.jsonl'), 'w') as f:
            for sample in training_data:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    original_data_path = '../data/datasets/'
    code_data_path = '../data/data_sample_code/'
    dataset_name = 'finqa'
    save_path = './data'
    data_generator = Train_Data_Generator(original_data_path, code_data_path, dataset_name)
    data_generator.generate_train_data(save_path)

    # tool_library = load_tool_library(tool_library_path)
    # # print(tool_library.keys())
    # print(tool_library['get_column_by_name'])

    # code = '''def solution(table_data):
    # column_name = 'Number of donors'
    # column_1 = get_column_by_name(table_data, column_name)
    # total = sum_column(column_1)
    # index_1 = get_row_index_by_value(table_data, 'Bronze')
    # bronze = get_column_cell_value(index_1, column_1)
    # answer = divide(bronze, total)
    # return answer'''

    # function_calls = extract_functions_calls(code)
    # print(function_calls)