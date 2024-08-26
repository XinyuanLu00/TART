import sys
import os
import argparse
import json
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llama_training.lora_inference import LoRA_Inference
from table_utils import linearize_table

def parse_args():
    parser = argparse.ArgumentParser()
    # dataset arguments
    parser.add_argument("--data_path", type=str, help="the path to the dataset")
    parser.add_argument("--output_path", type=str, help="the path to save the output")
    parser.add_argument("--num_samples", type=int, default=100, help="the number of samples to load")
    parser.add_argument("--batch_size", type=int, default=10, help="the inference batch size")
    # model arguments
    parser.add_argument("--base_model_name", type=str, help="the base model name")
    parser.add_argument("--checkpoint_path", type=str, help="the checkpoint path")
    parser.add_argument("--cache_dir", type=str, default=None, help="The cache directory to save the model")
    parser.add_argument("--load_in_8bit", action='store_true', help="load the model in 8 bits precision")
    parser.add_argument("--load_in_4bit", action='store_true', help="load the model in 4 bits precision")
    parser.add_argument("--trust_remote_code", type=bool, default=True, help="Enable `trust_remote_code`")
    # parser.add_argument("--use_auth_token", type=bool, default=True, help="Use HF auth token to access the model")
    parser.add_argument("--huggingface_token", type=str, default=None, help="The HF auth token")
    # inference arguments
    parser.add_argument("--batch_inference", action='store_true', help="Enable batch inference")
    parser.add_argument("--do_sample", type=bool, default=True, help="Enable sampling")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling softmax temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="The maximum number of tokens to generate")
    args = parser.parse_args()
    return args

class ExpGenInference:
    def __init__(self, args) -> None:
        self.args = args
        self.data_path = args.data_path
        self.dataset = self.load_data()
        self.lora_inference = LoRA_Inference(self.args)

    def extract_solution_code(self, content):
        solution_start = content.find("def solution(")
        if solution_start != -1:
            solution_block = content[solution_start:]
            return solution_block
        return None

    # Format the data sample
    def format_data_sample(self, sample):
        # get input table
        input_table = linearize_table(sample)
        function_calls = self.extract_solution_code(sample['tool_maker_output'])
        
        # get each part of the prompt
        task_str = f"Given the following table, question, and python code solution, generate the explanations in natural language embedded with function calls.\n\n{input_table}\n\n{function_calls}"

        # format the prompt
        task = f"### INSTRUCTION:\n{task_str.strip()}\n\n"
        response = f"### RESPONSE:\n"

        parts = [part for part in [task, response] if part]
        formatted_prompt = "".join(parts)
        formatted_prompt = formatted_prompt.replace('\\n', '\n')
        return formatted_prompt

    def load_data(self):
        with open(self.data_path, "r") as f:
            data = json.load(f)
        dataset = []
        for sample in data:
            sample['explanation_input'] = self.format_data_sample(sample)
            # print(sample['explanation_input'])
            dataset.append(sample)

        dataset = dataset if self.args.num_samples == -1 else dataset[:self.args.num_samples]
        print(f"Loaded {len(dataset)} samples")
        return dataset

    def post_process_output(self, output):
        output = output.split('### END')[0].strip()
        return output

    def inference_single_sample(self, sample):
        input_sample = self.format_data_sample(sample)
        result = self.lora_inference.generate([input_sample])[0]
        result = self.post_process_output(result)
        return result

    def inference_on_dataset(self):
        output_data = []
        print("Start inference...")
        
        # batch inference
        if self.args.batch_inference:
            for i in tqdm(range(0, len(self.dataset), self.args.batch_size)):
                batch = self.dataset[i:i+self.args.batch_size]
                input_str = [sample['explanation_input'] for sample in batch]
                output_str = self.lora_inference.generate_batch(input_str)
                for j, sample in enumerate(batch):
                    sample['explanation_output'] = self.post_process_output(output_str[j])
                    output_data.append(sample)
        else:
            # # single inference
            for sample in tqdm(self.dataset):
                input_str = sample['explanation_input']
                output_str = self.lora_inference.generate([input_str])
                sample['explanation_output'] = self.post_process_output(output_str[0])
                #print('explanation_input:', sample['explanation_input'],'explanation_output:', sample['explanation_output'])

                output_data.append(sample)

        # save the output
        with open(self.args.output_path, "w") as f:
            json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    data_inference = ExpGenInference(args)
    data_inference.inference_on_dataset()

    # single_inference = ExpGenInference(args)
    # sample = {
    #     "id": "8",
    #     "table_caption": "Cookies baked",
    #     "table_column_names": [
    #         "Day",
    #         "Number of cookies"
    #     ],
    #     "table_content_values": [
    #         [
    #             "Friday",
    #             "163"
    #         ],
    #         [
    #             "Saturday",
    #             "281"
    #         ],
    #         [
    #             "Sunday",
    #             "263"
    #         ]
    #     ],
    #     "question": "Hannah baked cookies each day for a bake sale. How many more cookies did Hannah bake on Saturday than on Sunday?",
    #     "answer": "18",
    #     "gt_explanations": "Find the numbers in the table.\n\nSaturday: 281\nSunday: 263\n\nNow subtract: 281 - 263 = 18.\n\nHannah baked 18 more cookies on Saturday.",
    #     "table_formatter_input": "### INSTRUCTION:\nGiven the following table and question, format the table into a python array.\n\nCaption: Cookies baked\nTable:\n|| Friday | 163 ||\n|| Saturday | 281 ||\n|| Sunday | 263 ||\n\nQuestion: Hannah baked cookies each day for a bake sale. How many more cookies did Hannah bake on Saturday than on Sunday?\n\n### RESPONSE:\n",
    #     "table_formatter_output": "table_data = [['Day', 'Number of cookies'], ['Friday', 163], ['Saturday', 281], ['Sunday', 263]]",
    #     "tool_maker_input": "### INSTRUCTION:\nGiven the following table and question, generate the python code to solve it.\n\nCaption = Cookies baked\ntable_data = [['Day', 'Number of cookies'], ['Friday', 163], ['Saturday', 281], ['Sunday', 263]]\nQuestion = Hannah baked cookies each day for a bake sale. How many more cookies did Hannah bake on Saturday than on Sunday?\n\n### RESPONSE:\n",
    #     "tool_maker_output": "def get_column_cell_value(row_index, column):\n    return column[int(row_index)]\n\ndef subtract(minuend, subtrahend):\n    return float(minuend) - float(subtrahend)\n\ndef get_column_by_name(table, column_name):\n    column_index = table[0].index(str(column_name))\n    column = []\n    for row in table:\n        column.append(row[column_index])\n    return column\n\ndef get_row_index_by_value(table, row_value):\n    for i in range(len(table)):\n        if str(row_value) in table[i]:\n            return i\n\ndef solution(table_data):\n    column_name = 'Number of cookies'\n    column_1 = get_column_by_name(table_data, column_name)\n    index_1 = get_row_index_by_value(table_data, 'Saturday')\n    index_2 = get_row_index_by_value(table_data, 'Sunday')\n    cookies_1 = get_column_cell_value(index_1, column_1)\n    cookies_2 = get_column_cell_value(index_2, column_1)\n    answer = subtract(cookies_1, cookies_2)\n    return answer"
    # }
    # output = single_inference.inference_single_sample(sample)
    # print("Model Output:", output)