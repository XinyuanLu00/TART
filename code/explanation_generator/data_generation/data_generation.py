import json
import os
import argparse
from utils import OpenAIModel
from tqdm import tqdm
import tiktoken

def linearize_table(table):
    # Construct the table
    table_str = 'Table:\n'
    if 'table_column_names' in table:
        header = '|| ' + ' | '.join(table['table_column_names']) + ' ||\n'
        table_str += header
    for row in table['table_content_values']:
        table_str += '|| ' + ' | '.join(row) + ' ||\n'
    table_str = table_str.strip()

    # Generate the caption and question
    caption = f"Caption: {table['table_caption']}" if table['table_caption'] is not None else 'Caption: None'
    question = f"Question: {table['question']}"
    answer = f"Answer: {table['answer']}"
    # FINQA: include the context
    context = f"Context: {table['context']}" if 'context' in table else 'Context: None'

    #linearized_table = f"{caption}\n{table_str}\n\n{question}\n\n{answer}"
    # FINQA: Construct linearized table with context
    linearized_table = f'''{caption}\n{table_str}\n\n{context}\n\n{question}\n\n{answer}'''
    # if has_explanation:
    #     explanation = f"Explanation: {table['gt_explanations']}"
    #     linearized_table += f"\n\n{explanation}"

    return linearized_table

def load_python_code(sample_id, code_directory):
    file_path = os.path.join(code_directory, f'sample{sample_id}.py')
    #print(f"Attempting to load file: {file_path}")
    if not os.path.exists(file_path):
        # Log or print that the file does not exist
        #print(f"Warning: Python code file not found for sample ID {sample_id}. Skipping.")
        return None  # Return None to indicate the file does not exist
    with open(file_path, 'r') as file:
        return file.read()


class GPT4_Tool_Creation:
    def __init__(self, args)-> None:
        self.args = args
        self.openai_api = OpenAIModel(args.API_KEY, args.model_name, args.stop_words, args.max_new_tokens)
        self.python_code_dir = args.python_code_dir  # Directory containing Python code files
        self.save_path = args.save_path
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.dataset = args.dataset
        self.split = args.split
        self.num_eval_samples = args.num_eval_samples
        # self.has_explanation = {
        #     'tabmwp': True, 
        #     'tabfact': False, 
        #     'finqa': True, 
        #     'tatqa': False,
        #     'scitab': False, 
        #     'fetaqa': False, 
        #     'hybridqa': False, 
        #     'wtq': False, 
        #     'hitab': False, 
        #     'pubhealthtab': False
        # }

    def load_table_dataset(self):
        with open(os.path.join(self.args.data_path, self.args.dataset, f'{self.args.split}.json'), 'r') as f:
            data = json.load(f)
        print(f'{self.args.dataset}: Loaded {len(data)} samples for generation.')
        return data

    def prompt_construction(self, test_sample):
        #print("Sample data:", test_sample)  # Debug output to see what the sample data looks like
        python_code = load_python_code(test_sample['id'], self.python_code_dir)
        if python_code is None:
            return None  # Skip this sample if the Python code is not available
        #test_table = linearize_table(test_sample, self.args.has_explanation)
        with open(f'./prompts/prompt_{self.args.dataset}.txt', 'r') as f:
            prompt_template = f.read()
        #print(f"Loaded prompt template for {self.args.dataset}: {prompt_template[:100]}")  # Show beginning of template
        prompt = prompt_template.replace('[[PYTHON_CODE]]', python_code)
        return prompt


    def count_price(self, full_prompts, batch_outputs):
        price_table = {'gpt-4': [0.03, 0.06], 'text-davinci-003': [0.02, 0.02], 'gpt-3.5-turbo': [0.0015, 0.002]}
        total_price = 0
        for full_prompt, batch_output in zip(full_prompts, batch_outputs):
            encoding = tiktoken.encoding_for_model(self.model_name)
            input_token_count = len(encoding.encode(full_prompt))
            output_token_count = len(encoding.encode(batch_output))
            price = (price_table[self.model_name][0] * input_token_count + price_table[self.model_name][1] * output_token_count) / 1000
            total_price += price
        return total_price

    def count_tokens(self, full_prompts, batch_outputs):
        total_counts = 0
        for full_prompt, batch_output in zip(full_prompts, batch_outputs):
            encoding = tiktoken.encoding_for_model(self.model_name)
            text = full_prompt + batch_output
            token_count = len(encoding.encode(text))
            total_counts += token_count
        return total_counts

    def batch_GPT3_inference(self, batch_size=10):
        dataset = self.load_table_dataset()
        dataset = dataset[:self.args.num_eval_samples] if self.args.num_eval_samples >= 0 else dataset
        outputs = []
        total_price = 0
        dataset_chunks = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            full_prompts = [self.prompt_construction(sample) for sample in chunk]
            valid_prompts = [prompt for prompt in full_prompts if prompt is not None]  # Filter out None values
            if not valid_prompts:  
                print("No valid prompts available for this chunk. Skipping.")
                continue
            #print(f"Processing {len(valid_prompts)} valid prompts.")
            try:
                batch_outputs = self.openai_api.batch_generate(full_prompts)
                for sample, output in zip([s for s in chunk if self.prompt_construction(s) is not None], batch_outputs):
                    outputs.append({
                        'id': sample['id'],
                        'question': sample['question'],
                        'gold': sample['answer'],
                        'prediction': output
                    })
                    # Assume count_price is defined to handle a list of prompts and their outputs
                    total_price += self.count_price(full_prompts, batch_outputs)
            except Exception as e:
                print(f'Error during batch generation: {e}')
        
        print(f"Generated {len(outputs)} examples.")
        #print(f"Total price: {total_price}")
        if not os.path.exists(self.args.save_path):
            os.mkdir(self.args.save_path)
        with open(os.path.join(self.args.save_path, f'{self.args.dataset}_{self.model_name}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        
        return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="Tool Discovery and Creation Script")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset.")
    parser.add_argument('--API_KEY', type=str, required=True, help="OpenAI API Key.")
    parser.add_argument('--model_name', type=str, default='gpt-4', help="Model name for OpenAI API.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name.")
    parser.add_argument('--split', type=str, default='train', help="Dataset split to use.")
    parser.add_argument('--num_eval_samples', type=int, default=-1, help="Number of evaluation samples. Use -1 for all samples.")
    parser.add_argument('--python_code_dir', type=str, required=True, help="Directory containing Python code files.")
    parser.add_argument('--stop_words', nargs='+', default=['------'], help="Stop words for generation.")
    parser.add_argument('--max_new_tokens', type=int, default=2048, help="Maximum new tokens to generate.")
   # parser.add_argument('--has_explanation', type=bool, default=False, help="Include explanations in the prompt.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the results.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    tool_creator = GPT4_Tool_Creation(args)
    tool_creator.batch_GPT3_inference(batch_size=10)
