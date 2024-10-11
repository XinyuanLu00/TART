import json
import os
import re
import argparse
from utils import OpenAIModel
from tqdm import tqdm
import tiktoken

def linearize_table(table, has_explanation=False):
    # construct the table
    table_str = 'Table:\n'
    if 'table_column_names' in table:
        header = '|| ' + ' | '.join(table['table_column_names']) + ' ||\n'
        table_str += header
    for row in table['table_content_values']:
        table_str += '|| ' + ' | '.join(row) + ' ||\n'
    table_str = table_str.strip()

    # generate the caption and claim
    caption = f"Caption: {table['table_caption']}" if table['table_caption'] is not None else 'Caption: None'
    # FINQA: include the context
    # context = f"Context: {table['context']}" if 'context' in table else 'Context: None'
    question = f"Question: {table['question']}"
    answer = f"Answer: {table['answer']}"
    # Construct linearized table
    linearize_table = f'''{caption}\n{table_str}\n\n{question}'''

    # FINQA: Construct linearized table with context
    # linearized_table = f'''{caption}\n{table_str}\n\n{context}\n\n{question}\n\n{answer}'''
    if has_explanation:
        explanation = f"Explanation: {table['gt_explanations']}"
        linearize_table = f'''{caption}\n{table_str}\n\n{question}\n\n{answer}\n\n{explanation}'''
    else:
        linearize_table = f'''{caption}\n{table_str}\n\n{question}\n\n{answer}'''
    return linearize_table

def clean_output(tool_maker_output):
    clean_text = re.sub(r"^\s*Here are the formatted tables as Python arrays:*?\n", "", tool_maker_output, flags=re.I|re.S)
    clean_text = re.sub(r"^\s*#.*?\n", "", clean_text, flags=re.M)  # Remove comment lines
    clean_text = re.sub(r"^\s*'''", "", clean_text, flags=re.M)  # Remove triple quotes if present
    clean_text = clean_text.strip()
    return clean_text

class GPT4_Tool_Creation:
    def __init__(self, args) -> None:
        self.args = args
        self.openai_api = OpenAIModel(args.API_KEY, args.model_name, args.stop_words, args.max_new_tokens)
        self.save_path = args.save_path
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.dataset = args.dataset
        self.split = args.split
        self.num_eval_samples = args.num_eval_samples
        self.has_explanation = {
            'tabmwp': True, 
            'tabfact': False, 
            'finqa': True, 
            'tatqa': False,
            'scitab': False, 
            'fetaqa': False, 
            'hybridqa': False, 
            'wtq': False, 
            'hitab': False, 
            'pubhealthtab': False
        }
    
    def load_table_dataset(self):
        with open(os.path.join(self.data_path, self.dataset, f'{self.split}.json'), 'r') as f:
            data = json.load(f)
        print(f'{self.dataset}:Loaded {len(data)} samples for generation.')
        return data

    def prompt_construction(self, test_sample):
        # load in-context examples
        with open(f'./prompts/prompt_{self.dataset}.txt', 'r') as f:
            prompt_template = f.read()
        # linearize the table including the context
        test_table = linearize_table(test_sample, self.has_explanation[self.dataset])
        # construct the prompt
        prompt = prompt_template.replace('[[LINEARIZED_TABLE]]', test_table)
        return prompt
    
    def count_price(self, full_prompts, batch_outputs):
        price_table = {'gpt-4': [0.03, 0.06], 'text-davinci-003': [0.02, 0.02], 'gpt-3.5-turbo': [0.0015, 0.002],'gpt-4o': [0.005, 0.015]}
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

    def batch_GPT3_inference(self, batch_size = 10):
        # load raw dataset
        dataset = self.load_table_dataset()

        dataset = dataset if self.num_eval_samples < 0 else dataset[:self.num_eval_samples]
        outputs = []
        total_price = 0
        # split dataset into chunks
        dataset_chunks = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
        #for chunk in tqdm(dataset_chunks[36:210]):
        for chunk in tqdm(dataset_chunks):
            # create prompt
            full_prompts = [self.prompt_construction(sample) for sample in chunk]
            # print(full_prompts[0])
            # batch_outputs = self.openai_api.batch_generate(full_prompts)
            
            try:
                batch_outputs = self.openai_api.batch_generate(full_prompts)
                # create output
                for sample, output in zip(chunk, batch_outputs):
                    clean_output_text = clean_output(output)
                    outputs.append({
                        'id': sample['id'],
                        'question': sample['question'],
                        'answer': sample['answer'],
                        'table_formatter_output': clean_output_text
                    })
                total_price += self.count_price(full_prompts, batch_outputs)
            except:
                # generate one by one if batch generation fails
                for sample, full_prompt in zip(chunk, full_prompts):
                    clean_output_text = clean_output(output)
                    try:
                        output = self.openai_api.generate(full_prompt)
                        outputs.append({
                            'id': sample['id'],
                            'question': sample['question'],
                            'answer': sample['answer'],
                            'table_formatter_output': clean_output_text
                        })
                        total_price += self.count_price([full_prompt], [output])
                    except:
                        print('Error in generating example: ', sample['id'])

        # remove examples with duplicate ids from the result
        outputs = list({output['id']: output for output in outputs}.values())
        print(f"Generated {len(outputs)} examples.")
        print(f"Total price: {total_price}")
        
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        # save the outputs
        with open(os.path.join(self.save_path, f'{self.dataset}_test_output_{self.model_name}_{self.num_eval_samples}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        
        return outputs

def parse_args():
    parser = argparse.ArgumentParser(description="Tool Discovery and Creation Script")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset.")
    parser.add_argument('--API_KEY', type=str, required=True, help="OpenAI API Key.")
    parser.add_argument('--model_name', type=str, default='gpt-4', help="Model name for OpenAI API.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name.")
    parser.add_argument('--split', type=str, default='test', help="Dataset split to use.")
    parser.add_argument('--num_eval_samples', type=int, default=10, help="Number of evaluation samples. Use -1 for all samples.")
    parser.add_argument('--stop_words', nargs='+', default=['------'], help="Stop words for generation.")
    parser.add_argument('--max_new_tokens', type=int, default=2048, help="Maximum new tokens to generate.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the results.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    table_reasoner = GPT4_Tool_Creation(args)
    table_reasoner.batch_GPT3_inference(batch_size=10)