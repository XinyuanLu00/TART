import json
import os
import argparse
from utils import OpenAIModel
from tqdm import tqdm
import tiktoken

def linearize_table(table):
    # construct the table
    table_str = 'Table:\n'
    if 'table_column_names' in table:
        header = '|| ' + ' | '.join(table['table_column_names']) + ' ||\n'
        table_str += header
    for row in table['table_content_values']:
        table_str += '|| ' + ' | '.join(row) + ' ||\n'
    table_str = table_str.strip()

    caption = f"Caption: {table['table_caption']}" if table['table_caption'] is not None else 'Caption: None'
    question = f"Question: {table['question']}"
    answer = f"Answer: {table['answer']}"
    linearized_table = f'''{caption}\n{table_str}\n\n{question}\n\n{answer}'''
    
    # if has_explanation:
    #     explanation = f"Explanation: {table['gt_explanations']}"
    #     linearized_table += f"\n\n{explanation}"

    return linearized_table

class GPT4_Tool_Creation:
    def __init__(self, args):
        self.args = args
        self.openai_api = OpenAIModel(args.API_KEY, args.model_name, args.stop_words, args.max_new_tokens)
        self.save_path = args.save_path
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.dataset = args.dataset
        self.split = args.split
        self.num_eval_samples = args.num_eval_samples
        self.valid_ids = self.load_valid_ids()

    def load_valid_ids(self):
        valid_ids = set()
        with open(os.path.join('../../', 'tool_maker', 'data', f'{self.dataset}_train.jsonl'), 'r') as f:
            for line in f:
                data = json.loads(line)
                valid_ids.add(data['id'])
        return valid_ids

    def load_table_dataset(self):
        with open(os.path.join(self.data_path, self.dataset, f'{self.split}.json'), 'r') as f:
            data = json.load(f)
        data = [sample for sample in data if sample['id'] in self.valid_ids]
        print(f'{self.dataset}: Loaded {len(data)} samples for generation.')
        return data

    def prompt_construction(self, test_sample):
        test_table = linearize_table(test_sample)
        # Load prompt template from a file and replace placeholder with actual table string
        with open(f'./prompts/prompt_{self.dataset}.txt', 'r') as file:
            prompt_template = file.read()
        prompt_template = prompt_template.replace('[[LINEARIZED_TABLE]]', test_table)
        return prompt_template

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

    def batch_GPT3_inference(self, batch_size = 10):
        dataset = self.load_table_dataset()
        outputs = []
        total_price = 0
        dataset_chunks = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            full_prompts = [self.prompt_construction(sample) for sample in chunk]
            try:
                batch_outputs = self.openai_api.batch_generate(full_prompts)
                for sample, output in zip(chunk, batch_outputs):
                    outputs.append({
                        'id': sample['id'],
                        'question': sample['question'],
                        'gold': sample['answer'],
                        'explanation': output
                    })
                total_price += self.count_price(full_prompts, batch_outputs)
            except Exception as e:
                print('Error in generating example: ', e)

        print(f"Generated {len(outputs)} examples.")
        print(f"Total price: {total_price}")
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        with open(os.path.join(self.save_path, f'{self.dataset}_{self.model_name}.json'), 'w') as f:
            json.dump(outputs, f, indent=2, ensure_ascii=False)
        return outputs

def parse_args():
    parser = argparse.ArgumentParser(description="Tool Discovery and Creation Script")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset.")
    #parser.add_argument('--has_explanation', type=bool, default=False, help="Whether the dataset has explanations.")
    parser.add_argument('--API_KEY', type=str, required=True, help="OpenAI API Key.")
    parser.add_argument('--model_name', type=str, default='gpt-4', help="Model name for OpenAI API.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name.")
    parser.add_argument('--split', type=str, default='train', help="Dataset split to use.")
    parser.add_argument('--num_eval_samples', type=int, default=10, help="Number of evaluation samples. Use -1 for all samples.")
    parser.add_argument('--stop_words', nargs='+', default=['------'], help="Stop words for generation.")
    parser.add_argument('--max_new_tokens', type=int, default=2048, help="Maximum new tokens to generate.")
    parser.add_argument('--save_path', type=str, required=True, help="Path to save the results.")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    table_reasoner = GPT4_Tool_Creation(args)
    table_reasoner.batch_GPT3_inference(batch_size=10)
