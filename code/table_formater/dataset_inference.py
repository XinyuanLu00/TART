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

class TableFormaterInference:
    def __init__(self, args) -> None:
        self.args = args
        self.data_path = args.data_path
        self.dataset = self.load_data()
        self.lora_inference = LoRA_Inference(self.args)

    # Format the data sample
    def format_data_sample(self, sample):
        # get linearized table
        input_table = linearize_table(sample)
        # get each part of the prompt
        task_str = f"Given the following table and question, format the table into a python array.\n\n{input_table}"

        # format the prompt
        # start = "Read the Instruction below and provide an answer.\n"
        task = f"### INSTRUCTION:\n{task_str.strip()}\n\n"
        # response = f"### RESPONSE:\ntable_data = "
        response = f"### RESPONSE:\n"

        parts = [part for part in [task, response] if part]
        # parts = [part for part in [start, task, response, end] if part]
        formatted_prompt = "".join(parts)
        formatted_prompt = formatted_prompt.replace('\\n', '\n')
        return formatted_prompt

    def load_data(self):
        with open(self.data_path, "r") as f:
            data = json.load(f)
        dataset = []
        for sample in data:
            sample['table_formatter_input'] = self.format_data_sample(sample)
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
                input_str = [sample['table_formatter_input'] for sample in batch]
                output_str = self.lora_inference.generate_batch(input_str)
                for j, sample in enumerate(batch):
                    sample['table_formatter_output'] = self.post_process_output(output_str[j])
                    output_data.append(sample)
        else:
            # # single inference
            for sample in tqdm(self.dataset):
                input_str = sample['table_formatter_input']
                output_str = self.lora_inference.generate([input_str])
                sample['table_formatter_output'] = self.post_process_output(output_str[0])
                output_data.append(sample)
                #print('table_formatter_input:', sample['table_formatter_input'],'table_formatter_output:', sample['table_formatter_output'])

        # save the output
        with open(self.args.output_path, "w") as f:
            json.dump(output_data, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    data_inference = TableFormaterInference(args)
    data_inference.inference_on_dataset()

    # single_inference = TableFormaterInference(args)
    # sample = {
    #     "id": "2",
    #     "table_caption": "Kilometers walked",
    #     "table_column_names": [],
    #     "table_content_values": [
    #         [
    #             "Day",
    #             "Kilometers"
    #         ],
    #         [
    #             "Tuesday",
    #             "0"
    #         ],
    #         [
    #             "Wednesday",
    #             "0"
    #         ],
    #         [
    #             "Thursday",
    #             "4"
    #         ],
    #         [
    #             "Friday",
    #             "0"
    #         ],
    #         [
    #             "Saturday",
    #             "0"
    #         ]
    #     ],
    #     "question": "Allie kept track of how many kilometers she walked during the past 5 days. What is the range of the numbers?",
    #     "answer": "4",
    #     "gt_explanations": "Read the numbers from the table.\n\n0, 0, 4, 0, 0\n\nFirst, find the greatest number. The greatest number is 4.\n\nNext, find the least number. The least number is 0.\n\nSubtract the least number from the greatest number:\n\n4 \u2212 0 = 4\n\nThe range is 4."
    # }
    # output = single_inference.inference_single_sample(sample)
    # print("Model Output:", output)