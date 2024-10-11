import argparse
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from reasoning_executor.metrics import compare_results

class Dataset_Execution:
    def __init__(self, args):
        self.args = args
        self.result_data = self.load_result_data()

    def load_result_data(self):
        with open(self.args.result_path, "r") as f:
            result_data = json.load(f)
        return result_data

    def execute_programs(self):
        correct_count = 0
        incorrect_samples = 0
        correct_samples = []
        for sample in self.result_data:
            try:
                output = sample['output'].split("Therefore, the answer is")[1].strip()
                answer = sample['answer'].strip()
                print(f"ID: {sample['id']}, G.T: {answer}, Pred: {output}")
                if compare_results(output, answer):
                    correct_count += 1
                    correct_samples.append(sample)
                else:
                    incorrect_samples += 1
            except IndexError:
                print(f"ID: {sample['id']}, Output Error")
                incorrect_samples += 1

        total_samples = len(self.result_data)
        accuracy = (correct_count / total_samples) * 100 if total_samples > 0 else 0
        print(f"CoT File: {args.result_path}")
        print(f"Accuracy: {accuracy}%")
        print(f"Number of correct samples: {correct_count}")
        # print("Correct samples:")
        # for sample in correct_samples:
        #     print(sample['id'])

def parse_args():
    parser = argparse.ArgumentParser(description='Result Evaluation')
    parser.add_argument('--verbose', action='store_true', help='Print the execution process')
    parser.add_argument('--result_path', type=str, help='Path to the result JSON file')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.result_path = './output/tabfact_test_output_cl188.json'
    dataset_execution = Dataset_Execution(args)
    dataset_execution.execute_programs()
