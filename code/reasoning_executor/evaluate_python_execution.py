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

    def format_python_program(self, table_data, func_calls):
        program = ""
        program += f"{table_data}\n\n"
        program += f"{func_calls}\n\n"
        program += "answer = solution(table_data)"
        return program

    def execute_sample(self, sample):
        func_calls, table_data = sample['tool_maker_output'], sample['table_formatter_output']
        full_program = self.format_python_program(table_data, func_calls)
        # execute the python program
        answer = self.safe_execute_code(full_program)
        return answer

    def safe_execute_code(self, code):
        try:
            exec(code, globals())
            return globals()['answer']
        except Exception as e:
            return None

    def execute_programs(self):
        exec_count = 0
        correct_count = 0
        correct_samples = []
        for sample in self.result_data:
            answer = self.execute_sample(sample)
            print(f"ID: {sample['id']}, G.T: {sample['answer']}, Pred: {answer}")
            if answer is not None:
                exec_count += 1
                if compare_results(str(answer), sample['answer']) == True:
                    correct_count += 1
                    correct_samples.append(sample)
        
        print(f"Execution Rate: {(exec_count/len(self.result_data))*100}%")
        print(f"Accuracy: {(correct_count/exec_count)*100}%")
        print(f"Number of correct samples: {correct_count}")

def parse_args():
    parser = argparse.ArgumentParser(description='Program Interpreter')
    parser.add_argument('--verbose', action='store_true', help='Print the execution process')
    parser.add_argument('--result_path', type=str, help='Path to save the results')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    args.result_path = '../tool_maker/output/finqa_test_output_lm2_cl.json'
    dataset_execution = Dataset_Execution(args)
    dataset_execution.execute_programs()