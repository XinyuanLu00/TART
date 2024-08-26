import json
import argparse
from tqdm import tqdm
import os
import sys
from fractions import Fraction

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from reasoning_executor.metrics import compare_results

def custom_serializer(obj):
    if isinstance(obj, Fraction):
        return float(obj)
    raise TypeError("Object of type '{}' is not JSON serializable".format(type(obj).__name__))

##set to list
# def custom_serializer(obj):
#     if isinstance(obj, set):
#         return list(obj)
#     raise TypeError("Object of type '{}' is not JSON serializable".format(type(obj).__name__))


def load_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_data(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4, default=custom_serializer)

def post_process_output(output):
    try:
        result = output.split("Therefore, the answer is")[1].split("\n")[0].strip()
        return result if result else None
    except IndexError:
        #print("Expected substring not found in output")
        return None

execution_environment = {}
def execute_python_code(tool_maker_output, table_formatter_output):
    full_program = f"{table_formatter_output}\n{tool_maker_output}\nanswer = solution(table_data)"  
    try:
        exec(full_program, execution_environment, execution_environment)
        return execution_environment.get('answer'), None
    except Exception as e:
        return None, print(str(e))

def load_cot_results(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def parse_args():
    parser = argparse.ArgumentParser(description='Combine TART and COT Inference Results')
    parser.add_argument('--tart_results_path', type=str, required=True, help="Path to TART results JSON file")
    parser.add_argument('--cot_results_path', type=str, required=True, help="Path to COT results JSON file")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the combined results")
    return parser.parse_args()

def main():
    args = parse_args()
    tart_results = load_data(args.tart_results_path)
    cot_results = load_cot_results(args.cot_results_path)
    # Create a dictionary to store both original and post-processed outputs
    cot_dict = {res['id']: {'cot_output': res['output'], 'cot_answer': post_process_output(res['output'])} for res in cot_results}

    non_executable_samples = []
    error_messages = {}
    correct_tart = 0
    correct_cot = 0
    wrong_tart = 0
    wrong_cot = 0

    # Execute TART results and substitute with CoT where execution fails
    for sample in tart_results:
        #tart program execution
        execution_result, error = execute_python_code(sample['tool_maker_output'], sample['table_formatter_output'])

        if execution_result is None:
            non_executable_samples.append(sample)
            error_messages[sample['id']] = error
            if sample['id'] in cot_dict:
                # Store both original and processed CoT outputs
                sample.update({
                    'cot_output': cot_dict[sample['id']]['cot_output'],
                    'cot_answer': cot_dict[sample['id']]['cot_answer'],
                    'source': 'CoT'
                })
                correct = compare_results(str(sample['cot_answer']), sample['answer'])
                sample['correct'] = correct
                if correct:
                    correct_cot += 1
                else:
                    wrong_cot += 1
        else:
            sample['pred'] = execution_result
            sample['source'] = 'TART'
            correct = compare_results(str(execution_result), sample['answer'])
            sample['correct'] = correct
            if correct:
                correct_tart += 1
            else:
                wrong_tart += 1

    # Save final combined results
    save_data(tart_results, args.output_path)

    total_samples = len(tart_results)
    final_accuracy = ((correct_tart + correct_cot) / total_samples) * 100 if total_samples else 0

    result_text = f"{args.output_path}_results.txt"
    with open(result_text, 'w') as f:
        f.write(f"Total TART samples: {len(tart_results)}, executable samples: {len(tart_results) - len(non_executable_samples)}, Non-executable samples: {len(non_executable_samples)}, correct TART samples: {correct_tart}, wrong TART samples: {wrong_tart}\n")
        f.write(f"Errors in non-executable samples: {error_messages}\n")
        f.write(f"Total CoT samples: {len(non_executable_samples)}, correct CoT samples: {correct_cot}, wrong CoT samples: {wrong_cot}\n")
        f.write(f"Total samples: {total_samples}, Correct samples: {correct_tart + correct_cot}, Final Accuracy: {final_accuracy:.2f}%\n")
        for sample in tart_results:
            if 'cot_answer' in sample:
                pred = sample['cot_answer']
                source = 'CoT'
            else:
                pred = sample.get('pred', None)
                source = 'TART'
            f.write(f"ID: {sample['id']}, G.T: {sample['answer']}, Pred: {pred}, Source: {source}, Correct: {sample['correct']}\n")
        
    print(f"Results saved to {args.output_path}")
    print(f"Total samples: {total_samples}, Correct samples: {correct_tart + correct_cot}, Final Accuracy: {final_accuracy:.2f}%")

if __name__ == "__main__":
    main()
