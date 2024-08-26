import re
import argparse
import json
from metrics import compare_results

class SnippetGenerator:
    def __init__(self, gt_program):
        self.gt_program = gt_program
        self.snippets = self.extract_snippets(gt_program)

    def extract_snippets(self, text):
        # Regular expression to find all occurrences of text within <<< >>>
        pattern = re.compile(r'<<<(.*?)>>>')
        # Find all matches
        code_snippets = pattern.findall(text)
        return(code_snippets)

class ProgramInterpreter:
    def __init__(self, sample, verbose=False):
        self.context = {}  # Tracks the state of the program
        self.initialize_program(sample)  # Tracks the current program being executed
        self.verbose = verbose
        self.sample = sample

    def initialize_program(self, sample):
        self.current_program = []
        # extract table_data
        table_data = sample['table_formatter_output']
        # extract function definitions
        function_definitions = self.extract_functions(sample['tool_maker_output'].strip())

        self.current_program.append(table_data+'\n')
        for func in function_definitions:
            self.current_program.append(func+'\n')
        # print('\n'.join(self.current_program))

    def extract_functions(self, code_string):
        functions = []
        lines = code_string.split('\n')
        function_str = ""
        in_function = False

        for line in lines:
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
        functions = [func for func in functions if not func.startswith('def solution')]

        return functions
    
    def execute_program(self):
        # execute the program
        program = self.sample['explanation_output']
        snippet_generator = SnippetGenerator(program)
        for idx, snippet in enumerate(snippet_generator.snippets):
            self.execute_snippet(snippet)

        answer = self.context['answer']
        return answer

    def update_current_program(self, code):
        codes = [line.strip() for line in code.split(';;;')]
        if self.verbose:
            print(f">>> Executable Codes:\n {codes}\n")
        self.current_program.extend(codes)
        return True

    def execute_snippet(self, snippet):
        # Extracting and executing the Python code inside <<< >>>
        # code = re.findall(r"<<<(.*?)>>>", snippet, re.DOTALL)[0]
        code = snippet
        need_execution = self.update_current_program(code)

        # Execute the code in a safe environment
        if need_execution:
            program = "\n".join(self.current_program)
            if self.verbose:
                print(f"Executing code: {program}")
            local_vars = {}
            exec(program, globals(), local_vars)
            self.context.update(local_vars)
            # print(f">>> Variables:\n {self.context}\n")

class Dataset_Execution:
    def __init__(self, args):
        self.args = args
        self.result_data = self.load_result_data()

    def load_result_data(self):
        with open(self.args.result_path, "r") as f:
            result_data = json.load(f)
        return result_data

    def execute_sample(self, sample):
        try:
            interpreter = ProgramInterpreter(sample, verbose=self.args.verbose)
            answer = interpreter.execute_program()
        except Exception as e:
            print(e)
            answer = None
        return answer

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
    parser.add_argument('--result_path', type=str, required=True, help='Path to save the results')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dataset_execution = Dataset_Execution(args)
    dataset_execution.execute_programs()

    # data_sample = {
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
    #     "tool_maker_output": "def get_column_cell_value(row_index, column):\n    return column[int(row_index)]\n\ndef subtract(minuend, subtrahend):\n    return float(minuend) - float(subtrahend)\n\ndef get_column_by_name(table, column_name):\n    column_index = table[0].index(str(column_name))\n    column = []\n    for row in table:\n        column.append(row[column_index])\n    return column\n\ndef get_row_index_by_value(table, row_value):\n    for i in range(len(table)):\n        if str(row_value) in table[i]:\n            return i\n\ndef solution(table_data):\n    column_name = 'Number of cookies'\n    column_1 = get_column_by_name(table_data, column_name)\n    index_1 = get_row_index_by_value(table_data, 'Saturday')\n    index_2 = get_row_index_by_value(table_data, 'Sunday')\n    cookies_1 = get_column_cell_value(index_1, column_1)\n    cookies_2 = get_column_cell_value(index_2, column_1)\n    answer = subtract(cookies_1, cookies_2)\n    return answer",
    #     "explanation_input": "### INSTRUCTION:\nGiven the following table, question, and python code solution, generate the explanations in natural language embedded with function calls.\n\nCaption: Cookies baked\nTable:\n|| Friday | 163 ||\n|| Saturday | 281 ||\n|| Sunday | 263 ||\n\nQuestion: Hannah baked cookies each day for a bake sale. How many more cookies did Hannah bake on Saturday than on Sunday?\n\ndef solution(table_data):\n    column_name = 'Number of cookies'\n    column_1 = get_column_by_name(table_data, column_name)\n    index_1 = get_row_index_by_value(table_data, 'Saturday')\n    index_2 = get_row_index_by_value(table_data, 'Sunday')\n    cookies_1 = get_column_cell_value(index_1, column_1)\n    cookies_2 = get_column_cell_value(index_2, column_1)\n    answer = subtract(cookies_1, cookies_2)\n    return answer\n\n### RESPONSE:\n",
    #     "explanation_output": "First, we should get the column that has the number of cookies <<<column_name = 'Number of cookies' ;;; column_1 = get_column_by_name(table_data, column_name)>>>.\nThen, we find the row index for 'Saturday' and 'Sunday' <<<index_1 = get_row_index_by_value(table_data, 'Saturday') ;;; index_2 = get_row_index_by_value(table_data, 'Sunday')>>>.\nNext, we get the number of cookies baked on 'Saturday' and 'Sunday' <<<cookies_1 = get_column_cell_value(index_1, column_1) ;;; cookies_2 = get_column_cell_value(index_2, column_1)>>>.\nFinally, we calculate the difference in the number of cookies baked on 'Saturday' and 'Sunday' <<<answer = subtract(cookies_1, cookies_2)>>>.\nThis gives us the answer to how many more cookies Hannah baked on Saturday than on Sunday."
    # }
    # interpreter = ProgramInterpreter(data_sample, verbose=True)
    # answer = interpreter.execute_program()
    # print(f"Answer: {answer}")