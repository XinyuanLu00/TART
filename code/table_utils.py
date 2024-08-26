import re

# def linearize_table(table):
#     # construct the table
#     table_str = 'Table:\n'
#     if 'table_column_names' in table:
#         if len(table['table_column_names']) > 0:
#             header = '|| ' + ' | '.join(table['table_column_names']) + ' ||\n'
#             table_str += header
#     for row in table['table_content_values']:
#         table_str += '|| ' + ' | '.join(row) + ' ||\n'
#     table_str = table_str.strip()

#     # generate the caption and claim
#     caption = f"Caption: {table['table_caption']}" if table['table_caption'] is not None else 'Caption: None'
#     question = f"Question: {table['question']}"

#     linearize_table = f'''{caption}\n{table_str}\n\n{question}'''
#     return linearize_table
def linearize_table(table):
    # construct the table
    table_str = 'Table:\n'
    if 'table_column_names' in table:
        header = '|| ' + ' | '.join(table['table_column_names']) + ' ||\n'
        table_str += header
    
    # Check if 'table_content_values' exists before attempting to use it
    if 'table_content_values' in table:
        for row in table['table_content_values']:
            table_str += '|| ' + ' | '.join(row) + ' ||\n'
    else:
        table_str += "No content values available.\n"  # Handle missing content gracefully

    table_str = table_str.strip()

    # generate the caption and claim
    caption = f"Caption: {table.get('table_caption', 'None')}"  # Use .get() to handle missing captions
    question = f"Question: {table.get('question', 'No question provided')}"

    return f'''{caption}\n{table_str}\n\n{question}'''

# Extract the solution code block from the input string
def extract_solution_code_blocks(content):
    solution_start = content.find("def solution(")
    solution_end = content.find("print(solution(table_data))")
    if solution_start != -1 and solution_end != -1:
        solution_block = content[solution_start:solution_end]
        return solution_block
    return None

# Extract the whole line that defines the table_data variable
def extract_table_data_line(content):
    table_data_match = re.search(r'table_data\s*=\s*(\[\[.*\]\])', content, re.DOTALL)
    if table_data_match:
        return table_data_match.group(0)
    return None

# Extract the print statement that calls the solution function
def extract_solution_call(content):
    solution_call_match = re.search(r'print\(solution\(table_data\)\)', content)
    if solution_call_match:
        return solution_call_match.group(0)
    return None

if __name__ == '__main__':
    table = {
        "id": "5",
        "table_caption": "Donor levels",
        "table_column_names": [
            "Donation level",
            "Number of donors"
        ],
        "table_content_values": [
            [
                "Gold",
                "15"
            ],
            [
                "Silver",
                "68"
            ],
            [
                "Bronze",
                "58"
            ]
        ],
        "question": "The Burlington Symphony categorizes its donors as gold, silver, or bronze depending on the amount donated. What fraction of donors are at the bronze level? Simplify your answer.",
        "answer": "58/141",
        "gt_explanations": "Find how many donors are at the bronze level.\n\n58\n\nFind how many donors there are in total.\n\n15 + 68 + 58 = 141\n\nDivide 58 by141.\n\n\\frac{58}{141}\n\n\\frac{58}{141} of donors are at the bronze level.",
        "python_code": "Python Code:\ntable_data = [['Donation level', 'Number of donors'], ['Gold', 15], ['Silver', 68], ['Bronze', 58]]\n\ndef solution(table_data):\n    column_name = 'Number of donors'\n    column_1 = get_column_by_name(table_data, column_name)\n    total = sum_column(column_1)\n    index_1 = get_row_index_by_value(table_data, 'Bronze')\n    bronze = get_column_cell_value(index_1, column_1)\n    answer = divide(bronze, total)\n    return answer\n\nprint(solution(table_data))"
    }

    print(linearize_table(table))
    print(extract_solution_code_blocks(table['python_code']))
    print(extract_table_data_line(table['python_code']))
    print(extract_solution_call(table['python_code']))