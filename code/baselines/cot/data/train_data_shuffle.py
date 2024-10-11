import json
import random

def merge_and_shuffle_jsonl(file_paths, output_file):
    all_entries = []

    # Read and collect all entries from each file
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                all_entries.append(json.loads(line))
    
    # Shuffle the collected entries
    random.shuffle(all_entries)

    # Write the shuffled entries to a new file
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for entry in all_entries:
            out_file.write(json.dumps(entry) + '\n')

# Define the paths to your files
file_paths = [
    './tabmwp_train.jsonl',
    './tabfact_train.jsonl',
    './pubhealthtab_train.jsonl',
    './scitab_train.jsonl',
    './finqa_train.jsonl'
]

# Define the output file path
output_file = './train_all.jsonl'

# Call the function
merge_and_shuffle_jsonl(file_paths, output_file)
