table_data = [['Day', 'Number of cookies'], ['Friday', 163], ['Saturday', 281], ['Sunday', 263]]

def get_column_by_name(table, column_name):
    column_index = table[0].index(column_name)
    column = []
    for row in table:
        column.append(row[column_index])
    return column
                                
def find_min_index(column, with_header=True):
    column = column[1:] if with_header else column
    min_value = min(column)
    return column.index(min_value) + 1 if with_header else column.index(min_value)
                                                
def get_column_cell_value(row_index, column):
    return column[row_index]
                                                        
def solution(table_data):
    column_name = 'Number of cookies'
    column_1 = get_column_by_name(table_data, column_name)
    min_index = find_min_index(column_1)
    column_2 = get_column_by_name(table_data, 'Day')
    answer = get_column_cell_value(min_index, column_2)
    return answer
                                                                                    
table_data = [['Day', 'Number of cookies'], ['Friday', 163], ['Saturday', 281], ['Sunday', 263]]
print(solution(table_data))
