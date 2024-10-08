'''
Caption: Train tickets sold
Table:
|| Day | Number of tickets ||
|| Friday | 71 ||
|| Saturday | 74 ||
|| Sunday | 75 ||
|| Monday | 72 ||

Question: The transportation company tracked the number of train tickets sold in the past 4 days. On which day were the fewest train tickets sold?
'''

table_data = [['Day', 'Number of tickets'], ['Friday', 71], ['Saturday', 74], ['Sunday', 75], ['Monday', 72]]

# get the column by name
def get_column_by_name(table, column_name):
    column_index = table[0].index(column_name)
    column = []
    for row in table:
        column.append(row[column_index])
    return column

# find the index of the minimum value in a column
def find_min_index(column, with_header=True):
    column = column[1:] if with_header else column
    min_value = min(column)
    return column.index(min_value) + 1 if with_header else column.index(min_value)

# get the value of a cell for a given column
def get_column_cell_value(row_index, column):
    return column[row_index]

def solution(table_data):
    column_name = 'Number of tickets'
    column_1 = get_column_by_name(table_data, column_name)
    min_index = find_min_index(column_1)
    column_2 = get_column_by_name(table_data, 'Day')
    answer = get_column_cell_value(min_index, column_2)
    return answer

print(solution(table_data))

'''
Caption: Donor levels
Table:
|| Donation level | Number of donors ||
|| Gold | 15 ||
|| Silver | 68 ||
|| Bronze | 58 ||

Question: The Burlington Symphony categorizes its donors as gold, silver, or bronze depending on the amount donated. What fraction of donors are at the bronze level? Simplify your answer.
'''

table_data = [['Donation level', 'Number of donors'], ['Gold', 15], ['Silver', 68], ['Bronze', 58]]

# get the column by name
def get_column_by_name(table, column_name):
    column_index = table[0].index(column_name)
    column = []
    for row in table:
        column.append(row[column_index])
    return column

# get the value of a cell for a given column
def get_column_cell_value(row_index, column):
    return column[row_index]

# get row index by value
def get_row_index_by_value(table, row_value):
    for i in range(len(table)):
        if table[i][0] == row_value:
            return i

# sum up all the values in a column
def sum_column(column, with_header=True):
    column = column[1:] if with_header else column
    return sum(column)

# divide two numbers
def divide(numerator, denominator):
    return numerator / denominator

def solution(table_data):
    column_name = 'Number of donors'
    column_1 = get_column_by_name(table_data, column_name)
    total = sum_column(column_1)
    index_1 = get_row_index_by_value(table_data, 'Bronze')
    bronze = get_column_cell_value(index_1, column_1)
    answer = divide(bronze, total)
    return answer

print(solution(table_data))

'''
Caption: Games won by the Springtown baseball team
Table:
|| Year | Games won ||
|| 2013 | 8 ||
|| 2014 | 8 ||
|| 2015 | 17 ||
|| 2016 | 7 ||
|| 2017 | 16 ||

Question: Fans of the Springtown baseball team compared the number of games won by their team each year. According to the table, what was the rate of change between 2014 and 2015?
'''

table_data = [['Year', 'Games won'], [2013, 8], [2014, 8], [2015, 17], [2016, 7], [2017, 16]]

# get the column by name
def get_column_by_name(table, column_name):
    column_index = table[0].index(column_name)
    column = []
    for row in table:
        column.append(row[column_index])
    return column

# get the value of a cell for a given column
def get_column_cell_value(row_index, column):
    return column[row_index]

# get row index by value
def get_row_index_by_value(table, row_value):
    for i in range(len(table)):
        if table[i][0] == row_value:
            return i

# subtract two numbers
def subtract(minuend, subtrahend):
    return minuend - subtrahend

# divide two numbers
def divide(numerator, denominator):
    return numerator / denominator

def solution(table_data):
    column_name = 'Games won'
    column_1 = get_column_by_name(table_data, column_name)
    index_1 = get_row_index_by_value(table_data, 2014)
    index_2 = get_row_index_by_value(table_data, 2015)
    games_1 = get_column_cell_value(index_1, column_1)
    games_2 = get_column_cell_value(index_2, column_1)
    answer = divide(subtract(games_2, games_1), subtract(2015, 2014))
    return answer

print(solution(table_data))

'''
Caption: null
Table:
||  ||
|| hot sauce | $3/lb ||
|| soy sauce | $3/lb ||
|| mayonnaise | $2/lb ||
|| ketchup | $3/lb ||
|| mustard | $6/lb ||
|| Dijon mustard | $5/lb ||

Question: Brittany went to the store and bought 1.4 pounds of mustard. How much did she spend?
'''

table_data = [['', ''], ['hot sauce', '$3/lb'], ['soy sauce', '$3/lb'], ['mayonnaise', '$2/lb'], ['ketchup', '$3/lb'], ['mustard', '$6/lb'], ['Dijon mustard', '$5/lb']]

# get the column by index
def get_column_by_index(table, column_index):
    column = []
    for row in table:
        if len(row) > column_index:
            column.append(row[column_index])
        else:
            column.append(None)
    return column

# get the value of a cell for a given column
def get_column_cell_value(row_index, column):
    return column[row_index]

# get row index by value
def get_row_index_by_value(table, row_value):
    for i in range(len(table)):
        if table[i][0] == row_value:
            return i

# extract the price from a string
def extract_price(price_string):
    return float(price_string.replace('$', '').replace('/lb', ''))

# multiply two numbers
def multiply(num1, num2):
    return num1 * num2

def solution(table_data):
    column_index = 1
    column_1 = get_column_by_index(table_data, column_index)
    index_1 = get_row_index_by_value(table_data, 'mustard')
    price_string = get_column_cell_value(index_1, column_1)
    price = extract_price(price_string)
    answer = multiply(price, 1.4)
    return answer

print(solution(table_data))