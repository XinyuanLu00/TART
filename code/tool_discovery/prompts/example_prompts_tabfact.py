'''
Caption: cnbc prime 's the profit 200
Table:
|| year | date | driver | team | manufacturer | laps | - | race time | average speed (mph) ||
|| 1990 | july 15 | tommy ellis | john jackson | buick | 300 | 317.4 (510.805) | 3:41:58 | 85.797 ||
|| 1990 | october 14 | rick mast | ag dillard motorsports | buick | 250 | 264.5 (425.671) | 2:44:37 | 94.405 ||
|| 1991 | july 14 | kenny wallace | rusty wallace racing | pontiac | 300 | 317.4 (510.805) | 2:54:38 | 109.093 ||
|| 1991 | october 13 | ricky craven | ricky craven | chevrolet | 250 | 264.5 (425.671) | 2:54:43 | 90.832 ||
|| 1992 | july 12 | jeff burton | filmar racing | oldsmobile | 300 | 317.4 (510.805) | 3:18:34 | 95.907 ||
|| 1992 | august 23 | joe nemechek | nemco motorsports | chevrolet | 250 | 264.5 (425.671) | 2:47:14 | 94.897 ||
|| 1993 | august 22 | robert pressley | daniel welch | chevrolet | 250 | 264.5 (425.671) | 2:57:12 | 89.56 ||
|| 1994 | may 7 | derrike cope | ron zock | ford | 250 | 264.5 (425.671) | 2:59:16 | 88.527 ||
|| 1995 | may 13 | chad little | mark rypien motorsports | ford | 250 | 264.5 (425.671) | 2:31:11 | 104.972 ||
|| 1996 | july 12 | randy lajoie | bace motorsports | chevrolet | 200 | 211.6 (340.537) | 2:10:57 | 96.953 ||
|| 1997 | may 10 | mike mclaughlin | team 34 | chevrolet | 200 | 211.6 (340.537) | 2:45:25 | 76.752 ||
|| 1998 | may 9 | buckshot jones | buckshot racing | pontiac | 200 | 211.6 (340.537) | 2:05:55 | 100.829 ||
|| 1999 | may 8 | elton sawyer | akins - sutton motorsports | ford | 200 | 211.6 (340.537) | 2:03:42 | 103.324 ||
|| 2000 | may 13 | tim fedewa | cicci - welliver racing | chevrolet | 200 | 211.6 (340.537) | 2:22:04 | 89.366 ||
|| 2001 | may 12 | jason keller | ppc racing | ford | 200 | 211.6 (340.537) | 1:56:47 | 108.714 ||
|| 2002 | may 11 | bobby hamilton , jr | team rensi motorsports | ford | 200 | 211.6 (340.537) | 1:55:02 | 110.368 ||
|| 2003 | july 19 | david green | brewco motorsports | pontiac | 200 | 211.6 (340.537) | 1:57:33 | 108.005 ||

Question: Is it true that the race time on may 11 exceeded two hours?
'''

table_data = [["year", "date", "driver", "team", "manufacturer", "laps", "-", "race time", "average speed (mph)"],["1990", "july 15", "tommy ellis", "john jackson", "buick", "300", "317.4 (510.805)", "3:41:58", "85.797"],["1990", "october 14", "rick mast", "ag dillard motorsports", "buick", "250", "264.5 (425.671)", "2:44:37", "94.405"],["1991", "july 14", "kenny wallace", "rusty wallace racing", "pontiac", "300", "317.4 (510.805)", "2:54:38", "109.093"],["1991", "october 13", "ricky craven", "ricky craven", "chevrolet", "250", "264.5 (425.671)", "2:54:43", "90.832"],["1992", "july 12", "jeff burton", "filmar racing", "oldsmobile", "300", "317.4 (510.805)", "3:18:34", "95.907"],["1992", "august 23", "joe nemechek", "nemco motorsports", "chevrolet", "250", "264.5 (425.671)", "2:47:14", "94.897"],["1993", "august 22", "robert pressley", "daniel welch", "chevrolet", "250", "264.5 (425.671)", "2:57:12", "89.56"],["1994", "may 7", "derrike cope", "ron zock", "ford", "250", "264.5 (425.671)", "2:59:16", "88.527"],["1995", "may 13", "chad little", "mark rypien motorsports", "ford", "250", "264.5 (425.671)", "2:31:11", "104.972"],["1996", "july 12", "randy lajoie", "bace motorsports", "chevrolet", "200", "211.6 (340.537)", "2:10:57", "96.953"],["1997", "may 10", "mike mclaughlin", "team 34", "chevrolet", "200", "211.6 (340.537)", "2:45:25", "76.752"],["1998", "may 9", "buckshot jones", "buckshot racing", "pontiac", "200", "211.6 (340.537)", "2:05:55", "100.829"],["1999", "may 8", "elton sawyer", "akins - sutton motorsports", "ford", "200", "211.6 (340.537)", "2:03:42", "103.324"],["2000", "may 13", "tim fedewa", "cicci - welliver racing", "chevrolet", "200", "211.6 (340.537)", "2:22:04", "89.366"],["2001", "may 12", "jason keller", "ppc racing", "ford", "200", "211.6 (340.537)", "1:56:47", "108.714"],["2002", "may 11", "bobby hamilton , jr", "team rensi motorsports", "ford", "200", "211.6 (340.537)", "1:55:02", "110.368"],["2003", "july 19", "david green", "brewco motorsports", "pontiac", "200", "211.6 (340.537)", "1:57:33", "108.005"]]

# get the column by name
def get_column_by_name(table, column_name):
    column_index = table[0].index(column_name)
    column = []
    for row in table[1:]:
        column.append(row[column_index])
    return column

# parse the time from seconds to hours
def parse_time_to_hours(time_str):
    hours, minutes, seconds = map(int, time_str.split(':'))
    return hours + minutes / 60 + seconds / 3600

# compare two numbers
def greater(num1, num2):
    return num1 > num2

def solution(table_data):
    race_times = get_column_by_name(table_data, "race time")
    dates = get_column_by_name(table_data, "date")
    answer = False
    for date, race_time in zip(dates, race_times):
        if "may 11" in date.lower():
            parsed_time = parse_time_to_hours(race_time)
            if greater(parsed_time, 2):
                answer = True
                break
    return answer

print(solution(table_data))

'''
Caption: ghost whisperer (season 3)
Table:
|| no in series | no in season | title | directed by | written by | original air date | us viewers (millions) ||
|| 45 | 1 | the underneath | john gray | john gray | september 28 , 2007 | 8.72 ||
|| 46 | 2 | don't try this at home | ian sander | teddy tenenbaum & laurie mccarthy | october 5 , 2007 | 8.91 ||
|| 47 | 3 | haunted hero | eric laneuville | breen frazier & karl schaefer | october 12 , 2007 | 8.90 ||
|| 48 | 4 | no safe place | peter o'fallon | jeannine renshaw | october 19 , 2007 | 8.95 ||
|| 49 | 5 | weight of what was | gloria muzio | pk simonds | october 26 , 2007 | 9.99 ||
|| 50 | 6 | double exposure | eric laneuville | laurie mccarthy | november 2 , 2007 | 9.18 ||
|| 51 | 7 | unhappy medium | frederick eo toye | breen frazier | november 9 , 2007 | 9.85 ||
|| 52 | 8 | bad blood | peter werner | teddy tenenbaum | november 16 , 2007 | 9.56 ||
|| 53 | 9 | all ghosts lead to grandview | frederick eo toye | pk simonds & laurie mccarthy | november 23 , 2007 | 9.98 ||
|| 54 | 10 | holiday spirit | steven robman | jeannine renshaw | december 14 , 2007 | 9.80 ||
|| 55 | 11 | slam (aka slambook) | mark rosman | karl schaefer & daniel sinclair | january 11 , 2008 | 9.86 ||
|| 56 | 12 | first do no harm | ian sander | john gray | january 18 , 2008 | 9.91 ||
|| 57 | 13 | home but not alone | eric laneuville | pk simonds & laurie mccarthy | april 4 , 2008 | 9.06 ||
|| 58 | 14 | the grave sitter | frederick eo toye | john gray | april 11 , 2008 | 8.55 ||
|| 59 | 15 | horror show | ian sander | jeannine renshaw | april 25 , 2008 | 8.98 ||
|| 60 | 16 | deadbeat dads | gloria muzio | mark b perry | may 2 , 2008 | 9.21 ||
|| 61 | 17 | stranglehold (part 1) | eric laneuville | laurie mccarthy & pk simonds | may 9 , 2008 | 8.78 ||

Question: Is it true that breen frazier wrote unhappy medium?
'''

table_data = [["no in series","no in season","title","directed by","written by", "original air date","us viewers (millions)"],["45", "1", "the underneath", "john gray", "john gray", "september 28, 2007", "8.72"],["46", "2", "don't try this at home", "ian sander", "teddy tenenbaum & laurie mccarthy", "october 5, 2007", "8.91"],["47", "3", "haunted hero", "eric laneuville", "breen frazier & karl schaefer", "october 12, 2007", "8.90"],["48", "4", "no safe place", "peter o'fallon", "jeannine renshaw", "october 19, 2007", "8.95"],["49", "5", "weight of what was", "gloria muzio", "pk simonds", "october 26, 2007", "9.99"],["50", "6", "double exposure", "eric laneuville", "laurie mccarthy", "november 2, 2007", "9.18"],["51", "7", "unhappy medium", "frederick eo toye", "breen frazier", "november 9, 2007", "9.85"],["52", "8", "bad blood", "peter werner", "teddy tenenbaum", "november 16, 2007", "9.56"],["53", "9", "all ghosts lead to grandview", "frederick eo toye", "pk simonds & laurie mccarthy", "november 23, 2007", "9.98"],["54", "10", "holiday spirit", "steven robman", "jeannine renshaw", "december 14, 2007", "9.80"],["55", "11", "slam (aka slambook)", "mark rosman", "karl schaefer & daniel sinclair", "january 11, 2008", "9.86"],["56", "12", "first do no harm", "ian sander", "john gray", "january 18, 2008", "9.91"],["57", "13", "home but not alone", "eric laneuville", "pk simonds & laurie mccarthy", "april 4, 2008", "9.06"],["58", "14", "the grave sitter", "frederick eo toye", "john gray", "april 11, 2008", "8.55"],["59", "15", "horror show", "ian sander", "jeannine renshaw", "april 25, 2008", "8.98"],["60", "16", "deadbeat dads", "gloria muzio", "mark b perry", "may 2, 2008", "9.21"],["61", "17", "stranglehold (part 1)", "eric laneuville", "laurie mccarthy & pk simonds", "may 9, 2008", "8.78"]]

# get the column by name
def get_column_by_name(table, column_name):
    column_index = table[0].index(column_name)
    column = []
    for row in table[1:]:
        column.append(row[column_index])
    return column

def solution(table_data):
    column_written_by = get_column_by_name(table_data, "written by")
    answer = "breen frazier" in column_written_by
    return answer

print(solution(table_data))

'''
Caption: none
Table:
||  | date | vs | opponent | score | attendance | record ||
|| 1 | january 5 , 1991 | at | detroit turbos | 8 - 18 | 6847 | loss ||
|| 2 | january 11 , 1991 | vs | new england blazers | 11 - 10 | 14789 | win ||
|| 3 | january 19 , 1991 | at | new york saints | 13 - 19 | 9081 | loss ||
|| 4 | january 26 , 1991 | vs | new york saints | 13 - 8 | 16282 | win ||
|| 5 | february 10 , 1991 | vs | detroit turbos | 12 - 14 | 16642 | loss ||
|| 6 | february 22 , 1991 | at | new england blazers | 11 - 13 | 7095 | loss ||
|| 7 | february 28 , 1991 | vs | pittsburgh bulls | 11 - 9 | 13712 | win ||
|| 8 | march 9 , 1991 | at | pittsburgh bulls | 15 - 7 | 8589 | win ||
|| 9 | march 17 , 1991 | vs | baltimore thunder | 14 - 17 | 16289 | loss ||

Question: Is it true that the score for game 9 was 11 - 9?
'''

table_data = [["", "date", "vs", "opponent", "score", "attendance", "record"],["1", "january 5 , 1991", "at", "detroit turbos", "8 - 18", "6847", "loss"],["2", "january 11 , 1991", "vs", "new england blazers", "11 - 10", "14789", "win"],["3", "january 19 , 1991", "at", "new york saints", "13 - 19", "9081", "loss"],["4", "january 26 , 1991", "vs", "new york saints", "13 - 8", "16282", "win"],["5", "february 10 , 1991", "vs", "detroit turbos", "12 - 14", "16642", "loss"],["6", "february 22 , 1991", "at", "new england blazers", "11 - 13", "7095", "loss"],["7", "february 28 , 1991", "vs", "pittsburgh bulls", "11 - 9", "13712", "win"],["8", "march 9 , 1991", "at", "pittsburgh bulls", "15 - 7", "8589", "win"],["9", "march 17 , 1991", "vs", "baltimore thunder", "14 - 17", "16289", "loss"]]

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

# Check if one number equals to another number
def equal_to(num1, num2):
    return num1 == num2
       
def solution(table_data):
    column_index = 4
    column_4 = get_column_by_index(table_data, column_index)
    game_9_score = get_column_cell_value(8, column_4)
    expected_score = "11 - 9"
    answer = equal_to(game_9_score, expected_score)
    return answer

print(solution(table_data))

'''
Caption: 2008 - 09 nbl season
Table:
|| date | home team | score | away team | venue | crowd | box score | report ||
|| 24 february | south dragons | 94 - 81 | townsville crocodiles | hisense arena | 3613 | box score | - ||
|| 25 february | melbourne tigers | 117 - 99 | new zealand breakers | state netball and hockey centre | 2998 | box score | - ||
|| 26 february | townsville crocodiles | 82 - 77 | south dragons | townsville entertainment centre | 4505 | box score | - ||
|| 27 february | new zealand breakers | 97 - 103 | melbourne tigers | north shore events centre | 4500 | box score | - ||
|| 28 february | south dragons | 101 - 78 | townsville crocodiles | hisense arena | 3007 | box score | - ||

Question: Is it true that the townsville crocodiles scored 82 points and the south dragons scored 77 points?
'''

table_data = [["date","home team","score","away team","venue","crowd","box score","report"],["24 february","south dragons","94 - 81","townsville crocodiles","hisense arena","3613","box score","-"],["25 february","melbourne tigers","117 - 99","new zealand breakers","state netball and hockey centre","2998","box score","-"],["26 february","townsville crocodiles","82 - 77","south dragons","townsville entertainment centre","4505","box score","-"],["27 february","new zealand breakers","97 - 103","melbourne tigers","north shore events centre","4500","box score","-"],["28 february","south dragons","101 - 78","townsville crocodiles","hisense arena","3007","box score","-"]]

# get the column by name
def get_column_by_name(table, column_name):
    column_index = table[0].index(column_name)
    column = []
    for row in table[1:]:
        column.append(row[column_index])
    return column

# Check if one number equals to another number
def equal_to(num1, num2):
    return num1 == num2

def solution(table_data):
    home_teams = get_column_by_name(table_data, "home team")
    scores = get_column_by_name(table_data, "score")
    away_teams = get_column_by_name(table_data, "away team")
    for home_team, score, away_team in zip(home_teams, scores, away_teams):
        if home_team.lower() == "townsville crocodiles" and away_team.lower() == "south dragons":
            answer = equal_to(score, "82 - 77")
            break 
    return answer

print(solution(table_data))
