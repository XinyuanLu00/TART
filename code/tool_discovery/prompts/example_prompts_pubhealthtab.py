'''
Caption: Sacred Work: Planned Parenthood and Its Clergy Alliances - Tom Davis - Google Books
Table:
|| Title | Sacred Work: Planned Parenthood and Its Clergy Alliances ||
|| Author | Tom Davis ||
|| Edition | illustrated ||
|| Publisher | Rutgers University Press, 2005 ||
|| ISBN | 0813534933, 9780813534930 ||
|| Length | 245 pages ||
|| Subjects | Social Science Sociology of ReligionSocial Science / Sociology of Religion ||
|| Export Citation | BiBTeX EndNote RefMan ||

Question: Is it true that Planned Parenthood has a history of clergy alliances?
'''

table_data = [["Title", "Sacred Work: Planned Parenthood and Its Clergy Alliances"],["Author", "Tom Davis"],["Edition", "illustrated"],["Publisher", "Rutgers University Press, 2005"],["ISBN", "0813534933, 9780813534930"],["Length", "245 pages"],["Subjects", "Social Science Sociology of ReligionSocial Science / Sociology of Religion"],["Export Citation", "BiBTeX EndNote RefMan"]
]

# Get the row by name
def get_row_by_name(table, key):
    for row in table:
        if row[0].lower() == key.lower():
            return row[1]
    return None

def solution(table_data):
    title = get_row_by_name(table_data, "Title")
    subjects = get_row_by_name(table_data, "Subjects")
    answer = True
    if "clergy alliances" in title.lower() or "clergy" in subjects.lower():
            return answer
    answer = False
    return answer

print(solution(table_data))

'''
Caption: BUCS Return to Play
Table:
|| Monday 7 September | Entries open ||
|| Friday 18 September | Entries close ||
|| Friday 2 October | Leagues live on BUCS Play ||
|| Monday 5 October - Wednesday 16 December | Fixture window ||

Question: Is it true that on Friday 2 October leagues live on BUCS Play?
'''

table_data = [["Monday 7 September", "Entries open"],["Friday 18 September", "Entries close"],["Friday 2 October", "Leagues live on BUCS Play"],["Monday 5 October - Wednesday 16 December", "Fixture window"]]

# Check if a specific event occurs on a given date
def check_event_on_date(table, date_query, event_query):
    for row in table:
        date, event = row
        if date.lower() == date_query.lower() and event_query.lower() in event.lower():
            return True
    return False

def solution(table_data):
    query_date = "Friday 2 October"
    query_event = "Leagues live on BUCS Play"
    answer = check_event_on_date(table_data, query_date, query_event)
    return answer

print(solution(table_data))

'''
Caption: Reperfusion Strategies in Acute Coronary Syndromes | Circulation Research
Table:
|| ACCF | American College of Cardiology Foundation ||
|| AHA | American Heart Association ||
|| D2B | door-to-balloon ||
|| ED | emergency department ||
|| FMC | first medical contact ||
|| MI | myocardial infarction ||
|| NSTEACS | non-ST-segment elevation acute coronary syndrome ||
|| STEMI | ST-segment elevation MI ||
|| TIMI | thrombolysis in myocardial infarction ||

Question: Is it true that D2B stands for door-to-belly as a medical acronym?
'''
table_data = [["ACCF", "American College of Cardiology Foundation"],["AHA", "American Heart Association"],["D2B", "door-to-balloon"],["ED", "emergency department"],["FMC", "first medical contact"],["MI", "myocardial infarction"],["NSTEACS", "non-ST-segment elevation acute coronary syndrome"],["STEMI", "ST-segment elevation MI"],["TIMI", "thrombolysis in myocardial infarction"]
]

# Get the row by name
def get_row_by_name(table, acronym):
    for row in table: 
        if row[0] == acronym:
            return row[1]
    return None

def solution(table_data):
    definition = get_row_by_name(table_data, "D2B")
    answer = definition.lower() == "door-to-belly"
    return answer

print(solution(table_data)) 
