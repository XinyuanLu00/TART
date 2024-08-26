'''
Caption: null
Table:
||   | October 31 2009 | November 1 2008 ||
|| Fair value of forward exchange contracts asset (liability) | $6427 | $-23158 (23158) ||
|| Fair value of forward exchange contracts after a 10% unfavorable movement in foreign currency exchange rates asset (liability) | $20132 | $-9457 (9457) ||
|| Fair value of forward exchange contracts after a 10% favorable movement in foreign currency exchange rates liability | $-6781 (6781) | $-38294 (38294) ||

Context:  interest rate to a variable interest rate based on the three-month libor plus 2.05% ( 2.05 % ) ( 2.34% ( 2.34 % ) as of october 31 , 2009 ) . if libor changes by 100 basis points , our annual interest expense would change by $ 3.8 million . foreign currency exposure as more fully described in note 2i . in the notes to consolidated financial statements contained in item 8 of this annual report on form 10-k , we regularly hedge our non-u.s . dollar-based exposures by entering into forward foreign currency exchange contracts . the terms of these contracts are for periods matching the duration of the underlying exposure and generally range from one month to twelve months . currently , our largest foreign currency exposure is the euro , primarily because our european operations have the highest proportion of our local currency denominated expenses . relative to foreign currency exposures existing at october 31 , 2009 and november 1 , 2008 , a 10% ( 10 % ) unfavorable movement in foreign currency exchange rates over the course of the year would not expose us to significant losses in earnings or cash flows because we hedge a high proportion of our year-end exposures against fluctuations in foreign currency exchange rates . the market risk associated with our derivative instruments results from currency exchange rate or interest rate movements that are expected to offset the market risk of the underlying transactions , assets and liabilities being hedged . the counterparties to the agreements relating to our foreign exchange instruments consist of a number of major international financial institutions with high credit ratings . we do not believe that there is significant risk of nonperformance by these counterparties because we continually monitor the credit ratings of such counterparties . while the contract or notional amounts of derivative financial instruments provide one measure of the volume of these transactions , they do not represent the amount of our exposure to credit risk . the amounts potentially subject to credit risk ( arising from the possible inability of counterparties to meet the terms of their contracts ) are generally limited to the amounts , if any , by which the counterparties 2019 obligations under the contracts exceed our obligations to the counterparties . the following table illustrates the effect that a 10% ( 10 % ) unfavorable or favorable movement in foreign currency exchange rates , relative to the u.s . dollar , would have on the fair value of our forward exchange contracts as of october 31 , 2009 and november 1 , 2008: . fair value of forward exchange contracts after a 10% ( 10 % ) unfavorable movement in foreign currency exchange rates asset ( liability ) . . . . . . . . . $ 20132 $ ( 9457 ) fair value of forward exchange contracts after a 10% ( 10 % ) favorable movement in foreign currency exchange rates liability . . . . . . . . . . . . . . . . . . . . . . $ ( 6781 ) $ ( 38294 ) the calculation assumes that each exchange rate would change in the same direction relative to the u.s . dollar . in addition to the direct effects of changes in exchange rates , such changes typically affect the volume of sales or the foreign currency sales price as competitors 2019 products become more or less attractive . our sensitivity analysis of the effects of changes in foreign currency exchange rates does not factor in a potential change in sales levels or local currency selling prices.

Question: What is the interest expense in 2009?
'''

table_data = [["", "October 31 2009", "November 1 2008"],["Fair value of forward exchange contracts asset (liability)", "$6427", "$-23158 (23158)"],["Fair value of forward exchange contracts after a 10% unfavorable movement in foreign currency exchange rates asset (liability)", "$20132", "$-9457 (9457)"],["Fair value of forward exchange contracts after a 10% favorable movement in foreign currency exchange rates liability", "$-6781 (6781)", "$-38294 (38294)"]]

context = "interest rate to a variable interest rate based on the three-month libor plus 2.05% ( 2.05 % ) ( 2.34% ( 2.34 % ) as of october 31 , 2009 ) . if libor changes by 100 basis points , our annual interest expense would change by $ 3.8 million . foreign currency exposure as more fully described in note 2i . in the notes to consolidated financial statements contained in item 8 of this annual report on form 10-k , we regularly hedge our non-u.s . dollar-based exposures by entering into forward foreign currency exchange contracts . the terms of these contracts are for periods matching the duration of the underlying exposure and generally range from one month to twelve months . currently , our largest foreign currency exposure is the euro , primarily because our european operations have the highest proportion of our local currency denominated expenses . relative to foreign currency exposures existing at october 31 , 2009 and november 1 , 2008 , a 10% ( 10 % ) unfavorable movement in foreign currency exchange rates over the course of the year would not expose us to significant losses in earnings or cash flows because we hedge a high proportion of our year-end exposures against fluctuations in foreign currency exchange rates . the market risk associated with our derivative instruments results from currency exchange rate or interest rate movements that are expected to offset the market risk of the underlying transactions , assets and liabilities being hedged . the counterparties to the agreements relating to our foreign exchange instruments consist of a number of major international financial institutions with high credit ratings . we do not believe that there is significant risk of nonperformance by these counterparties because we continually monitor the credit ratings of such counterparties . while the contract or notional amounts of derivative financial instruments provide one measure of the volume of these transactions , they do not represent the amount of our exposure to credit risk . the amounts potentially subject to credit risk ( arising from the possible inability of counterparties to meet the terms of their contracts ) are generally limited to the amounts , if any , by which the counterparties 2019 obligations under the contracts exceed our obligations to the counterparties . the following table illustrates the effect that a 10% ( 10 % ) unfavorable or favorable movement in foreign currency exchange rates , relative to the u.s . dollar , would have on the fair value of our forward exchange contracts as of october 31 , 2009 and november 1 , 2008: . fair value of forward exchange contracts after a 10% ( 10 % ) unfavorable movement in foreign currency exchange rates asset ( liability ) . . . . . . . . . $ 20132 $ ( 9457 ) fair value of forward exchange contracts after a 10% ( 10 % ) favorable movement in foreign currency exchange rates liability . . . . . . . . . . . . . . . . . . . . . . $ ( 6781 ) $ ( 38294 ) the calculation assumes that each exchange rate would change in the same direction relative to the u.s . dollar . in addition to the direct effects of changes in exchange rates , such changes typically affect the volume of sales or the foreign currency sales price as competitors 2019 products become more or less attractive . our sensitivity analysis of the effects of changes in foreign currency exchange rates does not factor in a potential change in sales levels or local currency selling prices."

import re

# extract a numeric value from a context string
def extract_value_from_context(context, pattern):
    search_result = re.search(pattern, context)
    if search_result:
        return float(search_result.group(1))
    else:
        return None

def solution(context):
    answer = extract_value_from_context(context, r"\$ (\d+\.\d+) million")
    return answer

print(solution(context))

'''
Caption: null
Table:
||   | September 30, 2008 | September 30, 2009 | September 30, 2010 | September 30, 2011 | September 30, 2012 | September 30, 2013 ||
|| Apple Inc. | $100 | $163 | $250 | $335 | $589 | $431 ||
|| S&P 500 Index | $100 | $93 | $103 | $104 | $135 | $161 ||
|| S&P Computer Hardware Index | $100 | $118 | $140 | $159 | $255 | $197 ||
|| Dow Jones US Technology Supersector Index | $100 | $111 | $124 | $128 | $166 | $175 ||

Context: table of contents company stock performance the following graph shows a five-year comparison of cumulative total shareholder return , calculated on a dividend reinvested basis , for the company , the s&p 500 index , the s&p computer hardware index , and the dow jones u.s . technology supersector index . the graph assumes $ 100 was invested in each of the company 2019s common stock , the s&p 500 index , the s&p computer hardware index , and the dow jones u.s . technology supersector index as of the market close on september 30 , 2008 . data points on the graph are annual . note that historic stock price performance is not necessarily indicative of future stock price performance . fiscal year ending september 30 . copyright 2013 s&p , a division of the mcgraw-hill companies inc . all rights reserved . copyright 2013 dow jones & co . all rights reserved . *$ 100 invested on 9/30/08 in stock or index , including reinvestment of dividends . september 30 , september 30 , september 30 , september 30 , september 30 , september 30.

Question: by how much did apple inc . outperform the s&p computer hardware index over the above mentioned 6 year period?
'''

table_data = [["", "september 30 2008", "september 30 2009", "september 30 2010", "september 30 2011", "september 30 2012", "september 30 2013"],["apple inc .", "$ 100", "$ 163", "$ 250", "$ 335", "$ 589", "$ 431"],["s&p 500 index", "$ 100", "$ 93", "$ 103", "$ 104", "$ 135", "$ 161"],["s&p computer hardware index", "$ 100", "$ 118", "$ 140", "$ 159", "$ 255", "$ 197"],["dow jones us technology supersector index", "$ 100", "$ 111", "$ 124", "$ 128", "$ 166", "$ 175"]]

context = "table of contents company stock performance the following graph shows a five-year comparison of cumulative total shareholder return , calculated on a dividend reinvested basis , for the company , the s&p 500 index , the s&p computer hardware index , and the dow jones u.s . technology supersector index . the graph assumes $ 100 was invested in each of the company 2019s common stock , the s&p 500 index , the s&p computer hardware index , and the dow jones u.s . technology supersector index as of the market close on september 30 , 2008 . data points on the graph are annual . note that historic stock price performance is not necessarily indicative of future stock price performance . fiscal year ending september 30 . copyright 2013 s&p , a division of the mcgraw-hill companies inc . all rights reserved . copyright 2013 dow jones & co . all rights reserved . *$ 100 invested on 9/30/08 in stock or index , including reinvestment of dividends . september 30 , september 30 , september 30 , september 30 , september 30 , september 30."

# get row index by value
def get_row_index_by_value(table, row_value):
    for i in range(len(table)):
        if table[i][0] == row_value:
            return i
        
# extract the price from a string
def extract_price(price_string):
    return float(price_string.replace('$', '').replace(',', '')) 

# get the column by index
def get_column_by_index(table, column_index):
    column = []
    for row in table:
        if len(row) > column_index:
            column.append(row[column_index])
        else:
            column.append(None)
    return column

# subtract two numbers
def subtract(minuend, subtrahend):
    return minuend - subtrahend

# calculate the change rate
def calculate_change_rate(start_value, end_value):
    return ((end_value - start_value) / start_value) 

def solution(table_data):
    entity_column = get_column_by_index(table_data, 0)
    apple_row_index = entity_column.index("apple inc .")
    sp_hardware_row_index = entity_column.index("s&p computer hardware index")
    apple_final_value = extract_price(table_data[apple_row_index][6])
    sp_hardware_final_value = extract_price(table_data[sp_hardware_row_index][6])
    apple_growth = calculate_change_rate(extract_price(table_data[apple_row_index][1]), apple_final_value)
    sp_hardware_growth = calculate_change_rate(extract_price(table_data[sp_hardware_row_index][1]), sp_hardware_final_value)
    answer = subtract(apple_growth, sp_hardware_growth)
    return answer

print(solution(table_data))

'''
Caption: null
Table:
|| balance december 31 2006 | $ 740507 ||
|| additions during period 2014depreciation and amortization expense | 96454 ||
|| deductions during period 2014disposition and retirements of property | -80258 ( 80258 ) ||
|| balance december 31 2007 | 756703 ||
|| additions during period 2014depreciation and amortization expense | 101321 ||
|| deductions during period 2014disposition and retirements of property | -11766 ( 11766 ) ||
|| balance december 31 2008 | 846258 ||
|| additions during period 2014depreciation and amortization expense | 103.698 ||
|| deductions during period 2014disposition and retirements of property | -11869 ( 11869 ) ||
|| balance december 31 2009 | $ 938087 ||

Context: federal realty investment trust schedule iii summary of real estate and accumulated depreciation 2014continued three years ended december 31 , 2009 reconciliation of accumulated depreciation and amortization ( in thousands ) .

Question: considering the years 2006-2009 , what is the value of the average additions?
'''

table_data = [["balance december 31 2006", "$ 740507"],["additions during period 2014depreciation and amortization expense", "96454"],["deductions during period 2014disposition and retirements of property", "-80258 ( 80258 )"],["balance december 31 2007", "756703"],["additions during period 2014depreciation and amortization expense", "101321"],["deductions during period 2014disposition and retirements of property", "-11766 ( 11766 )"],["balance december 31 2008", "846258"],["additions during period 2014depreciation and amortization expense", "103.698"],["deductions during period 2014disposition and retirements of property", "-11869 ( 11869 )"],["balance december 31 2009", "$ 938087"]]

context = "federal realty investment trust schedule iii summary of real estate and accumulated depreciation 2014continued three years ended december 31 , 2009 reconciliation of accumulated depreciation and amortization ( in thousands ) ."

# extract the price from a string
def extract_price(price_string):
    return float(price_string.replace('$', '').replace(',', ''))

# add two numbers
def add(num1, num2):
    return num1 + num2

# multiply two numbers
def multiply(num1, num2):
    return num1 * num2

# calculate the average numbers
def average(values):
    total = sum(values)
    count = len(values)
    return total / count

def solution(table_data):
    additions_values = [extract_price(row[1]) for row in table_data if "additions during period" in row[0]]
    additions_values[-1] = multiply(additions_values[-1], 1000)
    answer = average(additions_values)    
    return answer

print(solution(table_data))

'''
Caption: null
Table:
||  | 2019 | 2018 | 2017 ||
|| cost of sales | $ 20628 | $ 18733 | $ 12569 ||
|| research and development | 75305 | 81444 | 51258 ||
|| selling marketing general and administrative | 51829 | 50988 | 40361 ||
|| special charges | 2538 | 2014 | 2014 ||
|| total stock-based compensation expense | $ 150300 | $ 151165 | $ 104188 ||

Context: expected term 2014 the company uses historical employee exercise and option expiration data to estimate the expected term assumption for the black-scholes grant-date valuation . the company believes that this historical data is currently the best estimate of the expected term of a new option , and that generally its employees exhibit similar exercise behavior . risk-free interest rate 2014 the yield on zero-coupon u.s . treasury securities for a period that is commensurate with the expected term assumption is used as the risk-free interest rate . expected dividend yield 2014 expected dividend yield is calculated by annualizing the cash dividend declared by the company 2019s board of directors for the current quarter and dividing that result by the closing stock price on the date of grant . until such time as the company 2019s board of directors declares a cash dividend for an amount that is different from the current quarter 2019s cash dividend , the current dividend will be used in deriving this assumption . cash dividends are not paid on options , restricted stock or restricted stock units . in connection with the acquisition , the company granted restricted stock awards to replace outstanding restricted stock awards of linear employees . these restricted stock awards entitle recipients to voting and nonforfeitable dividend rights from the date of grant . stock-based compensation expensexp p the amount of stock-based compensation expense recognized during a period is based on the value of the awards that are ultimately expected to vest . forfeitures are estimated at the time of grant and revised , if necessary , in subsequent periods if actual forfeitures differ from those estimates . the term 201cforfeitures 201d is distinct from 201ccancellations 201d or 201cexpirations 201d and represents only the unvested portion of the surrendered stock-based award . based on an analysis of its historical forfeitures , the company has applied an annual forfeitureff rate of 5.0% ( 5.0 % ) to all unvested stock-based awards as of november 2 , 2019 . this analysis will be re-evaluated quarterly and the forfeiture rate will be adjusted as necessary . ultimately , the actual expense recognized over the vesting period will only be for those awards that vest . total stock-based compensation expense recognized is as follows: . as of november 2 , 2019 and november 3 , 2018 , the company capitalized $ 6.8 million and $ 7.1 million , respectively , of stock-based compensation in inventory . additional paid-in-capital ( apic ) pp poolp p ( ) the company adopted asu 2016-09 during fiscal 2018 . asu 2016-09 eliminated the apic pool and requires that excess tax benefits and tax deficiencies be recorded in the income statement when awards are settled . as a result of this adoption the company recorded total excess tax benefits of $ 28.7 million and $ 26.2 million in fiscal 2019 and fiscal 2018 , respectively , from its stock-based compensation payments within income tax expense in its consolidated statements of income . for fiscal 2017 , the apic pool represented the excess tax benefits related to stock-based compensation that were available to absorb future tax deficiencies . if the amount of future tax deficiencies was greater than the available apic pool , the company recorded the excess as income tax expense in its consolidated statements of income . for fiscal 2017 , the company had a sufficient apic pool to cover any tax deficiencies recorded and as a result , these deficiencies did not affect its results of operations . analog devices , inc . notes to consolidated financial statements 2014 ( continued ) .

Question: what is the growth rate in the r&d in 2019?
'''
table_data = [["", "2019", "2018", "2017"],["cost of sales", "$ 20628", "$ 18733", "$ 12569"],["research and development", "75305", "81444", "51258"],["selling marketing general and administrative", "51829", "50988", "40361"],["special charges", "2538", "2014", "2014"],["total stock-based compensation expense", "$ 150300", "$ 151165", "$ 104188"]]

context = "expected term 2014 the company uses historical employee exercise and option expiration data to estimate the expected term assumption for the black-scholes grant-date valuation . the company believes that this historical data is currently the best estimate of the expected term of a new option , and that generally its employees exhibit similar exercise behavior . risk-free interest rate 2014 the yield on zero-coupon u.s . treasury securities for a period that is commensurate with the expected term assumption is used as the risk-free interest rate . expected dividend yield 2014 expected dividend yield is calculated by annualizing the cash dividend declared by the company 2019s board of directors for the current quarter and dividing that result by the closing stock price on the date of grant . until such time as the company 2019s board of directors declares a cash dividend for an amount that is different from the current quarter 2019s cash dividend , the current dividend will be used in deriving this assumption . cash dividends are not paid on options , restricted stock or restricted stock units . in connection with the acquisition , the company granted restricted stock awards to replace outstanding restricted stock awards of linear employees . these restricted stock awards entitle recipients to voting and nonforfeitable dividend rights from the date of grant . stock-based compensation expensexp p the amount of stock-based compensation expense recognized during a period is based on the value of the awards that are ultimately expected to vest . forfeitures are estimated at the time of grant and revised , if necessary , in subsequent periods if actual forfeitures differ from those estimates . the term 201cforfeitures 201d is distinct from 201ccancellations 201d or 201cexpirations 201d and represents only the unvested portion of the surrendered stock-based award . based on an analysis of its historical forfeitures , the company has applied an annual forfeitureff rate of 5.0% ( 5.0 % ) to all unvested stock-based awards as of november 2 , 2019 . this analysis will be re-evaluated quarterly and the forfeiture rate will be adjusted as necessary . ultimately , the actual expense recognized over the vesting period will only be for those awards that vest . total stock-based compensation expense recognized is as follows: . as of november 2 , 2019 and november 3 , 2018 , the company capitalized $ 6.8 million and $ 7.1 million , respectively , of stock-based compensation in inventory . additional paid-in-capital ( apic ) pp poolp p ( ) the company adopted asu 2016-09 during fiscal 2018 . asu 2016-09 eliminated the apic pool and requires that excess tax benefits and tax deficiencies be recorded in the income statement when awards are settled . as a result of this adoption the company recorded total excess tax benefits of $ 28.7 million and $ 26.2 million in fiscal 2019 and fiscal 2018 , respectively , from its stock-based compensation payments within income tax expense in its consolidated statements of income . for fiscal 2017 , the apic pool represented the excess tax benefits related to stock-based compensation that were available to absorb future tax deficiencies . if the amount of future tax deficiencies was greater than the available apic pool , the company recorded the excess as income tax expense in its consolidated statements of income . for fiscal 2017 , the company had a sufficient apic pool to cover any tax deficiencies recorded and as a result , these deficiencies did not affect its results of operations . analog devices , inc . notes to consolidated financial statements 2014 ( continued ) ." 

# get the column by name
def get_column_by_name(table, column_name):
    column_index = table[0].index(column_name)
    column = []
    for row in table:
        column.append(row[column_index])
    return column

# extract the price from a string
def extract_price(price_string):
    return float(price_string.replace('$', '').replace(',', '')) 

# calculate the change rate
def calculate_change_rate(start_value, end_value):
    return ((end_value - start_value) / start_value)

def solution(table_data):
    rd_column_2019 = get_column_by_name(table_data, "2019")
    rd_2019 = extract_price(rd_column_2019[2])  
    rd_column_2018 = get_column_by_name(table_data, "2018")
    rd_2018 = extract_price(rd_column_2018[2])  
    answer = calculate_change_rate(rd_2018, rd_2019)
    return answer

print(solution(table_data))