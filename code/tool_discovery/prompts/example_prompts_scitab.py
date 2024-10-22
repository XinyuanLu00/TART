'''
Caption: A bag-of-concepts model improves relation extraction in a narrow knowledge domain with limited data Table 1: Performance of supervised learning models with different features.
Table:
|| Feature | LR P | LR R | LR F1 | SVM P | SVM R | SVM F1 | ANN P | ANN R | ANN F1 ||
|| +BoW | 0.93 | 0.91 | 0.92 | 0.94 | 0.92 | 0.93 | 0.91 | 0.91 | 0.91 ||
|| +BoC (Wiki-PubMed-PMC) | 0.94 | 0.92 | 0.93 | 0.94 | 0.92 | 0.93 | 0.91 | 0.91 | 0.91 ||
|| +BoC (GloVe) | 0.93 | 0.92 | 0.92 | 0.94 | 0.92 | 0.93 | 0.91 | 0.91 | 0.91 ||
|| +ASM | 0.90 | 0.85 | 0.88 | 0.90 | 0.86 | 0.88 | 0.89 | 0.89 | 0.89 ||
|| +Sentence Embeddings(SEs) | 0.89 | 0.89 | 0.89 | 0.90 | 0.86 | 0.88 | 0.88 | 0.88 | 0.88 ||
|| +BoC(Wiki-PubMed-PMC)+SEs | 0.92 | 0.92 | 0.92 | 0.94 | 0.92 | 0.93 | 0.91 | 0.91 | 0.91 ||

Question: Is it true that The models using BoC outperform models using BoW as well as ASM features?
'''

table_data = [["Feature", "LR P", "LR R", "LR F1", "SVM P", "SVM R", "SVM F1", "ANN P", "ANN R", "ANN F1"],["+BoW", "0.93", "0.91", "0.92", "0.94", "0.92", "0.93", "0.91", "0.91", "0.91"],["+BoC (Wiki-PubMed-PMC)", "0.94", "0.92", "0.93", "0.94", "0.92", "0.93", "0.91", "0.91", "0.91"],["+BoC (GloVe)", "0.93", "0.92", "0.92", "0.94", "0.92", "0.93", "0.91", "0.91", "0.91"],["+ASM", "0.90", "0.85", "0.88", "0.90", "0.86", "0.88", "0.89", "0.89", "0.89"],["+Sentence Embeddings(SEs)", "0.89", "0.89", "0.89", "0.90", "0.86", "0.88", "0.88", "0.88", "0.88"],["+BoC(Wiki-PubMed-PMC)+SEs", "0.92", "0.92", "0.92", "0.94", "0.92", "0.93", "0.91", "0.91", "0.91"]]

# Calculate average F1 scores
def average_f1(table, feature_prefix):
    f1_scores = []
    for row in table[1:]:  
        if feature_prefix in row[0]:
            f1_scores.append(float(row[3]))  
            f1_scores.append(float(row[6]))  
            f1_scores.append(float(row[9]))  
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0

def solution(table_data):
    bow_f1 = average_f1(table_data, "+BoW")
    asm_f1 = average_f1(table_data, "+ASM")
    boc_f1 = average_f1(table_data, "+BoC")
    answer = boc_f1 > bow_f1 and boc_f1 > asm_f1
    return answer

print(solution(table_data))

'''
Caption: Towards Quantifying the Distance between Opinions Table 5: We compare the quality of variants of Opinion Distance measures on opinion clustering task with ARI.
Table:
||  | Difference Function | Seanad Abolition | Video Games | Pornography ||
|| OD-parse | Absolute | 0.01 | -0.01 | 0.07 ||
|| OD-parse | JS div. | 0.01 | -0.01 | -0.01 ||
|| OD-parse | EMD | 0.07 | 0.01 | -0.01 ||
|| OD | Absolute | 0.54 | 0.56 | 0.41 ||
|| OD | JS div. | 0.07 | -0.01 | -0.02 ||
|| OD | EMD | 0.26 | -0.01 | 0.01 ||
|| OD (no polarity shifters) | Absolute | 0.23 | 0.08 | 0.04 ||
|| OD (no polarity shifters) | JS div. | 0.09 | -0.01 | -0.02 ||
|| OD (no polarity shifters) | EMD | 0.10 | 0.01 | -0.01 ||

Question: Is it true that OD significantly outperforms OD-parse: We observe that compared to OD-parse, OD is much more accurate?
'''

table_data = [["", "Difference Function", "Seanad Abolition", "Video Games", "Pornography"],["OD-parse", "Absolute", "0.01", "-0.01", "0.07"],["OD-parse", "JS div.", "0.01", "-0.01", "-0.01"],["OD-parse", "EMD", "0.07", "0.01", "-0.01"],["OD", "Absolute", "0.54", "0.56", "0.41"],["OD", "JS div.", "0.07", "-0.01", "-0.02"],["OD", "EMD", "0.26", "-0.01", "0.01"],["OD (no polarity shifters)", "Absolute", "0.23", "0.08", "0.04"],["OD (no polarity shifters)", "JS div.", "0.09", "-0.01", "-0.02"],["OD (no polarity shifters)", "EMD", "0.10", "0.01", "-0.01"]]

# Calculate the average scores
def average_scores(table, method):
    scores = []
    for row in table:
        if row[0].startswith(method):
            scores.extend([float(x) for x in row[2:]])
    return sum(scores) / len(scores)

def solution(table_data):
    od_parse_score = average_scores(table_data[1:], "OD-parse")
    od_score = average_scores(table_data[1:], "OD")
    answer = od_score > od_parse_score
    return answer

print(solution(table_data))  

'''
Caption: Evaluating Layers of Representation in Neural Machine Translation on Part-of-Speech and Semantic Tagging Tasks Table 2: POS and SEM tagging accuracy with baselines and an upper bound. MFT: most frequent tag; UnsupEmb: classifier using unsupervised word embeddings; Word2Tag: upper bound encoder-decoder.
Table:
||  | MFT | UnsupEmb | Word2Tag ||
|| POS | 91.95 | 87.06 | 95.55 ||
|| SEM | 82.00 | 81.11 | 91.41 ||

Question: Is it true that The UnsupEmb baseline performs rather poorly on both POS and SEM tagging?
'''

table_data = [["", "MFT", "UnsupEmb", "Word2Tag"],["POS", "91.95", "87.06", "95.55"],["SEM", "82.00", "81.11", "91.41"]]

# Check if the unsupemb is poor
def is_unsupemb_poor(data):
    unsupemb_scores = []
    for row in data[1:]:  
        unsupemb_scores.append(float(row[2]))  
    return all(score < 90 for score in unsupemb_scores)

def solution(table_data):
    answer = is_unsupemb_poor(table_data)
    return answer

print(solution(table_data))  
