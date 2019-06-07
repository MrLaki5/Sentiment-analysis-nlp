import pandas as pd
from stemmer import stemmer
import tokenizer
from eng_dict import buildEnglish
from ger_dict import build_german
import levenshtein
import json


# Function for calculating sum of weights of comment from specific dictionary
def comment_weight_calculation(dict_curr, t_original, t_stemmed, distance_filter):
    summ = 0
    for token, stemm_token in zip(t_original, t_stemmed):
        if stemm_token in dict_curr:
            min_distance = -1
            temp_weight = 0

            for key in dict_curr[stemm_token].keys():
                curr_distance = levenshtein.iterative_levenshtein(key, token)
                if curr_distance < min_distance or min_distance < 0:
                    min_distance = curr_distance
                    temp_weight = dict_curr[stemm_token][key]
            if temp_weight <= distance_filter:
                # TODO remove if needed :)
                if temp_weight < 0:
                    temp_weight *= 0.44
                summ += temp_weight
    return summ


# START MAIN
# Load dictionaries
# English
engDict = buildEnglish()
engDictStemmed = stemmer.stem_dictionary(engDict)
# Log nonlinear stems in engDict
cnt = 0
with open("./stemmer/logs/log_nonlinear_eng.txt", "w", encoding="utf8") as f:
    for item in engDictStemmed:
        if len(engDictStemmed[item]) > 1:
            f.write("\n\n" + str(cnt) + ": " + item)
            f.write("\n" + "Items(" + str(len(engDictStemmed[item])) + "): ")
            for key in engDictStemmed[item]:
                f.write("\n\t" + key + " -> " + str(engDictStemmed[item][key]))
            cnt += 1
# German
gerDict = build_german()
gerDictStemmed = stemmer.stem_dictionary(gerDict)
# Log nonlinear stems in gerDict
cnt = 0
with open("./stemmer/logs/log_nonlinear_ger.txt", "w", encoding="utf8") as f:
    for item in gerDictStemmed:
        if len(gerDictStemmed[item]) > 1:
            f.write("\n\n" + str(cnt) + ": " + item)
            f.write("\n" + "Items(" + str(len(gerDictStemmed[item])) + "): ")
            for key in gerDictStemmed[item]:
                f.write("\n\t" + key + " -> " + str(gerDictStemmed[item][key]))
            cnt += 1

list_summ_ger = []
list_summ_eng = []
list_out = []

# Load data set
with open("../movie_dataset/stemmed_dict.json", encoding="utf8") as f:
    data_set = json.load(f)
# Iterate through data set
for data in data_set:
    sentiment_class = data['class_att']
    tokens_stemmed = data['tokens_original']
    tokens_original = data['tokens_stemmed']
    summ_eng = comment_weight_calculation(engDictStemmed, tokens_original, tokens_stemmed, 80)
    summ_ger = comment_weight_calculation(gerDictStemmed,  tokens_original, tokens_stemmed, 80)
    list_summ_eng.append(summ_eng)
    list_summ_ger.append(summ_ger)
    list_out.append(sentiment_class)
print(list_summ_eng)
print("Finished")
