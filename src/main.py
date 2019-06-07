import pandas as pd
from stemmer import stemmer
from eng_dict import buildEnglish
from ger_dict import build_german
import levenshtein
import json
from src.naive_bayes import naive_bayes
import os.path
import comment_process_pool


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

# Load data set from json
data_set_json = None
if os.path.isfile("../movie_dataset/stemmed_dict.json"):
    with open("../movie_dataset/stemmed_dict.json", encoding="utf8") as f:
        data_set_json = json.load(f)

# Load data set
data_set = pd.read_csv("../movie_dataset/SerbMR-2C.csv")

# Menu
work_flag = 1
while work_flag == 1:
    print("Choose action:")
    print("--------------------------")
    print("1. Lexicon neural network")
    print("2. Bayes-naive")
    print("3. Do tokenization of comments")
    print("4. Exit")
    print("--------------------------")
    user_action = input("Action: ")
    if user_action is "1":
        if data_set_json is not None:
            list_summ_ger = []
            list_summ_eng = []
            list_out = []
            for data in data_set_json:
                sentiment_class = data['class_att']
                tokens_stemmed = data['tokens_original']
                tokens_original = data['tokens_stemmed']
                summ_eng = comment_weight_calculation(engDictStemmed, tokens_original, tokens_stemmed, 80)
                summ_ger = comment_weight_calculation(gerDictStemmed,  tokens_original, tokens_stemmed, 80)
                list_summ_eng.append(summ_eng)
                list_summ_ger.append(summ_ger)
                list_out.append(sentiment_class)
            print("List of sum eng dic:" + str(list_summ_eng))
            print("List of sum ger dic:" + str(list_summ_ger))
            print("List of should outcome:" + str(list_out))
            print("Finished")
        else:
            print("Tokenization of comment has not been done.")
    elif user_action is "2":
        print(naive_bayes())
    elif user_action is "3":
        thread_flag = 1
        thread_num = 1
        while thread_flag is 1:
            print("Choose number of threads: ")
            print("--------------------------")
            thread_num = input("Action: ")
            try:
                thread_num = int(thread_num)
                thread_flag = 0
            except Exception as ex:
                pass
            process = comment_process_pool.CommentProcessPool(thread_num)
            process.start(data_set)
            data_processed = process.get_data()
            with open("../movie_dataset/stemmed_dict.json", "w", encoding='utf-8') as f:
                json.dump(data_processed, f)
            data_set_json = data_processed
        print("Process finished!")
    elif user_action is "4":
        work_flag = 0
