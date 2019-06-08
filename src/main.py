import pandas as pd
from stemmer import stemmer
from eng_dict import buildEnglish
from ger_dict import build_german
import sentiment_logic
import json
from src.naive_bayes import naive_bayes
import os.path
import comment_process_pool
import plotting
import logging
from sklearn.metrics import accuracy_score
from datetime import datetime
from neural_nets import keras_adaline


# Logger configuration for both console and file
FILE_LOG = True
CONSOLE_LOG = True

if __name__ == '__main__':
    log_name = datetime.now().strftime('%Y-%m-%d--%H-%M-%S') + ".log"
    if FILE_LOG:
        logging.basicConfig(level=logging.DEBUG, handlers=[logging.FileHandler('./logs/' + log_name, 'w', 'utf-8')])
        if CONSOLE_LOG:
            logging.getLogger().addHandler(logging.StreamHandler())
    else:
        if CONSOLE_LOG:
            logging.basicConfig(level=logging.DEBUG)
    logging.debug("Logger started!")


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
if os.path.isfile("../movie_dataset/stemmed_dict_2.json"):
    with open("../movie_dataset/stemmed_dict_2.json", encoding="utf8") as f:
        data_set_json = json.load(f)

# Load data set
data_set = pd.read_csv("../movie_dataset/SerbMR-2C.csv")

# Menu
work_flag = 1
classes_num = 2
while work_flag == 1:
    print("Choose action:")
    print("--------------------------")
    print("1. Lexicon neural network")
    print("2. Bayes-naive")
    print("3. Do tokenization of comments")
    print("4. Choose number of classes, current: " + str(classes_num))
    print("5. Adaline without bias")
    print("6. Adaline with bias")
    print("7. Exit")
    print("--------------------------")
    user_action = input("Action: ")
    if user_action is "1":
        if data_set_json is not None:
            list_summ_ger = []
            list_summ_eng = []
            list_out = []
            for data in data_set_json:
                sentiment_class = data['class_att']
                tokens_original = data['tokens_original']
                tokens_stemmed = data['tokens_stemmed']
                summ_eng = sentiment_logic.comment_weight_calculation(engDictStemmed, "English", tokens_original, tokens_stemmed, 5, modification_use=False, amplification_use=False)
                summ_ger = sentiment_logic.comment_weight_calculation(gerDictStemmed, "German", tokens_original, tokens_stemmed, 5, modification_use=False, amplification_use=False)
                list_summ_eng.append(summ_eng)
                list_summ_ger.append(summ_ger)
                list_out.append(sentiment_class)
            print("Number of comments: " + str(len(list_summ_ger)))
            print("List of sum eng dic:" + str(list_summ_eng))
            print("List of sum ger dic:" + str(list_summ_ger))
            print("List of should outcome:" + str(list_out))

            # Calculate accuracy and confusion matrix
            y_ger = []
            y_eng = []
            y_true = []

            # Two classes
            if classes_num is 2:
                for y in list_summ_ger:
                    if y >= 0:
                        y_ger.append(1)
                    else:
                        y_ger.append(-1)
                for y in list_summ_eng:
                    if y >= 0:
                        y_eng.append(1)
                    else:
                        y_eng.append(-1)
                for y in list_out:
                    if y == 'POSITIVE':
                        y_true.append(1)
                    else:
                        y_true.append(-1)

            # Three classes

            #TODO find propper boundary, these are just random estimates for now
            boundary_eng = 3.0
            boundary_ger = 1.0

            if classes_num is 3:
                for y in list_summ_ger:
                    if y >= boundary_ger:
                        y_ger.append(1)
                    elif y > (-1)*boundary_ger:
                        y_ger.append(0)
                    else:
                        y_ger.append(-1)
                for y in list_summ_eng:
                    if y >= boundary_eng:
                        y_eng.append(1)
                    elif y > (-1) * boundary_eng:
                        y_eng.append(0)
                    else:
                        y_eng.append(-1)
                for y in list_out:
                    if y == 'POSITIVE':
                        y_true.append(1)
                    elif y == 'NEUTRAL':
                        y_true.append(0)
                    else:
                        y_true.append(-1)

            cm1 = plotting.calculate_normalized_confusion_matrix(y_true, y_eng, classes_num, title="Eng leksikon")
            plotting.show_confusion_matrix()
            print(accuracy_score(y_true, y_eng))
            cm2 = plotting.calculate_normalized_confusion_matrix(y_true, y_ger, classes_num, title="Ger leksikon")
            plotting.show_confusion_matrix()
            print(accuracy_score(y_true, y_ger))


            print("Finished")
        else:
            print("Tokenization of comment has not been done.")
    elif user_action is "2":
        print(naive_bayes(classes_num))
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
        with open("../movie_dataset/stemmed_dict_" + str(classes_num) + ".json", "w", encoding='utf-8') as f:
            json.dump(data_processed, f, ensure_ascii=False)
        data_set_json = data_processed
        print("Process finished!")
    elif user_action is "4":
        class_flag = 1
        class_temp = 1
        while class_flag is 1:
            print("Choose number of classes (2 or 3): ")
            print("--------------------------")
            class_temp = input("Action: ")
            try:
                class_temp = int(class_temp)
                if class_temp is 2 or class_temp is 3:
                    class_flag = 0
            except Exception as ex:
                pass
        classes_num = class_temp
        # Load data set from json
        data_set_json = None
        if os.path.isfile("../movie_dataset/stemmed_dict_" + str(classes_num) + ".json"):
            with open("../movie_dataset/stemmed_dict_" + str(classes_num) + ".json", encoding="utf8") as f:
                data_set_json = json.load(f)
        else:
            data_set_json = None

        # Load data set
        data_set = pd.read_csv("../movie_dataset/SerbMR-" + str(classes_num) + "C.csv")
    elif user_action is "5":
        results = keras_adaline(data_set_json, bias=False)
        print(results)
        print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    elif user_action is "6":
        results = keras_adaline(data_set_json, bias=True)
        print(results)
        print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    elif user_action is "7":
        work_flag = 0
