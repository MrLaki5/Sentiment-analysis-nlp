import pandas as pd
from stemmer import stemmer
from eng_dict import build_english
from ger_dict import build_german
import sentiment_logic
import json
from src.ml_algorithms import naive_bayes
from src.ml_algorithms import SVM
from src.ml_algorithms import log_reg
import os.path
import comment_process_pool
import plotting
import logging
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from neural_nets import keras_adaline, keras_1_layer_perceptron, keras_mlp_loop_all, keras_mlp_prepare_data

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
engDict, engDictPreProc = build_english()
engDictStemmed = stemmer.stem_dictionary(engDict)
engDictPreProcStemmed = stemmer.stem_dictionary(engDictPreProc)

# German
gerDict, gerDictPreProc = build_german()
gerDictStemmed = stemmer.stem_dictionary(gerDict)
gerDictPreProcStemmed = stemmer.stem_dictionary(gerDictPreProc)


# Load data set from json
data_set_json = None
if os.path.isfile("../movie_dataset/stemmed_dict_2.json"):
    with open("../movie_dataset/stemmed_dict_2.json", encoding="utf8") as f:
        data_set_json = json.load(f)

# Load matrix mlp from json
mlp_patrix_json = None
if os.path.isfile("../movie_dataset/mlp_matrix_2.json"):
    with open("../movie_dataset/mlp_matrix_2.json", encoding="utf8") as f:
        mlp_patrix_json = json.load(f)

# Load data set
data_set = pd.read_csv("../movie_dataset/SerbMR-2C.csv")

# Menu
work_flag = 1
classes_num = 2
while work_flag == 1:
    print("--------------------------")
    print("Choose action:")
    print("--------------------------")
    print("1. Lexicon sum (no ML)")
    print("2. Bayes-naive")
    print("3. SVM")
    print("4. Logistic Regression")
    print("5. Do tokenization of comments")
    print("6. Choose number of classes, current: " + str(classes_num))
    print("7. Adaline without bias")
    print("8. Adaline with bias")
    print("9. One layer perceptron")
    print("10. MLP")
    print("11. Pre process MLP matrix")
    print("12. Exit")
    print("--------------------------")
    user_action = input("Action: ")
    if user_action == "1":
        if data_set_json is not None:
            list_summ_ger = []
            list_summ_eng = []
            list_out = []
            # Should use preprocessed dict
            preproc_flag = 1
            preproc_num = 2
            while preproc_flag == 1:
                print("--------------------------")
                print("Should lexicons be preprocessed:")
                print("--------------------------")
                print("1. Yes")
                print("2. No")
                print("--------------------------")
                preproc_num = input("Choose: ")
                try:
                    preproc_num = int(preproc_num)
                    if preproc_num is 1 or preproc_num is 2:
                        preproc_flag = 0
                except Exception as ex:
                    pass
            if preproc_num is 1:
                eng_dict_curr = engDictPreProcStemmed
                ger_dict_curr = gerDictPreProcStemmed
            else:
                eng_dict_curr = engDictStemmed
                ger_dict_curr = gerDictStemmed
            # Should use negation
            negation_flag = 1
            negation_num = 0
            negation = False
            while negation_flag == 1:
                print("--------------------------")
                print("Should use negation:")
                print("--------------------------")
                print("1. Yes")
                print("2. No")
                print("--------------------------")
                negation_num = input("Choose: ")
                try:
                    negation_num = int(negation_num)
                    if negation_num is 1 or negation_num is 2:
                        negation_flag = 0
                except Exception as ex:
                    pass
            if negation_num is 1:
                negation = True
            else:
                negation = False
            # Levenshtein's distance
            leven_flag = 1
            leven_num = 5
            while leven_flag == 1:
                print("--------------------------")
                print("Choose Levenshtein's distance (optimal 5):")
                print("--------------------------")
                leven_num = input("Choose: ")
                try:
                    leven_num = int(leven_num)
                    if leven_num >= 0:
                        leven_flag = 0
                except Exception as ex:
                    pass
            for data in data_set_json:
                sentiment_class = data['class_att']
                tokens_original = data['tokens_original']
                tokens_stemmed = data['tokens_stemmed']
                summ_eng = sentiment_logic.comment_weight_calculation(eng_dict_curr, "English", tokens_original, tokens_stemmed, leven_num, modification_use=negation, amplification_use=False)
                summ_ger = sentiment_logic.comment_weight_calculation(ger_dict_curr, "German", tokens_original, tokens_stemmed, leven_num, modification_use=negation, amplification_use=False)

                list_summ_eng.append(summ_eng)
                list_summ_ger.append(summ_ger)
                list_out.append(sentiment_class)

            scaler = StandardScaler()
            summ_eng = np.array(list_summ_eng).reshape(-1, 1)
            summ_ger = np.array(list_summ_ger).reshape(-1, 1)
            scaled_data_eng = scaler.fit_transform(summ_eng)
            scaled_data_ger = scaler.fit_transform(summ_ger)
            scaled_data_eng = scaled_data_eng.reshape(1, -1)
            scaled_data_ger = scaled_data_ger.reshape(1, -1)
            print("Number of comments: " + str(len(list_summ_ger)))
            print("List of sum eng dic:" + str(scaled_data_eng))
            print("List of sum ger dic:" + str(scaled_data_ger))
            print("List of should outcome:" + str(list_out))
            # Sum weights
            summ = []
            for ger_sum, eng_sum in zip(scaled_data_eng[0], scaled_data_ger[0]):
                summ.append(ger_sum + eng_sum)

            # Calculate accuracy and confusion matrix
            y_both = []
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
                for y in summ:
                    if y >= 0:
                        y_both.append(1)
                    else:
                        y_both.append(-1)
                for y in list_out:
                    if y == 'POSITIVE':
                        y_true.append(1)
                    else:
                        y_true.append(-1)

            # Three classes
            boundary_eng = 4.6
            boundary_ger_left = 0.38
            boundary_ger_right = 0.45
            boundary_both = 0.595

            if classes_num is 3:
                for y in list_summ_ger:
                    if y >= boundary_ger_right:
                        y_ger.append(1)
                    elif y > (-1)*boundary_ger_left:
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
                for y in summ:
                    if y >= boundary_both:
                        y_both.append(1)
                    elif y > (-1) * boundary_both:
                        y_both.append(0)
                    else:
                        y_both.append(-1)
                for y in list_out:
                    if y == 'POSITIVE':
                        y_true.append(1)
                    elif y == 'NEUTRAL':
                        y_true.append(0)
                    else:
                        y_true.append(-1)

            cm1 = plotting.calculate_normalized_confusion_matrix(y_true, y_eng, classes_num, title="Eng lexicon")
            plotting.show_confusion_matrix()
            print(accuracy_score(y_true, y_eng))
            cm2 = plotting.calculate_normalized_confusion_matrix(y_true, y_ger, classes_num, title="Ger lexicon")
            plotting.show_confusion_matrix()
            print(accuracy_score(y_true, y_ger))
            cm3 = plotting.calculate_normalized_confusion_matrix(y_true, y_both, classes_num, title="Both lexicons")
            plotting.show_confusion_matrix()
            print(accuracy_score(y_true, y_both))
            print("Finished")
        else:
            print("Tokenization of comment has not been done.")
    elif user_action == "2":
        print(naive_bayes(classes_num))
    elif user_action == "3":
        print(SVM(classes_num))
    elif user_action == "4":
        print(log_reg(classes_num))
    elif user_action == "5":
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
    elif user_action == "6":
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
    elif user_action == "7":
        if classes_num != 2:
            print("Adeline works only with 2 classes")
            continue
        results = keras_adaline(data_set_json, bias=False)
        print(results)
        print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    elif user_action == "8":
        if classes_num != 2:
            print("Adeline works only with 2 classes")
            continue
        results = keras_adaline(data_set_json, bias=True)
        print(results)
        print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    elif user_action == "9":
        results = keras_1_layer_perceptron(data_set_json, classes_num)
        print(results)
        print("Accuracy: %.2f%% (%.2f%%)" % (results.mean(), results.std()))
    elif user_action == "10":
        reduction_flag = 1
        reduction_num = 0
        reduction = "PCA"
        while reduction_flag == 1:
            print("--------------------------")
            print("Choose feature vector dimensionality reduction algorithm:")
            print("--------------------------")
            print("1. PCA")
            print("2. LSA - TruncatedSVD")
            print("--------------------------")
            reduction_num = input("Choose: ")
            try:
                reduction_num = int(reduction_num)
                if reduction_num is 1 or reduction_num is 2:
                    reduction_flag = 0
            except Exception as ex:
                pass
        if reduction_num is 1:
            reduction = "PCA"
        else:
            reduction = "TruncatedSVD"

        order_flag = 1
        order_num = 0
        order = "reduce_first"
        while order_flag == 1:
            print("--------------------------")
            print("Choose order of applying concatenation and reduction:")
            print("--------------------------")
            print("1. Reduce, then concatenate")
            print("2. Concate, then reduce")
            print("--------------------------")
            order_num = input("Choose: ")
            try:
                order_num = int(order_num)
                if order_num is 1 or order_num is 2:
                    order_flag = 0
            except Exception as ex:
                pass
        if order_num is 1:
            order = "reduce_first"
        else:
            order = "reduce_last"
        # TODO add mlp function
        keras_mlp_loop_all()
        # results = keras_mlp(data_set_json, classes_num, 5, reduction, order)
        # print(results)
        # print("Accuracy: %.2f%% (%.2f%%)" % (results.mean(), results.std()))
    elif user_action == "11":
        # mlp_patrix_json =
        keras_mlp_prepare_data(data_set_json, classes_num, 5)
        # with open("../movie_dataset/mlp_matrix_" + str(classes_num) + ".json", "w", encoding='utf-8') as f:
        #     json.dump(mlp_patrix_json, f, ensure_ascii=False)
    elif user_action == "12":
        work_flag = 0
