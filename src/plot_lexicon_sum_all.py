from stemmer import stemmer
from eng_dict import build_english
from ger_dict import build_german
import sentiment_logic
import json
import os.path
import plotting
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


classes_num = 3

data_set_json = None
if os.path.isfile("../movie_dataset/stemmed_dict_" + str(classes_num) + ".json"):
    with open("../movie_dataset/stemmed_dict_" + str(classes_num) + ".json", encoding="utf8") as f:
        data_set_json = json.load(f)

# English
engDict, engDictPreProc = build_english()
engDictStemmed = stemmer.stem_dictionary(engDict)
engDictPreProcStemmed = stemmer.stem_dictionary(engDictPreProc)

# German
gerDict, gerDictPreProc = build_german()
gerDictStemmed = stemmer.stem_dictionary(gerDict)
gerDictPreProcStemmed = stemmer.stem_dictionary(gerDictPreProc)

# All parameters
dicts_ger_all = [engDictStemmed, engDictPreProcStemmed]
dicts_eng_all = [gerDictStemmed, gerDictPreProcStemmed]
negation_all = [False, True]
levenshtein_all = [0, 1, 2, 3, 4, 5, 6]


preprocessed_name = "Not preprocessed dictionary"
# Plot for all parameters
for eng_dict_curr, ger_dict_curr in zip(dicts_eng_all, dicts_ger_all):

    for negation in negation_all:
        for leven_num in levenshtein_all:
            list_summ_eng = []
            list_summ_ger = []
            list_out = []
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
            y_true = []

            # Two classes
            if classes_num is 2:
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
            boundary_both = 0.595

            if classes_num is 3:
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

            cm3 = plotting.calculate_normalized_confusion_matrix(y_true, y_both, classes_num, title=preprocessed_name + ", negation: " + str(negation) + ", Levenshtein's distance: " + str(leven_num))
            plotting.show_confusion_matrix()
            print(accuracy_score(y_true, y_both))
    preprocessed_name = "Preprocessed dictionary"


