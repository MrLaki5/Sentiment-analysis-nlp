import levenshtein
import logging
from stemmer import stemmer
from eng_dict import build_english
from ger_dict import build_german
from comment_process_pool import zip_comment

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)


negation_words = ["ne"]


# Function for calculating sum of weights of comment from specific dictionary
def comment_weight_calculation(dict_curr, dict_name, t_original, t_stemmed, distance_filter, modification_use=False, amplification_use=False):
    summ = 0
    negation_flag = False
    br_poz = 0
    br_neg = 0
    not_found_words = []
    for token, stemm_token in zip(t_original, t_stemmed):
        if token in negation_words:
            if modification_use:
                negation_flag = True
            continue
        if stemm_token in dict_curr:
            min_distance = -1
            temp_weight = 0
            temp_word = ""
            temp_stem = ""
            temp_key = ""
            for key in dict_curr[stemm_token].keys():
                curr_distance = levenshtein.iterative_levenshtein(key, token)
                if curr_distance < min_distance or min_distance < 0:
                    min_distance = curr_distance
                    temp_weight = dict_curr[stemm_token][key]
                    temp_word = token
                    temp_stem = stemm_token
                    temp_key = key
            if min_distance <= distance_filter:
                if negation_flag:
                    negation_flag = False
                    temp_weight *= -1
                if temp_weight < 0:
                    br_neg += 1
                else:
                    br_poz += 1
                summ += temp_weight
                logging.debug("Dictionary: " + dict_name + ", comment word: " + temp_word + ", stem: " + temp_stem + ", paired word: " + temp_key + ", Weight: " + str(temp_weight) + ", distance: " + str(min_distance))
        else:
            negation_flag = False
            not_found_words.append(token)
    logging.debug("Dictionary: " + dict_name + ", comment total: " + str(br_poz) + " positives, " + str(br_neg) + " negatives found")
    logging.debug("Dictionary: " + dict_name + ", words not found: " + str(set(not_found_words)))
    return summ


def comment_weight_quick(comment, language):
    if language is "eng":
        dict_temp = build_english()
        stemmed_dict = stemmer.stem_dictionary(dict_temp)
        dict_name = "English"
    else:
        dict_temp = build_german()
        stemmed_dict = stemmer.stem_dictionary(dict_temp)
        dict_name = "German"
    tokens_stemmed, tokens_original = zip_comment(comment, "cwt.txt")
    comment_weight_calculation(stemmed_dict, dict_name, tokens_original, tokens_stemmed, 5, modification_use=False,
                               amplification_use=False)


# Function that returns vector of weight summ for every word in dictionary regarding given comment
def comment_weight_vector(dict_curr, t_original, t_stemmed, distance_filter):
    ret_vector = []
    name_vector = []
    for key, value in dict_curr.items():
        for key_inner, val_inner in value.items():
            name_vector.append(key + key_inner)
            ret_vector.append(0)
    for token, stemm_token in zip(t_original, t_stemmed):
        if stemm_token in dict_curr:
            min_distance = -1
            temp_weight = 0
            temp_word = ""
            for key in dict_curr[stemm_token].keys():
                curr_distance = levenshtein.iterative_levenshtein(key, token)
                if curr_distance < min_distance or min_distance < 0:
                    min_distance = curr_distance
                    temp_weight = dict_curr[stemm_token][key]
                    temp_word = key
            if min_distance <= distance_filter:
                ret_vector[name_vector.index(stemm_token + temp_word)] += temp_weight
    return ret_vector


