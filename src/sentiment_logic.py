import levenshtein
import logging
from stemmer import stemmer
from eng_dict import build_english
from ger_dict import build_german
from comment_process_pool import zip_comment

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)


# Function for calculating sum of weights of comment from specific dictionary
def comment_weight_calculation(dict_curr, dict_name, t_original, t_stemmed, distance_filter, modification_use=False, amplification_use=False):
    summ = 0
    negation_flag = False # TODO check if this works properly
    br_poz = 0
    br_neg = 0
    not_found_words = []
    for token, stemm_token in zip(t_original, t_stemmed):
        # TODO do something with this kind of words
        if token is "ne":
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
    dict = {}
    stemmed_dict = {}
    dict_name = ""
    if language is "eng":
        dict = build_english()
        stemmed_dict = stemmer.stem_dictionary(dict)
        dict_name = "English"
    else:
        dict = build_german()
        stemmed_dict = stemmer.stem_dictionary(dict)
        dict_name = "German"
    tokens_stemmed, tokens_original = zip_comment(comment, "cwt.txt")
    comment_weight_calculation(stemmed_dict, dict_name, tokens_original, tokens_stemmed, 5, modification_use=False,
                               amplification_use=False)

