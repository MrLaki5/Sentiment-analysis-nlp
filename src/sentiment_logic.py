import levenshtein
import logging

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)


# Function for calculating sum of weights of comment from specific dictionary
def comment_weight_calculation(dict_curr, dict_name, t_original, t_stemmed, distance_filter, modification_use=False, amplification_use=False):
    summ = 0
    negation_flag = False # TODO check if this works properly
    br_poz = 0
    br_neg = 0
    for token, stemm_token in zip(t_original, t_stemmed):
        # TODO do something with this kind of words
        if modification_use and token == "ne":
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
    logging.debug("Dictionary: " + dict_name + ", comment total: " + str(br_poz) + " positives, " + str(br_neg) + " negatives found")
    logging.debug("Dictionary: " + dict_name + ", words not found: " + str(set(t_original)))
    return summ
