pathPositive = "../lexicons/English/translations/positive-prevedeno-sredjeno.txt"
pathNegative = "../lexicons/English/translations/negative-prevedeno-sredjeno.txt"


def construct_eng_lexicon(positives, negatives):
    eng_dict = {}
    eng_dict_preprocessed = {}
    with open(positives, encoding='utf-8') as f1:
        for line in f1:
            word = line[:line.find("\n")]
            eng_dict[word] = 1
            eng_dict_preprocessed[word] = 1

    with open(negatives, encoding='utf-8') as f2:
        cnt = 0
        for line in f2:
            word = line[:line.find("\n")]
            if word in eng_dict:
                eng_dict[word] = 0
            else:
                eng_dict[word] = -1
            cnt += 1
            if cnt % 2 is 0:
                continue
            if word in eng_dict_preprocessed:
                eng_dict_preprocessed[word] = 0
            else:
                eng_dict_preprocessed[word] = -1
    return eng_dict, eng_dict_preprocessed


def build_english():
    return construct_eng_lexicon(pathPositive, pathNegative)
