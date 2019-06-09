LEX_PATH = "../lexicons/German/translation/sent_ger_both.txt"


def construct_ger_lexicon(lex_path):
    ger_dict = {}
    ger_dict_preprocessed = {}
    with open(lex_path, encoding='utf-8') as f1:
        cnt = 0
        for line in f1:
            line_split = line.split(",")
            word = line_split[0]
            number = float(line_split[1].split("\n")[0])
            ger_dict[word] = number
            ger_dict_preprocessed[word] = number
    min_weight, max_weight = find_min_and_max_weight(ger_dict)
    neg_value_multip = -1.0 / min_weight
    pos_value_multip = 1.0 / max_weight
    for key in ger_dict.keys():
        if ger_dict[key] < 0:
            ger_dict[key] *= neg_value_multip
            ger_dict_preprocessed[key] *= neg_value_multip * 0.46
        else:
            ger_dict[key] *= pos_value_multip
            ger_dict_preprocessed[key] *= pos_value_multip
    return ger_dict, ger_dict_preprocessed


def build_german():
    return construct_ger_lexicon(LEX_PATH)


def find_min_and_max_weight(lex):
    min_weight = 0
    max_weight = 0
    found_min = False
    found_max = False
    for key, value in lex.items():
        if value < min_weight or not found_min:
            min_weight = value
            found_min = True
        if value > max_weight or not found_max:
            max_weight = value
            found_max = True
    return min_weight, max_weight
