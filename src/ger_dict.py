LEX_PATH = "../lexicons/German/translation/sent_ger_both.txt"


def construct_ger_lexicon(lex_path):
    ger_dict = {}
    with open(lex_path, encoding='utf-8') as f1:
        for line in f1:
            line_split = line.split(",")
            word = line_split[0]
            number = float(line_split[1].split("\n")[0])
            ger_dict[word] = number
    return ger_dict


def build_german():
    return construct_ger_lexicon(LEX_PATH)

