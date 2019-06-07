import pandas as pd
from stemmer import stemmer
import tokenizer
from eng_dict import buildEnglish
from ger_dict import build_german
import levenshtein

from src.naive_bayes import naive_bayes


def prepare_for_stemming(prep_text):
    prep_text = prep_text.replace("č", "cx")
    prep_text = prep_text.replace("ć", "cy")
    prep_text = prep_text.replace("dž", "dx")
    prep_text = prep_text.replace("đ", "dy")
    prep_text = prep_text.replace("ž", "zx")
    prep_text = prep_text.replace("š", "sx")
    prep_text = prep_text.replace("nj", "ny")
    prep_text = prep_text.replace("lj", "ly")
    return prep_text


def call_stemmer(stemm_text):
    # Stemm data set
    with open("./stemmer/temp_in.txt", "w", encoding="utf8") as f:
        f.write(stemm_text)

    stemmer.stemm(stemmer.STEM_OPTION_KESELJ_SIPKA_GREEDY, "./stemmer/temp_in.txt", "./stemmer/temp_out.txt")

    # Load stemmed data set
    with open("./stemmer/temp_out.txt", encoding="utf8") as f:
        stemm_text = f.read()

    # Return the result of stemming
    return stemm_text


def stem_dictionary(st_dict):
    temp = []
    originals = []
    stemmed_dictionary = {}

    for word in st_dict:
        originals.append(word)
        temp.append(prepare_for_stemming(word)+","+str(st_dict[word]))

    with open("./stemmer/temp_in.txt", "w", encoding="utf8") as f:
        for item in temp:
            f.write(item)
            f.write("\n")

    stemmer.stemm(stemmer.STEM_OPTION_KESELJ_SIPKA_GREEDY, "./stemmer/temp_in.txt", "./stemmer/temp_dict_out.txt")

    with open("./stemmer/temp_dict_out.txt", encoding="utf8") as f1:
        line_counter = 0
        for line in f1:
            items = line.split(",")
            stem = items[0]
            sentiment = items[1]
            sentiment = sentiment[:sentiment.find("\n")]
            sentiment = float(sentiment)
            if stem not in stemmed_dictionary:
                stemmed_dictionary[stem] = {}
            stemmed_dictionary[stem][originals[line_counter]] = sentiment
            line_counter += 1

    return stemmed_dictionary


def zip_comment(raw_text):
    # Generate tokens for data set
    tokens_original = tokenizer.text_to_tokens(raw_text)
    token_text = ""
    for i in tokens_original:
        token_text = token_text + i + "\n"
    token_text = call_stemmer(token_text)
    tokens_stemmed = token_text.split("\n")
    cnt = 0
    while cnt < len(tokens_original):
        if tokens_stemmed[cnt] == "":
            del tokens_stemmed[cnt]
            del tokens_original[cnt]
        else:
            cnt += 1

    zipped = zip(tokens_stemmed, tokens_original)
    return zipped


def comment_weight_calculation(dict_curr, tokens, distance_filter):
    summ = 0
    for token, stemm_token in tokens:
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
engDictStemmed = stem_dictionary(engDict)
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
gerDictStemmed = stem_dictionary(gerDict)
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

list_summ_ger = []
list_summ_eng = []
list_out = []

# Load data set
data_set = pd.read_csv("../movie_dataset/SerbMR-2C.csv")
# Iterate through data set
for index, row in data_set.iterrows():
    raw_text = row['Text']
    sentiment_class = row['class-att']
    zipped_tokens = zip_comment(raw_text)
    summ_eng = comment_weight_calculation(engDictStemmed, zipped_tokens, 80)
    summ_ger = comment_weight_calculation(gerDictStemmed,  zipped_tokens, 80)
    list_summ_eng.append(summ_eng)
    list_summ_ger.append(summ_ger)
    list_out.append(sentiment_class)

# Naive Bayes
print(naive_bayes())

print("Finished")