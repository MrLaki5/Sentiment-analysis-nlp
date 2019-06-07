import pandas as pd
from stemmer import stemmer
import tokenizer
from eng_dict import buildEnglish
from ger_dict import build_german

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


# START MAIN
# Load dictionaries
# English
engDict = buildEnglish()
engDictStemmed = stem_dictionary(engDict)
cnt = 0
for item in engDictStemmed:
    if len(engDictStemmed[item]) > 1:
        print(cnt, item, engDictStemmed[item])
        cnt += 1
# German
gerDict = build_german()
gerDictStemmed = stem_dictionary(gerDict)
cnt = 0
for item in gerDictStemmed:
    if len(gerDictStemmed[item]) > 1:
        print(cnt, item, gerDictStemmed[item])
        cnt += 1

# Load data set
data_set = pd.read_csv("../movie_dataset/SerbMR-2C.csv")
# Iterate through data set
for index, row in data_set.iterrows():
    print(row[0])
    text = row[0]
    text = prepare_for_stemming(text)
    text = call_stemmer(text)
    # Generate tokens for data set
    tokens_original = tokenizer.text_to_tokens(row[0])
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
    for i, j in zip(tokens_stemmed, tokens_original):
        print(i + " " + j)
    print("Value: " + row[1])

# Naive Bayes
print(naive_bayes())