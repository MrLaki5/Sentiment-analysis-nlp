import pandas as pd
from stemmer import stemmer
import tokenizer

from lexicons.English.translations.tools.eng_dict import buildEnglish


def prepareForStemming(text):
    text = text.replace("č", "cx")
    text = text.replace("ć", "cy")
    text = text.replace("dž", "dx")
    text = text.replace("đ", "dy")
    text = text.replace("ž", "zx")
    text = text.replace("š", "sx")
    text = text.replace("nj", "ny")
    text = text.replace("lj", "ly")
    return text

def callStemmer(text, output_path):
    # Stemm data set
    with open("./stemmer/temp_in.txt", "w", encoding="utf8") as f:
        f.write(text)

    stemmer.stemm(stemmer.STEM_OPTION_KESELJ_SIPKA_GREEDY, "./stemmer/temp_in.txt", output_path)

    # Load stemmed data set
    with open("./stemmer/temp_out.txt", encoding="utf8") as f:
        text = f.read()

    # Return the result of stemming
    return text

def stemDictionary(dict):
    temp = []
    originals = []
    stemmedDictionary = {}
    for word in dict:
        originals.append(word)
        temp.append(prepareForStemming(word)+","+str(dict[word]))
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
            sentiment = int(sentiment)
            if stem not in stemmedDictionary:
                stemmedDictionary[stem] = {}
            stemmedDictionary[stem][originals[line_counter]] = sentiment

            line_counter+=1

    return stemmedDictionary

# START MAIN

# Load data set
data_set = pd.read_csv("../movie_dataset/SerbMR-2C.csv")
text = data_set['Text'][0]
text = prepareForStemming(text)

callStemmer(text, "./stemmer/temp_out.txt")

# Stemm data set
#with open("./stemmer/temp_in.txt", "w", encoding="utf8") as f:
#    f.write(text)
#
#stemmer.stemm(stemmer.STEM_OPTION_KESELJ_SIPKA_GREEDY, "./stemmer/temp_in.txt", "./stemmer/temp_out.txt")
#
## Load stemmed data set
#with open("./stemmer/temp_out.txt", encoding="utf8") as f:
#    text = f.read()

# Tokenize data set
tokens = tokenizer.text_to_tokens(text)
print(tokens)

engDict = buildEnglish()
engDictStemmed = stemDictionary(engDict)

print(engDictStemmed)
cnt = 0
for item in engDictStemmed:
    if len(engDictStemmed[item]) > 1:
        print(cnt, item, engDictStemmed[item])
        cnt += 1