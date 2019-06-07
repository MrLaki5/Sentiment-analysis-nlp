import subprocess

STEM_OPTION_KESELJ_SIPKA_GREEDY = 1
STEM_OPTION_KESELJ_SIPKA_OPTIMAL = 2
STEM_OPTION_MILOSEVIC = 3
STEM_OPTION_LJUBESIC_PANDZIC = 4


# Use Serbian stemmers to stemm words from src_file
def stemm(stem_option, src_file_path, out_file_path):
    subprocess.call(['java', '-jar', './stemmer/SCStemmers.jar', str(stem_option), src_file_path, out_file_path])


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


def call_stemmer(stemm_text, file_name = ""):
    # Stemm data set
    with open("./stemmer/" + file_name + "_temp_in.txt", "w", encoding="utf8") as f:
        f.write(stemm_text)

    stemm(STEM_OPTION_KESELJ_SIPKA_GREEDY, "./stemmer/" +
          file_name + "_temp_in.txt", "./stemmer/" + file_name + "_temp_out.txt")

    # Load stemmed data set
    with open("./stemmer/" + file_name + "_temp_out.txt", encoding="utf8") as f:
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

    stemm(STEM_OPTION_KESELJ_SIPKA_GREEDY, "./stemmer/temp_in.txt", "./stemmer/temp_dict_out.txt")

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
