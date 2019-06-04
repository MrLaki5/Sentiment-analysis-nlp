import pandas as pd
from stemmer import stemmer
import tokenizer


# Load data set
data_set = pd.read_csv("../movie_dataset/SerbMR-2C.csv")

# Stemm data set
with open("./stemmer/temp_in.txt", "w") as f:
    f.write(data_set["Text"][0])

stemmer.stemm(stemmer.STEM_OPTION_KESELJ_SIPKA_GREEDY, "./stemmer/temp_in.txt", "./stemmer/temp_out.txt")

# Load stemmed data set
with open("./stemmer/temp_out.txt") as f:
    text = f.read()

# Tokenize data set
tokens = tokenizer.text_to_tokens(text)
print(tokens)
