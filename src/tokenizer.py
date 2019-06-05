import nltk
import re
import string

# Update tokenizer library
nltk.download('punkt')


# Tokenize text to words
def text_to_tokens(text):
    # Tokenize
    tokens = nltk.word_tokenize(text)
    # Remove punctuation tokens
    punkt = string.punctuation
    punkt += '”'
    punkt += '“'
    tokens = [x.lower() for x in tokens if not re.fullmatch('[' + punkt + ']+', x)]
    return tokens
