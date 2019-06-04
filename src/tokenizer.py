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
    tokens = [x.lower() for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)]
    return tokens
