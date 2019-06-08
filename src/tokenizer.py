import nltk
import re
import string

# Update tokenizer library
nltk.download('punkt')

stop_words = ["a", "ako", "ali", "bi", "bih", "bila", "bili", "bilo", "bio", "bismo", "biste", "biti", "da", "do",
              "duž", "ga","hoće","hoćemo","hoćete","hoćeš","hoću","i","iako","ih","ili","iz","ja","je","jedna",
              "jedne","jedno","jer","jesam","jesi","jesmo","jest","jeste","jesu","joj","još","ju","kada",
              "kako","kao","koja","koje","koji","kojima","koju","kroz","li", "ma", "me","mene","meni",
              "mi","mimo","moj","moja","moje","mu","na","nad","nakon","nam","nama","nas","naš","naša",
              "naše","našeg","nego","neka","neki","nekog","neku","nema","netko","neće","nećemo","nećete",
              "nećeš","neću","nešto","ni","nije","nikoga","nikoje","nikoju","nisam","nisi","nismo",
              "niste","nisu","njega","njegov","njegova","njegovo","njemu","njezin","njezina","njezino",
              "njih","njihov","njihova","njihovo","njim","njima","njoj","nju","no","o","od","odmah","on",
              "ona","oni","ono","ova","pa","pak","po","pod","pored","pre","prije","s","sa","sam","samo",
              "se","sebe","sebi","si","smo","ste","su","sve","svi","svog","svoj","svoja","svoje","svom",
              "ta","tada","taj","tako","te","tebe","tebi","ti","to","toj","tome","tu","tvoj","tvoja",
              "tvoje","u","uz","vam","vama","vas","vaš","vaša","vaše","već","vi","vrlo","za","zar","će",
              "ćemo","ćete","ćeš","ću","što"]


# Tokenize text to words
def tokenize(text, without_stop):
    # Tokenize
    tokens = nltk.word_tokenize(text)
    tokens = [x.lower() for x in tokens]
    # Remove punctuation tokens
    punkt = string.punctuation
    punkt += '”'
    punkt += '“'
    punkt += "’"
    if without_stop:
        tokens = [x for x in tokens if not re.fullmatch('[' + punkt + ']+', x) and x not in stop_words and len(x) > 1]
    else:
        tokens = [x for x in tokens if not re.fullmatch('[' + punkt + ']+', x) and len(x) > 1]
    return tokens

def text_to_tokens(text):
    return tokenize(text, True)

def text_to_tokens_with_stop(text):
    return tokenize(text, False)


