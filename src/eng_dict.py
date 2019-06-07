pathPositive = "../lexicons/English/translations/positive-prevedeno-sredjeno.txt"
pathNegative = "../lexicons/English/translations/negative-prevedeno-sredjeno.txt"


def constructEngLexicon(positives, negatives):
    engDict = {}
    with open(positives, encoding='utf-8') as f1:
        for line in f1:
            word = line[:line.find("\n")]
            engDict[word] = 1

    with open(negatives, encoding='utf-8') as f2:
        for line in f2:
            word = line[:line.find("\n")]
            if word in engDict:
                engDict[word] = 0
            else:
                engDict[word] = -1
    return engDict

def buildEnglish():
    return constructEngLexicon(pathPositive, pathNegative)

