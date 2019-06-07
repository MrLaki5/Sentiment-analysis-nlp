import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import tokenizer
import plotting

def naive_bayes():
    train_data = pd.read_csv('../movie_dataset/SerbMR-2C.csv')
    train_X, val_X, train_y, val_y = train_test_split(train_data['Text'], train_data['class-att'], test_size=0.2)

    count_vect = CountVectorizer(tokenizer=tokenizer.text_to_tokens)
    train_X_counts = count_vect.fit_transform(train_X)
    val_X_counts = count_vect.transform(val_X)

    tfidf_transformer = TfidfTransformer()
    train_X_tfidf = tfidf_transformer.fit_transform(train_X_counts)
    val_X_tfidf = tfidf_transformer.transform(val_X_counts)

    naive_bayes = MultinomialNB()
    naive_bayes.fit(train_X_tfidf, train_y)

    y_pred = naive_bayes.predict(val_X_tfidf)

    plotting.calculate_confusion_matrix(val_y, y_pred, plotting.LABELS_TWO_CLASS)
    plotting.show_confusion_matrix()

    return accuracy_score(val_y, y_pred)

# print(naive_bayes())
