import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
import tokenizer
import plotting

def naive_bayes(class_num=2):
    if class_num == 2:
        train_data = pd.read_csv('../movie_dataset/SerbMR-2C.csv')
    else:
        train_data = pd.read_csv('../movie_dataset/SerbMR-3C.csv')

    splits = 5
    kf = KFold(n_splits=splits, shuffle=True)
    accuracy = 0

    # KFold(n_splits=2, random_state=None, shuffle=False)
    for train_index, test_index in kf.split(train_data):
        train_X = [train_data['Text'][i] for i in train_index]
        train_y = [train_data['class-att'][i] for i in train_index]
        val_X = [train_data['Text'][i] for i in test_index]
        val_y = [train_data['class-att'][i] for i in test_index]

        count_vect = CountVectorizer(tokenizer=tokenizer.text_to_tokens)
        train_X_counts = count_vect.fit_transform(train_X)
        val_X_counts = count_vect.transform(val_X)

        tfidf_transformer = TfidfTransformer()
        train_X_tfidf = tfidf_transformer.fit_transform(train_X_counts)
        val_X_tfidf = tfidf_transformer.transform(val_X_counts)

        naive_bayes = MultinomialNB()
        naive_bayes.fit(train_X_tfidf, train_y)

        y_pred = naive_bayes.predict(val_X_tfidf)

        if class_num == 2:
            plotting.calculate_normalized_confusion_matrix(val_y, y_pred, 2)
        else:
            plotting.calculate_normalized_confusion_matrix(val_y, y_pred, 3)
        plotting.show_confusion_matrix()

        accuracy += accuracy_score(val_y, y_pred)

    return accuracy/splits


#print(naive_bayes(3))
