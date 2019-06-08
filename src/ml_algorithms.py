import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import KFold
import tokenizer
import plotting

def training(algorithm, class_num=2):
    if class_num == 2:
        train_data = pd.read_csv('../movie_dataset/SerbMR-2C.csv')
    else:
        train_data = pd.read_csv('../movie_dataset/SerbMR-3C.csv')
    #train_X, val_X, train_y, val_y = train_test_split(train_data['Text'], train_data['class-att'], test_size=0.2)

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

        # naive_bayes = MultinomialNB()
        algorithm.fit(train_X_tfidf, train_y)

        y_pred = algorithm.predict(val_X_tfidf)

        # if class_num == 2:
        #     plotting.calculate_normalized_confusion_matrix(val_y, y_pred, 2)
        # else:
        #     plotting.calculate_normalized_confusion_matrix(val_y, y_pred, 3)
        # plotting.show_confusion_matrix()

        accuracy += accuracy_score(val_y, y_pred)

    return accuracy/splits

def SVM(class_num=2):
    svm = SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-3, random_state=42,
                        max_iter=5, tol=None)
    return training(svm, class_num)

def naive_bayes(class_num=2):
    return training(MultinomialNB(), class_num)


print(SVM(2))
