import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import numpy as np
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import tokenizer
import plotting
from sklearn.pipeline import Pipeline

def getCountVector(class_num=2):
    if class_num == 2:
        train_data = pd.read_csv('../movie_dataset/SerbMR-2C.csv')
    else:
        train_data = pd.read_csv('../movie_dataset/SerbMR-3C.csv')
    count_vect = CountVectorizer(tokenizer=tokenizer.text_to_tokens)
    train_X_counts = count_vect.fit_transform(train_data['Text'])
    return count_vect

def getOccurNumberDictionary(text, count_vect):
    dict_counter = {}
    for index, word in enumerate(count_vect.get_feature_names()):
        dict_counter[word] = 0

    for word in tokenizer.text_to_tokens(text):
        if word in dict_counter:
            dict_counter[word] += 1
    return dict_counter

def training(parameters, alg, class_num=2):
    if class_num == 2:
        train_data = pd.read_csv('../movie_dataset/SerbMR-2C.csv')
    else:
        train_data = pd.read_csv('../movie_dataset/SerbMR-3C.csv')
        # train_data = pd.read_csv('E:/Faks/M/OPJ/Projekat/bbc-text.csv')

    train_data_X = train_data['Text']
    train_data_y = train_data['class-att']

    X_train, X_test, y_train, y_test = train_test_split(train_data_X, train_data_y, test_size=0.2, random_state=7, stratify=train_data_y)


    text_clf = Pipeline([
        # ('vect', CountVectorizer(tokenizer=tokenizer.text_to_tokens, min_df=3, ngram_range=(1, 2))),
        # ('tfidf', TfidfTransformer()),
        ('tfidf', TfidfVectorizer(min_df=3, ngram_range=(1, 2), tokenizer=tokenizer.text_to_tokens)),
        ('alg', alg),
    ])

    gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

    gs_clf = gs_clf.fit(X_train, y_train)       # it returns optimized classifier that we can use to predict
    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    print(gs_clf.cv_results_)


    y_pred = gs_clf.predict(X_test)

    if class_num == 2:
        plotting.calculate_normalized_confusion_matrix(y_test, y_pred, 2)
    else:
        plotting.calculate_normalized_confusion_matrix(y_test, y_pred, 3)
    plotting.show_confusion_matrix()

    return accuracy_score(y_test, y_pred)

def SVM(class_num=2):
    parameters = {
        'alg__penalty': ('l1', 'l2'),
        'alg__alpha': (1e-2, 1e-3, 1e-4, 1e-5)
    }

    svm = SGDClassifier(loss='hinge', random_state=42,
                        max_iter=1000)

    return training(parameters, svm, class_num)

def naive_bayes(class_num=2):
    parameters = {}
    return training(parameters, MultinomialNB(), class_num)

def log_reg(class_num=2):
    parameters = {
        'alg__penalty': ('l1', 'l2'),
        'alg__C': (0.1, 1, 10, 50, 100)
    }
    log = LogisticRegression()

    return training(parameters, log, class_num)

# print(naive_bayes(3))
# print(SVM(3))
# print(log_reg(3))
