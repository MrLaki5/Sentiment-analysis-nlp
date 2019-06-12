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
from sklearn.model_selection import KFold
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


def training(algorithm, class_num=2):
    if class_num == 2:
        train_data = pd.read_csv('../movie_dataset/SerbMR-2C.csv')
    else:
        train_data = pd.read_csv('../movie_dataset/SerbMR-3C.csv')
        # train_data = pd.read_csv('E:/Faks/M/OPJ/Projekat/bbc-text.csv')

    splits = 5
    kf = KFold(n_splits=splits, shuffle=True)
    accuracy = 0

    # kf = KFold(n_splits=splits, random_state=12)
    for train_index, test_index in kf.split(train_data):
        train_X = [train_data['Text'][i] for i in train_index]
        train_y = [train_data['class-att'][i] for i in train_index]
        val_X = [train_data['Text'][i] for i in test_index]
        val_y = [train_data['class-att'][i] for i in test_index]
        # train_X = [train_data['text'][i] for i in train_index]
        # train_y = [train_data['category'][i] for i in train_index]
        # val_X = [train_data['text'][i] for i in test_index]
        # val_y = [train_data['category'][i] for i in test_index]

        # parameters = {
        #     'vect__ngram_range': [(1, 1), (1, 2)],
        #     'tfidf__use_idf': (True, False),
        #     'clf__alpha': (1e-2, 1e-3)
        # }

        text_clf = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenizer.text_to_tokens, min_df=3, ngram_range=(1, 2))),
            ('tfidf', TfidfTransformer()),
            # ('tfidf', TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), tokenizer=tokenizer.text_to_tokens)),
            ('clf', algorithm),
        ])

        # gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

        text_clf.fit(train_X, train_y)
        y_pred = text_clf.predict(val_X)

        if class_num == 2:
            plotting.calculate_normalized_confusion_matrix(val_y, y_pred, 2)
        else:
            plotting.calculate_normalized_confusion_matrix(val_y, y_pred, 3)
        plotting.show_confusion_matrix()

        # print(gs_clf.best_score_)
        # for param_name in sorted(parameters.keys()):
        #     print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

        accuracy += accuracy_score(val_y, y_pred)

    return accuracy/splits

def SVM(class_num=2):
    svm = SGDClassifier(loss='hinge', penalty='l2',
                        alpha=1e-3, random_state=42,
                        max_iter=1000)

    # degrees = [0, 1, 2, 3, 4, 5, 6]
    # for degree in degrees:
    #     svc = SVC(kernel='poly', degree=degree)
    #     print(str(degree) + ':' + str(training(svc, class_num)))
    svc = LinearSVC()
    return training(svm, class_num)

def naive_bayes(class_num=2):
    return training(MultinomialNB(), class_num)

def log_reg(class_num=2):
    # tolerance = [1e-5, 1e-4, 1e-3, 1e-1]
    # for t in tolerance:
    #     logreg = LogisticRegression(C=50, tol=t, dual=True)
    #     print((str(t) + ':' + str(training(logreg, class_num))))
    return training(LogisticRegression(C=50, dual=True), class_num)

# print(naive_bayes(2))
# print(SVM(3))
# print(log_reg(3))
