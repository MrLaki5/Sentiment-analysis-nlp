import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from sklearn.model_selection import KFold, train_test_split
from sentiment_logic import comment_weight_calculation, comment_weight_vector
from eng_dict import build_english
from ger_dict import build_german
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from stemmer import stemmer
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
import plotting
import ml_algorithms
from sklearn.decomposition import TruncatedSVD, PCA
from timeit import default_timer as timer
import json

ENG_DIM = 3132
GER_DIM = 1784
BAG_DIM = 114921

def class_encode(class_str):
    if class_str == 'NEGATIVE':
        return -1
    elif class_str == 'POSITIVE':
        return 1
    else:
        return 0


def build_adaline_no_bias():
    inputA = keras.Input(shape=(2,))

    x = Dense(1, activation="linear")(inputA)
    x = Dense(1, activation="tanh")(x)
    model = Model(inputs=inputA, outputs=x)
    model.compile(loss='mean_squared_error', optimizer="sgd", metrics=['accuracy'])
    return model


def build_adaline_with_bias():
    inputA = keras.Input(shape=(3,))

    x = Dense(1, activation="linear")(inputA)
    x = Dense(1, activation="tanh")(x)
    model = Model(inputs=inputA, outputs=x)
    model.compile(loss='mean_squared_error', optimizer="sgd", metrics=['accuracy'])
    return model


# For 2 class classification only
def keras_adaline(data_set_json, bias=False):
    _, engDict = build_english() # swap the dict if needed
    engDictStemmed = stemmer.stem_dictionary(engDict)
    _, gerDict= build_german() # swap the dict if needed
    gerDictStemmed = stemmer.stem_dictionary(gerDict)

    if not bias:
        estimator = KerasClassifier(build_fn=build_adaline_no_bias, epochs=200, batch_size=5, verbose=0)
    else:
        estimator = KerasClassifier(build_fn=build_adaline_with_bias, epochs=200, batch_size=5, verbose=0)

    splits = 5
    seed = 7
    np.random.seed(seed)
    kf = KFold(n_splits=splits, shuffle=True, random_state=seed)

    x = []
    y = []
    for data in data_set_json:
        sentiment_class = data['class_att']
        tokens_original = data['tokens_original']
        tokens_stemmed = data['tokens_stemmed']
        summ_eng = comment_weight_calculation(engDictStemmed, "English", tokens_original,
                                                              tokens_stemmed, 5, modification_use=False,
                                                              amplification_use=False)
        summ_ger = comment_weight_calculation(gerDictStemmed, "German", tokens_original, tokens_stemmed,
                                                              5, modification_use=False, amplification_use=False)

        one_x = [summ_eng, summ_ger]
        if bias:
            one_x.append(1)
        x.append(one_x)
        y.append(class_encode(sentiment_class))

    x = np.array(x)
    y = np.array(y)
    results = cross_val_score(estimator, x, y, cv=kf)

    return results

def build_1_layer_perceptron(num_of_classes):
    inputA = keras.Input(shape=(3,))

    x = Dense(1, activation="linear")(inputA)
    x = Dense(num_of_classes, activation="sigmoid")(x)
    x = Dense(num_of_classes, activation="softmax")(x)

    model = Model(inputs=inputA, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer="sgd", metrics=['accuracy'])
    return model


def build_1L_2C_perceptron():
    return build_1_layer_perceptron(2)


def build_1L_3C_perceptron():
    return build_1_layer_perceptron(3)


def keras_1_layer_perceptron(data_set_json, classes_num):
    _, engDict = build_english()  # swap the dict if needed
    engDictStemmed = stemmer.stem_dictionary(engDict)
    _, gerDict = build_german()  # swap the dict if needed
    gerDictStemmed = stemmer.stem_dictionary(gerDict)

    if classes_num == 2:
        estimator = KerasClassifier(build_fn=build_1L_2C_perceptron, epochs=200, batch_size=5)
    else:
        estimator = KerasClassifier(build_fn=build_1L_3C_perceptron, epochs=200, batch_size=5)

    splits = 5
    seed = 7
    np.random.seed(seed)
    kf = KFold(n_splits=splits, shuffle=True, random_state=seed)

    x = []
    y = []
    for data in data_set_json:
        sentiment_class = data['class_att']
        tokens_original = data['tokens_original']
        tokens_stemmed = data['tokens_stemmed']
        summ_eng = comment_weight_calculation(engDictStemmed, "English", tokens_original,
                                              tokens_stemmed, 5, modification_use=False,
                                              amplification_use=False)
        summ_ger = comment_weight_calculation(gerDictStemmed, "German", tokens_original, tokens_stemmed,
                                              5, modification_use=False, amplification_use=False)

        one_x = [summ_eng, summ_ger]
        one_x.append(1)
        x.append(one_x)
        y.append(class_encode(sentiment_class))

    x = np.array(x)
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x = sc.fit_transform(x)
    x = sc.transform(x)

    y = np.array(y)
    # One-hot encoding
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    # convert integers to dummy variables (i.e. one hot encoded)
    y = np_utils.to_categorical(encoded_Y)

    if classes_num == 2:
        model = build_1L_2C_perceptron()
    else:
        model = build_1L_3C_perceptron()

    # Version with our cross-validation:
    cvscores = []
    cms = []
    cmdata = []
    for train, test in kf.split(x, y):

        # Fit the model
        model.fit(x[train], y[train], epochs=100, batch_size=10, verbose=0)
        # evaluate the model
        scores = model.evaluate(x[test], y[test], verbose=0)

        y_pred = model.predict(x[test])
        y_pred_categorical = []
        for row in y_pred:
            pred_class = np.argmax(row)
            y_pred_categorical.append(pred_class)
        y_pred = np.array(y_pred_categorical)

        y_test = y[test]
        y_test_categorical = []
        for row in y_test:
            pred_class = np.argmax(row)
            y_test_categorical.append(pred_class)
        y_test = np.array(y_test_categorical)

        cm = confusion_matrix(y_test, y_pred)

        cmdata.append([y_test, y_pred])
        cms.append(cm)

        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)

    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

    # results = cross_val_score(estimator, x, y, cv=kf)
    cnt = 1
    for cmpair in cmdata:
        plotting.calculate_normalized_confusion_matrix(cmpair[0], cmpair[1], classes_num, title="Fold "+str(cnt)+", accuracy: "+ str(cvscores[cnt-1]))
        cnt+=1
        plotting.show_confusion_matrix()

    return np.array(cvscores)

def build_deep_mlp():
    # define two sets of inputs
    inputEng = keras.Input(shape=(2,))
    inputGer = keras.Input(shape=(2,))
    inputBag = keras.Input(shape=(2,))


def keras_2_layer_perceptron():
    # define two sets of inputs
    inputA = keras.Input(shape=(2,))
    inputB = keras.Input(shape=(2,))

    # the first branch operates on the first input
    x = Dense(1, activation="relu")(inputA)
    x = Dense(4, activation="relu")(x)
    x = Model(inputs=inputA, outputs=x)

    # the second branch opreates on the second input
    y = Dense(64, activation="relu")(inputB)
    y = Dense(32, activation="relu")(y)
    y = Dense(4, activation="relu")(y)
    y = Model(inputs=inputB, outputs=y)

    # combine the output of the two branches
    combined = concatenate([x.output, y.output])

    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(2, activation="relu")(combined)
    z = Dense(1, activation="linear")(z)

    # our model will accept the inputs of the two branches and
    # then output a single value
    model = Model(inputs=[x.input, y.input], outputs=z)


def keras_mlp_prepare_data():
    pass


def keras_mlp(data_set_json, classes_num=2, levenshtein=5):
    print("Preparing lexicons")
    _, engDict = build_english()  # swap the dict if needed
    engDictStemmed = stemmer.stem_dictionary(engDict)
    _, gerDict = build_german()  # swap the dict if needed
    gerDictStemmed = stemmer.stem_dictionary(gerDict)
    print("Lexicons ready")
    print("Beginning to build feature vectors")
    count_vect = ml_algorithms.getCountVector(classes_num)

    x_eng = []
    x_ger = []
    x_bag = []
    x_composite = []
    y = []

    tmp = True # for Debugging only
    dbg = 0

    for data in data_set_json:
        sentiment_class = data['class_att']
        tokens_original = data['tokens_original']
        tokens_stemmed = data['tokens_stemmed']
        # Get features from english lexicon
        x_eng_row = comment_weight_vector(engDictStemmed, tokens_original, tokens_stemmed, levenshtein)

        # Get features from german lexicon
        x_ger_row = comment_weight_vector(gerDictStemmed, tokens_original, tokens_stemmed, levenshtein)

        # Get cumulative predictors from german and english lexicons
        summ_eng = comment_weight_calculation(engDictStemmed, "English", tokens_original,
                                              tokens_stemmed, 5, modification_use=False,
                                              amplification_use=False)
        summ_ger = comment_weight_calculation(gerDictStemmed, "German", tokens_original, tokens_stemmed,
                                              5, modification_use=False, amplification_use=False)

        # Get bag of words feature vector
        # TODO transform to tf-idf form!
        comment = ""
        for word in tokens_original:
            comment += " "+word
        dict = ml_algorithms.getOccurNumberDictionary(comment, count_vect)
        keys_list = list(dict.keys())
        keys_list.sort()
        x_bag_row = []
        for word in keys_list:
            x_bag_row.append(dict[word])

        x_eng.append(x_eng_row)
        x_ger.append(x_ger_row)
        x_bag.append(x_bag_row)
        x_composite.append(x_eng_row + x_ger_row + x_bag_row)

        y.append(class_encode(sentiment_class))

        # DEBUGGING
        # if dbg < 100:
        #    dbg+=1
        # else:
        #    break
        # if tmp:
        #    tmp = False
        #    print(len(x_eng_row))
        #    print(x_eng_row)
        #    print(len(x_ger_row))
        #    print(x_ger_row)
        #    print(len(x_bag_row))
        #    print(x_bag_row)

    # Create a collected feature matrix, dim is num_of_comments x (ENG_DIM+GER_DIM+BAG_DIM)
    x_composite_matrix = np.array(x_composite)
    # Split feature matrix into training (80%) and test (20%)
    # No kfolding because it would take too long to train, kfolding can be used on the training part (80%) to tune
    # the hyper parameters and pick the best model when the feature vector is already reduced to less dimensions
    x_train, x_test, y_train, y_test, x_eng_train, x_eng_test, x_ger_train, x_ger_test, x_bag_train, x_bag_test = train_test_split(x_composite_matrix, y, x_eng, x_ger, x_bag, test_size=0.2)

    reduction_all = ["TruncatedSVD", "PCA"]
    order_all = ["reduce_last", "reduce_first"]
    results = {}
    for reduction in reduction_all:
        for order in order_all:

            ndim = ENG_DIM+GER_DIM+BAG_DIM
            x_train_fit = np.array([])
            x_test_fit = np.array([])
            print("Current feature vector dimension: " + str(ndim))
            if order == "reduce_last":
                if reduction == "PCA":
                    # retain 95% variance
                    pca = PCA(0.95)
                    print("Attempting to fit PCA reduction with variance retain rate of 95%")
                    start = timer()
                    pca.fit(x_train)
                    end = timer()
                    print("Successful fitting of PCA reduction to "+str(pca.n_components_)+" components")
                    print("Fitting took "+str(end - start)+" seconds")
                    ndim = pca.n_components_

                    print("Attempting to reduce the training set and test set")
                    start = timer()
                    x_train_fit = pca.fit_transform(x_train)
                    x_test_fit = pca.fit_transform(x_test)
                    end = timer()
                    print("Successful reduction of training set to " + str(pca.n_components_) + " dimensions")
                    print("Reduction took " + str(end - start) + " seconds")

                elif reduction == "TruncatedSVD":
                    # 100 dimensions is recommended for LSA
                    reduce_to = 100
                    tsvd = TruncatedSVD(reduce_to)
                    print("Attempting to fit TruncatedSVD LSA to " + str(reduce_to) + " dimensions")
                    start = timer()
                    tsvd.fit(x_train)
                    end = timer()
                    # print("Successful fitting of TruncatedSVD reduction to "+str(tsvd.n_components_)+" components")
                    print("Fitting took " + str(end - start) + " seconds")
                    # ndim = tsvd.n_components_

                    print("Attempting to reduce the training set and test set")
                    start = timer()
                    x_train_fit = tsvd.fit_transform(x_train)
                    x_test_fit = tsvd.fit_transform(x_test)
                    end = timer()
                    # print("Successful reduction of training set to " + str(tsvd.n_components_) + " dimensions")
                    print("Reduction took " + str(end - start) + " seconds")
                else:
                    print("No dim reduction will be performed")
            else: # reduce_first:
                ndim = 0
                if reduction == "PCA":
                    # retain 95% variance
                    pca = PCA(0.95)
                    x_bag_train = np.array(x_bag_train)

                    print("Attempting to fit PCA reduction with variance retain rate of 95% to x_eng_train vector")
                    start = timer()
                    pca.fit(x_eng_train)
                    end = timer()
                    print("Successful fitting of PCA reduction to " + str(pca.n_components_) + " components")
                    print("Fitting took " + str(end - start) + " seconds")
                    ndim += pca.n_components_

                    print("Attempting to reduce x_eng_train and x_eng_test")
                    start = timer()
                    x_eng_train_fit = pca.fit_transform(x_eng_train)
                    x_eng_test_fit = pca.fit_transform(x_eng_test)
                    end = timer()
                    print("Successful reduction of x_eng to " + str(pca.n_components_) + " dimensions")
                    print("Reduction took " + str(end - start) + " seconds")

                    print("Attempting to fit PCA reduction with variance retain rate of 95% to x_ger_train vector")
                    start = timer()
                    pca.fit(x_ger_train)
                    end = timer()
                    print("Successful fitting of PCA reduction to " + str(pca.n_components_) + " components")
                    print("Fitting took " + str(end - start) + " seconds")
                    ndim += pca.n_components_

                    print("Attempting to reduce x_ger_train and x_ger_test")
                    start = timer()
                    x_ger_train_fit = pca.fit_transform(x_ger_train)
                    x_ger_test_fit = pca.fit_transform(x_ger_test)
                    end = timer()
                    print("Successful reduction of x_ger to " + str(pca.n_components_) + " dimensions")
                    print("Reduction took " + str(end - start) + " seconds")

                    print("Attempting to fit PCA reduction with variance retain rate of 95% to x_bag_train")
                    start = timer()
                    pca.fit(x_bag)
                    end = timer()
                    print("Successful fitting of PCA reduction to " + str(pca.n_components_) + " components")
                    print("Fitting took " + str(end - start) + " seconds")
                    ndim += pca.n_components_

                    print("Attempting to reduce x_bag_train and x_bag_test")
                    start = timer()
                    x_bag_train_fit = pca.fit_transform(x_bag_train)
                    x_bag_test_fit = pca.fit_transform(x_bag_test)
                    end = timer()
                    print("Successful reduction of x_bag to " + str(pca.n_components_) + " dimensions")
                    print("Reduction took " + str(end - start) + " seconds")

                    print("All reductions successful, new size of feature vector is " + str(ndim))

                    print("Concatenating reduced feature vectors")
                    x_train_fit = np.concatenate((x_eng_train_fit, x_ger_train_fit), axis = 1)
                    x_train_fit = np.concatenate((x_train_fit, x_bag_train_fit), axis = 1)
                    x_test_fit = np.concatenate((x_eng_test_fit, x_ger_test_fit),  axis=1)
                    x_test_fit = np.concatenate((x_test_fit, x_bag_test_fit), axis = 1)
                    print("Concatenation successful")

            if reduction not in results:
                results[reduction] = {}
            results[reduction][order] = {
                "x_train_fit": x_train_fit.tolist(),
                "x_test_fit": x_test_fit.tolist(),
                "y_train": y_train,
                "y_test": y_test
            }
    return results
