import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model, Sequential
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
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
from sklearn.feature_extraction.text import TfidfVectorizer
from tokenizer import text_to_tokens
from keras.callbacks import EarlyStopping

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
        estimator = KerasClassifier(build_fn=build_adaline_no_bias, epochs=100, batch_size=5, verbose=1)
    else:
        estimator = KerasClassifier(build_fn=build_adaline_with_bias, epochs=100, batch_size=5, verbose=1)

    splits = 5
    seed = 7
    np.random.seed(seed)
    kf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

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
    kf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

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
    old_y = y
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
    for train, test in kf.split(x, old_y):

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


# if dim_layer3 == 0, no second hidden layer
# dim_output is the number of classes
def build_mlp(dim_layer1, dim_layer2, dim_layer3, activation_layer3, dim_output):

    model = Sequential()
    model.add(Dense(dim_layer1, activation="relu", input_dim=dim_layer1))
    model.add(Dense(dim_layer2, activation="relu"))
    if not dim_layer3 == 0:
        model.add(Dense(dim_layer3, activation=activation_layer3))
    model.add(Dense(dim_output, activation="softmax"))

    # es = EarlyStopping(monitor='val_loss', mode='min', verbose = 1, patience=10)

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model


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


def keras_mlp_loop_all(classes_num):
    layer2_num = [10, 20, 50]
    layer3_num = [0, 10]
    layer3_activation = ["relu", "sigmoid"]

    accuracy_list = []
    model_list = []
    hacklist = []
    orders = ["reduce_first", "reduce_last"]
    reductions = ["PCA", "TruncatedSVD"]
    for reduction in reductions:
        for order in orders:
            print("###########")
            print("Class of models: "+order+" "+reduction)
            print("###########")
            if classes_num == 3 and not (order == "reduce_last" and reduction == "TruncatedSVD"):
                print("Not enough RAM memory to support "+order+" "+reduction)
                continue

            with open("../movie_dataset/mlp_matrix_" + str(classes_num) + "_"+ order + "_" + reduction + ".json", "r", encoding='utf-8') as f:
                results = json.load(f)
                x_train = results["x_train_fit"]
                x_test = results["x_test_fit"]
                y_train = results["y_train"]
                y_test = y_test_old = results["y_test"]

                x_train_60, x_validate, y_train_60, y_validate = train_test_split(x_train, y_train, test_size=0.25, stratify=y_train, random_state=7)

                # One-hot encoding
                encoder = LabelEncoder()
                encoder.fit(y_train)
                encoded_Y_60 = encoder.transform(y_train_60)
                encoded_y_validate = encoder.transform(y_validate)
                encoded_y_test = encoder.transform(y_test)
                # convert integers to dummy variables (i.e. one hot encoded)
                y_train_60 = np_utils.to_categorical(encoded_Y_60)
                y_validate = np_utils.to_categorical(encoded_y_validate)
                y_test = np_utils.to_categorical(encoded_y_test)

                number_of_features = len(x_train[0])

                for l2num in layer2_num:
                    for l3num in layer3_num:
                        for l3act in layer3_activation:
                            print("")
                            print("Model description: ")
                            print("Input layer: " + str(number_of_features)+ " neurons")
                            print("First hidden layer: " + str(l2num) + " neurons")
                            if not l3num == 0:
                                print("Second hidden layer: " + str(l3num) + " neurons, " +l3act+ " activations")
                            else:
                                print("No second hidden layer")
                            print("------------------------")
                            model = build_mlp(number_of_features, l2num, l3num, l3act, classes_num)

                            x_train_60 = np.array(x_train_60)
                            y_train_60 = np.array(y_train_60)
                            x_validate = np.array(x_validate)
                            y_validate = np.array(y_validate)
                            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
                            # ToDo Checkpointing mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)
                            history = model.fit(x_train_60, y_train_60, validation_data=(x_validate, y_validate),  verbose=1, epochs=1, callbacks = [es])
                            _, validation_accuracy = model.evaluate(x_validate, y_validate)
                            _, train_accuracy = model.evaluate(x_train_60, y_train_60)
                            print('Train: %.3f, Test: %.3f' % (train_accuracy, validation_accuracy))
                            print("------------------------")
                            accuracy_list.append(validation_accuracy)
                            model_list.append(model)
                            hacklist.append([x_test, y_test])


    print("###################")
    print("Final evaluation: ")

    best_index = np.argmax(accuracy_list)

    #get from hacklist
    x_test = hacklist[best_index][0]
    y_test = hacklist[best_index][1]
    #with open("../movie_dataset/mlp_matrix_" + str(classes_num) + "_" + order + "_" + reduction + ".json", "r",
    #          encoding='utf-8') as f:
    #    results = json.load(f)
    #    x_test = results["x_test_fit"]
    #    y_test = results["y_test"]


    best_model = model_list[best_index]
    print("Testing best model: ")
    _, test_accuracy = best_model.evaluate(np.array(x_test), np.array(y_test))
    print('Accuracy on test set: %.3f' % test_accuracy)
    print("------------------------")

    y_pred = best_model.predict(np.array(x_test))
    y_pred_categorical = []
    for row in y_pred:
        pred_class = np.argmax(row)
        y_pred_categorical.append(pred_class)
    y_pred = np.array(y_pred_categorical)

    y_test_old = y_test
    y_test_categorical = []
    for row in y_test_old:
        pred_class = np.argmax(row)
        y_test_categorical.append(pred_class)
    y_test_old = np.array(y_test_categorical)

    plotting.calculate_normalized_confusion_matrix(y_test_old, y_pred, class_num=classes_num,
                                                   title="Best hyperparameters combination")
    plotting.show_confusion_matrix()




# TODO fix memory error
def keras_mlp_prepare_data(data_set_json, classes_num=2, levenshtein=5):
    print("Preparing lexicons")
    _, engDict = build_english()  # swap the dict if needed
    engDictStemmed = stemmer.stem_dictionary(engDict)
    _, gerDict = build_german()  # swap the dict if needed
    gerDictStemmed = stemmer.stem_dictionary(gerDict)
    print("Lexicons ready")
    print("Beginning to build feature vectors")

    # put back together list of comments to be used with TfidfVectorizer
    comment_list = []
    for data in data_set_json:
        glued_comment = " ".join(data['tokens_original'])
        comment_list.append(glued_comment)

    print("Comments glued back together")
    tf_idf_vectorizer = TfidfVectorizer(tokenizer=text_to_tokens)
    tfidf_vectors = tf_idf_vectorizer.fit_transform(comment_list)

    # count_vect = ml_algorithms.getCountVector(classes_num)

    x_eng = []
    x_ger = []
    x_bag = []
    x_composite = []
    y = []

    xbags = tfidf_vectors.toarray().tolist()
    comment_list = []
    print("Starting comment parsing loop")
    cnt = 1

    for data, x_bag_row in zip(data_set_json, xbags):
        print("Comment "+str(cnt) + "/" +str(len(xbags)))
        cnt += 1
        sentiment_class = data['class_att']
        tokens_original = data['tokens_original']
        tokens_stemmed = data['tokens_stemmed']
        # Get features from english lexicon
        x_eng_row = comment_weight_vector(engDictStemmed, tokens_original, tokens_stemmed, levenshtein)

        # Get features from german lexicon
        x_ger_row = comment_weight_vector(gerDictStemmed, tokens_original, tokens_stemmed, levenshtein)

        # Get cumulative predictors from German and English lexicons, currently not used
        # summ_eng = comment_weight_calculation(engDictStemmed, "English", tokens_original,
        #                                      tokens_stemmed, 5, modification_use=False,
        #                                      amplification_use=False)
        # summ_ger = comment_weight_calculation(gerDictStemmed, "German", tokens_original, tokens_stemmed,
        #                                      5, modification_use=False, amplification_use=False)

        # Get bag of words feature vector

        # comment = ""
        # for word in tokens_original:
        #   comment += " "+word
        # dict = ml_algorithms.getOccurNumberDictionary(comment, count_vect)
        # keys_list = list(dict.keys())
        # keys_list.sort()
        # x_bag_row = []
        # for word in keys_list:
        #    x_bag_row.append(dict[word])

        x_eng.append(x_eng_row)
        x_ger.append(x_ger_row)
        x_bag.append(x_bag_row)
        x_composite.append(x_eng_row + x_ger_row + x_bag_row)

        # y is not one-hot encoded here
        y.append(class_encode(sentiment_class))

    xbags = [] #clear memory
    # Create a collected feature matrix, dim is num_of_comments x (ENG_DIM+GER_DIM+BAG_DIM)
    x_composite_matrix = np.array(x_composite)
    # Split feature matrix into training (80%) and test (20%)
    # Using a stratified split with a fixed seed (for result repeatability and caching purposes)
    # No kfolding because it would take too long to train, kfolding can be used on the training part (80%) to tune
    # the hyper parameters and pick the best model when the feature vector is already reduced to less dimensions
    x_train, x_test, y_train, y_test, x_eng_train, x_eng_test, x_ger_train, x_ger_test, x_bag_train, x_bag_test = \
        train_test_split(x_composite_matrix, y, x_eng, x_ger, x_bag, test_size=0.2, stratify=y, random_state=7)
    print("Stratified split 80-20 with random seed 7 for reproducibility successful")

    reduction_all = ["TruncatedSVD", "PCA"]
    order_all = ["reduce_last", "reduce_first"]

    for reduction in reduction_all:
        for order in order_all:

            results = {}

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
                    ncomp = 100
                    tsvd = TruncatedSVD(ncomp)
                    print("Attempting to fit TruncatedSVD LSA to " + str(ncomp) + " dimensions")
                    start = timer()
                    tsvd.fit(x_train)
                    end = timer()
                    print("Successful fitting of TruncatedSVD reduction to "+str(ncomp)+" components")
                    print("Fitting took " + str(end - start) + " seconds")
                    ndim = ncomp

                    print("Attempting to reduce the training set and test set")
                    start = timer()
                    x_train_fit = tsvd.fit_transform(x_train)
                    x_test_fit = tsvd.fit_transform(x_test)
                    end = timer()
                    print("Successful reduction of training set to " + str(ncomp) + " dimensions")
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
                elif reduction == "TruncatedSVD": # TruncatedSVD
                    # retain 95% variance
                    ncomp = 100
                    tsvd = TruncatedSVD(ncomp)
                    x_bag_train = np.array(x_bag_train)

                    print("Attempting to fit TruncatedSVD reduction to 100 components to x_eng_train vector")
                    start = timer()
                    tsvd.fit(x_eng_train)
                    end = timer()
                    print("Successful fitting of TruncatedSVD reduction to " + str(ncomp) + " components")
                    print("Fitting took " + str(end - start) + " seconds")
                    ndim += ncomp

                    print("Attempting to reduce x_eng_train and x_eng_test")
                    start = timer()
                    x_eng_train_fit = tsvd.fit_transform(x_eng_train)
                    x_eng_test_fit = tsvd.fit_transform(x_eng_test)
                    end = timer()
                    print("Successful reduction of x_eng to " + str(ncomp) + " dimensions")
                    print("Reduction took " + str(end - start) + " seconds")

                    print("Attempting to fit TruncatedSVD reduction to 100 components to x_ger_train vector")
                    start = timer()
                    tsvd.fit(x_ger_train)
                    end = timer()
                    print("Successful fitting of TruncatedSVD reduction to " + str(ncomp) + " components")
                    print("Fitting took " + str(end - start) + " seconds")
                    ndim += ncomp

                    print("Attempting to reduce x_ger_train and x_ger_test")
                    start = timer()
                    x_ger_train_fit = tsvd.fit_transform(x_ger_train)
                    x_ger_test_fit = tsvd.fit_transform(x_ger_test)
                    end = timer()
                    print("Successful reduction of x_ger to " + str(ncomp) + " dimensions")
                    print("Reduction took " + str(end - start) + " seconds")

                    print("Attempting to fit TruncatedSVD reduction t0 100 components to x_bag_train")
                    start = timer()
                    tsvd.fit(x_bag)
                    end = timer()
                    print("Successful fitting of TruncatedSVD reduction to " + str(ncomp) + " components")
                    print("Fitting took " + str(end - start) + " seconds")
                    ndim += ncomp

                    print("Attempting to reduce x_bag_train and x_bag_test")
                    start = timer()
                    x_bag_train_fit = tsvd.fit_transform(x_bag_train)
                    x_bag_test_fit = tsvd.fit_transform(x_bag_test)
                    end = timer()
                    print("Successful reduction of x_bag to " + str(ncomp) + " dimensions")
                    print("Reduction took " + str(end - start) + " seconds")

                    print("All reductions successful, new size of feature vector is " + str(ndim))

                    print("Concatenating reduced feature vectors")
                    x_train_fit = np.concatenate((x_eng_train_fit, x_ger_train_fit), axis = 1)
                    x_train_fit = np.concatenate((x_train_fit, x_bag_train_fit), axis = 1)
                    x_test_fit = np.concatenate((x_eng_test_fit, x_ger_test_fit),  axis=1)
                    x_test_fit = np.concatenate((x_test_fit, x_bag_test_fit), axis = 1)
                    print("Concatenation successful")

            results = {
                "x_train_fit": x_train_fit.tolist(),
                "x_test_fit": x_test_fit.tolist(),
                "y_train": y_train,
                "y_test": y_test
            }
            print("Writing to json file")
            with open("../movie_dataset/mlp_matrix_" + str(classes_num) + "_"+ order + "_" + reduction + ".json", "w", encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False)


