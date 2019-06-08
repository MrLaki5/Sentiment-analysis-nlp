import numpy as np
import pandas as pd
import keras
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from sklearn.model_selection import KFold
from sentiment_logic import comment_weight_calculation
from eng_dict import build_english
from ger_dict import build_german
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from stemmer import stemmer


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

    train_data = pd.read_csv('../movie_dataset/SerbMR-2C.csv')

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


def keras_2_layer():
    # define two sets of inputs
    inputA = keras.Input(shape=(32,))
    inputB = keras.Input(shape=(128,))

    # the first branch operates on the first input
    x = Dense(8, activation="relu")(inputA)
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

