import pandas as pd
import json
from sklearn.discriminant_analysis import unique_labels
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler


def generateTrainAndTest(opt):
    train_filename = ''
    test_filename = ''
    if opt == 'oscars':
        train_filename = 'oscar_movies_statistics.json'
        test_filename = 'oscar_movies_2023_statistics.json'
    elif opt == 'golden_globe':
        train_filename = 'golden_globe_movies_statistics.json'
        test_filename = 'golden_globe_movies_2023_statistics.json'
    else:
        print('Erro')
        exit(1)

    with open(train_filename, 'r') as train_file, open(test_filename, 'r') as test_file:
        train_info = json.load(train_file)
        test_info = json.load(test_file)

        df_train = pd.DataFrame.from_dict(train_info, orient='index')
        df_test = pd.DataFrame.from_dict(test_info, orient='index')

        ros = RandomOverSampler(random_state=80)
        x_resampled, y_resampled = ros.fit_resample(
            df_train.drop('class', axis=1), df_train['class'])

        return x_resampled, df_test.drop('class', axis=1), y_resampled, df_test['class']


def naiveBayes(opt):
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    model = CategoricalNB()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    proba = model.predict_proba(x_test)

    labels = unique_labels(y_test, y_pred)

    proba_y = pd.DataFrame(index=x_test.index.values,
                           data=proba, columns=labels)

    print(proba_y)


def decisionTree(opt):
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    model = tree.DecisionTreeClassifier(
        criterion='entropy', random_state=80)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    labels = unique_labels(y_test, y_pred)

    proba = model.predict_proba(x_test)
    proba_y = pd.DataFrame(index=x_test.index.values,
                           data=proba, columns=labels)
    print(proba_y)


def knn(opt):
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    model = KNeighborsClassifier(n_neighbors=16)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    labels = unique_labels(y_test, y_pred)

    proba = model.predict_proba(x_test)
    proba_y = pd.DataFrame(index=x_test.index.values,
                           data=proba, columns=labels)
    print(proba_y)


def randomForest(opt):
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    model = RandomForestClassifier(criterion='entropy', random_state=80)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    labels = unique_labels(y_test, y_pred)

    proba = model.predict_proba(x_test)

    proba_y = pd.DataFrame(index=x_test.index.values,
                           data=proba, columns=labels)

    print(proba_y)


if __name__ == '__main__':
    for name in ['oscars', 'golden_globe']:
        print('\nResultados do ' + name)
        print('\nNaive Bayes')
        naiveBayes(name)
        print('\nKNN')
        knn(name)
        print('\nDecision Tree')
        decisionTree(name)
        print('\nRandom Forest')
        randomForest(name)
