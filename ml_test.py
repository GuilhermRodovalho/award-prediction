import pandas as pd
import json
from sklearn.discriminant_analysis import unique_labels
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score)
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import graphviz
from imblearn.over_sampling import RandomOverSampler


def generateTrainAndTest(opt):
    filename = ''
    if opt == 'oscars':
        filename = 'oscar_movies_statistics.json'
    elif opt == 'golden_globe':
        filename = 'golden_globe_movies_statistics.json'
    else:
        print('Erro')
        exit(1)

    with open(filename, 'r') as train_file:
        train_info = json.load(train_file)
        df = pd.DataFrame.from_dict(train_info, orient='index')
        ros = RandomOverSampler(random_state=80)
        x_train, x_test, y_train, y_test = train_test_split(
            df.drop('class', axis=1), df['class'], test_size=0.2, stratify=df['class'], random_state=80)
        x_resampled, y_resampled = ros.fit_resample(
            x_train, y_train)
        return x_resampled, x_test, y_resampled, y_test


def naiveBayes(opt, detail):
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    model = GaussianNB()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuray = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="binary", pos_label='Winner')
    precision = precision_score(
        y_pred, y_test, average='binary', pos_label='Winner')
    recall = recall_score(y_pred, y_test, average='binary', pos_label='Winner')

    print("Accuracy:", accuray)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    if detail is True:
        labels = unique_labels(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=labels)
        disp.plot()
        plt.show()

        proba = model.predict_proba(x_test)

        proba_y = pd.concat([pd.DataFrame(index=x_test.index.values,
                                          data=proba, columns=labels), y_test], axis=1)

        print(proba_y)


def decisionTree(opt, detail):
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    model = tree.DecisionTreeClassifier(
        criterion='entropy', random_state=80)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuray = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="binary", pos_label='Winner')
    precision = precision_score(
        y_pred, y_test, average='binary', pos_label='Winner')
    recall = recall_score(y_pred, y_test, average='binary', pos_label='Winner')

    print("Accuracy:", accuray)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    if detail is True:

        labels = unique_labels(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=labels)

        disp.plot()
        plt.show()
        dot_data = tree.export_graphviz(
            model, out_file=None,
            feature_names=x_train.columns,
            class_names=labels,
            filled=True, rounded=True,
            special_characters=True)
        graph = graphviz.Source(dot_data)
        graph.render("oscars_train")

        proba = model.predict_proba(x_test)

        proba_y = pd.concat([pd.DataFrame(index=x_test.index.values,
                                          data=proba, columns=labels), y_test], axis=1)

        print(proba_y)


def knn(opt, detail):
    k = 0

    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    if (opt == 'oscars'):
        k = 10
    else:
        k = 5

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuray = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="binary", pos_label='Winner')
    precision = precision_score(
        y_pred, y_test, average='binary', pos_label='Winner')
    recall = recall_score(y_pred, y_test, average='binary', pos_label='Winner')

    print("Accuracy:", accuray)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    if detail is True:
        labels = unique_labels(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=labels)

        disp.plot()
        plt.show()

        proba = model.predict_proba(x_test)

        proba_y = pd.concat([pd.DataFrame(index=x_test.index.values,
                                          data=proba, columns=labels), y_test], axis=1)

        print(proba_y)


def randomForest(opt, detail):
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    model = RandomForestClassifier(criterion='entropy', random_state=80)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuray = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="binary", pos_label='Winner')
    precision = precision_score(
        y_pred, y_test, average='binary', pos_label='Winner')
    recall = recall_score(y_pred, y_test, average='binary', pos_label='Winner')

    print("Accuracy:", accuray)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    if detail is True:
        labels = unique_labels(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=labels)

        disp.plot()
        plt.show()

        proba = model.predict_proba(x_test)

        proba_y = pd.concat([pd.DataFrame(index=x_test.index.values,
                                          data=proba, columns=labels), y_test], axis=1)

        print(proba_y)


if __name__ == '__main__':
    details = False
    for name in ['oscars', 'golden_globe']:
        print('\nResultados do ' + name)
        print('\nNaive Bayes')
        naiveBayes(name, details)
        print('\nKNN')
        knn(name, details)
        print('\nDecision Tree')
        decisionTree(name, details)
        print('\nRandom Forest')
        randomForest(name, details)
