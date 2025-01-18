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
    recall_score,
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import statistics
import numpy as np


def generateTrainAndTest(opt):
    filename = ""
    if opt == "oscars":
        filename = "oscar_movies_data.json"
    elif opt == "golden_globe":
        filename = "golden_globe_movies_data.json"
    elif opt == "golden_globe_comedy":
        filename = "golden_globe_movies_comedy_data.json"
    else:
        print("Erro")
        exit(1)

    with open("data/" + filename, "r") as json_file:
        movies = json.load(json_file)
        for movie in movies:
            movies[movie].pop("year")
            movies[movie].pop("cerimony-date")

            if (
                movies[movie]["winner"].lower() == "falso"
                or movies[movie]["winner"].lower() == "false"
            ):
                movies[movie]["class"] = "Loser"
            else:
                movies[movie]["class"] = "Winner"
            movies[movie].pop("winner")

            user_reviews = list(map(int, movies[movie]["user-review"]))
            critic_review = [int(x) // 10.0 for x in movies[movie]["critic-review"]]
            movies[movie]["user-mean"] = round(statistics.mean(user_reviews), 2)
            movies[movie]["user-stdev"] = round(statistics.stdev(user_reviews), 2)
            movies[movie]["user-median"] = round(statistics.median(user_reviews), 2)
            movies[movie]["user-mode"] = round(statistics.mode(user_reviews), 2)
            movies[movie]["user-percentile-25"] = round(
                np.percentile(user_reviews, 25), 2
            )
            movies[movie]["user-percentile-75"] = round(
                np.percentile(user_reviews, 75), 2
            )
            movies[movie].pop("user-review")

            movies[movie]["critic-mean"] = round(statistics.mean(critic_review), 2)
            movies[movie]["critic-stdev"] = round(statistics.stdev(critic_review), 2)
            movies[movie]["critic-median"] = round(statistics.median(critic_review), 2)
            movies[movie]["critic-mode"] = round(statistics.mode(critic_review), 2)
            movies[movie]["critic-percentile-25"] = round(
                np.percentile(critic_review, 25), 2
            )
            movies[movie]["critic-percentile-75"] = round(
                np.percentile(critic_review, 75), 2
            )
            movies[movie].pop("critic-review")

        df = pd.DataFrame.from_dict(movies, orient="index")
        print(df.loc[df["class"] == "Winner"][["critic-mean", "user-mean"]])
        ros = RandomOverSampler(random_state=80)
        x_train, x_test, y_train, y_test = train_test_split(
            df.drop("class", axis=1),
            df["class"],
            test_size=0.2,
            stratify=df["class"],
            random_state=80,
        )
        # print(y_test)
        # print((y_train == 'Loser').sum())
        # print((y_train == 'Winner').sum())
        x_resampled, y_resampled = ros.fit_resample(x_train, y_train)
        # print((y_resampled == 'Loser').sum())
        # print((y_resampled == 'Winner').sum())
        return x_resampled, x_test, y_resampled, y_test


def naiveBayes(opt, detail):
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    model = GaussianNB()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuray = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="binary", pos_label="Winner")
    precision = precision_score(y_pred, y_test, average="binary", pos_label="Winner")
    recall = recall_score(y_pred, y_test, average="binary", pos_label="Winner")

    print("Accuracy:", accuray)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    if detail is True:
        labels = unique_labels(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot()
        plt.suptitle(opt + " naive bayes")
        plt.show()

        proba = model.predict_proba(x_test)

        proba_y = pd.concat(
            [
                pd.DataFrame(index=x_test.index.values, data=proba, columns=labels),
                y_test,
            ],
            axis=1,
        )

        print(proba_y)


def knn(opt, detail):
    k = 0

    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    if opt == "oscars":
        k = 14
    elif opt == "golden_globe":
        k = 5
    else:
        k = 15

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuray = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="binary", pos_label="Winner")
    precision = precision_score(y_pred, y_test, average="binary", pos_label="Winner")
    recall = recall_score(y_pred, y_test, average="binary", pos_label="Winner")

    print("Accuracy:", accuray)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    if detail is True:
        labels = unique_labels(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        disp.plot()
        plt.suptitle(opt + " knn")
        plt.show()

        proba = model.predict_proba(x_test)

        proba_y = pd.concat(
            [
                pd.DataFrame(index=x_test.index.values, data=proba, columns=labels),
                y_test,
            ],
            axis=1,
        )

        print(proba_y)


def randomForest(opt, detail):
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)
    min_leaf = 0

    if opt == "oscars":
        min_leaf = 15
    else:
        min_leaf = 3

    model = RandomForestClassifier(
        criterion="entropy",
        random_state=80,
        min_samples_leaf=min_leaf,
        min_samples_split=2,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuray = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="binary", pos_label="Winner")
    precision = precision_score(y_pred, y_test, average="binary", pos_label="Winner")
    recall = recall_score(y_pred, y_test, average="binary", pos_label="Winner")

    print("Accuracy:", accuray)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    if detail is True:
        labels = unique_labels(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

        disp.plot()
        plt.suptitle(opt + " random forest")
        plt.show()

        proba = model.predict_proba(x_test)

        proba_y = pd.concat(
            [
                pd.DataFrame(index=x_test.index.values, data=proba, columns=labels),
                y_test,
            ],
            axis=1,
        )

        print(proba_y)


if __name__ == "__main__":
    details = True
    for name in ["oscars", "golden_globe", "golden_globe_comedy"]:
        print("\nResultados do " + name)
        # generateTrainAndTest(name)
        print("\nNaive Bayes")
        naiveBayes(name, details)
        print("\nKNN")
        knn(name, details)
        print("\nRandom Forest")
        randomForest(name, details)
