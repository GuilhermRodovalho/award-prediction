import pandas as pd
import json
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
import matplotlib.pyplot as plt


def naiveBayes(movies):
    df = pd.DataFrame.from_dict(movies, orient='index')
    df.drop('year', axis=1, inplace=True)
    df.drop('cerimony-date', axis=1, inplace=True)
    df.drop('user-mean', axis=1, inplace=True)
    df.drop('user-stdev', axis=1, inplace=True)
    df.drop('user-median', axis=1, inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('winner', axis=1), df['winner'], test_size=0.2)

    model = GaussianNB()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuray = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="micro")

    print("Accuracy:", accuray)
    print("F1 Score:", f1)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=['True', 'False'])
    disp.plot()

    y_test_df = y_test.to_frame()
    y_test_df['pred'] = y_pred

    print(y_test_df)
    plt.show()


if __name__ == '__main__':
    with open('oscar_movies_statistics.json', 'r') as json_file:
        movies = json.load(json_file)
        naiveBayes(movies)
