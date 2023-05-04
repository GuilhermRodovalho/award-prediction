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


def generateTrainAndTest(movies):
    with open('oscars_train_test.json', 'w') as output_file:
        df = pd.DataFrame.from_dict(movies, orient='index')
        df.drop('year', axis=1, inplace=True)
        df.drop('cerimony-date', axis=1, inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(
            df.drop('winner', axis=1), df['winner'], test_size=0.2, stratify=df['winner'])
        json.dump({'x_train': X_train.to_dict('tight'), 'x_test': X_test.to_dict('tight'),
                  'y_train': y_train.to_dict(), 'y_test': y_test.to_dict()}, output_file)


def naiveBayes(movies):
    with open('oscars_train_test.json', 'r') as train_file:
        file_json = json.load(train_file)
        X_train = pd.DataFrame.from_dict(file_json['x_train'], orient='tight')
        X_test = pd.DataFrame.from_dict(file_json['x_test'], orient='tight')
        y_train = pd.Series(file_json['y_train'])
        y_test = pd.Series(file_json['y_test'])

        model = GaussianNB()

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuray = accuracy_score(y_pred, y_test)
        f1 = f1_score(y_pred, y_test, average="micro")

        print("Accuracy:", accuray)
        print("F1 Score:", f1)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=['Loser', 'Winner'])
        disp.plot()
        plt.show()


if __name__ == '__main__':
    with open('oscar_movies_statistics.json', 'r') as json_file:
        movies = json.load(json_file)
        naiveBayes(movies)
