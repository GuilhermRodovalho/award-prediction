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
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import statistics
import numpy as np


def generateTrainAndTest(opt):

    filename = ''
    if opt == 'oscars':
        filename = 'oscar_movies_data.json'
    elif opt == 'golden_globe':
        filename = 'golden_globe_movies_data.json'
    elif opt == 'golden_globe_comedy':
        filename = 'golden_globe_movies_comedy_data.json'
    elif opt == 'the_game_awards':
        filename = 'the_game_awards_data.json'
    elif opt == 'the_game_awards_2024':
        filename = 'the_game_awards_2024_data.json'
    else:
        print('Erro')
        exit(1)

    with open('data/'+filename, 'r') as json_file:
        movies = json.load(json_file)
        
        # First pass: collect all user reviews to calculate yearly averages
        yearly_user_reviews = {}
        for movie in movies:
            year = movies[movie].get("year", "unknown")
            if year not in yearly_user_reviews:
                yearly_user_reviews[year] = []

            user_reviews = movies[movie].get("user-review", []) or movies[movie].get("user-reviews", [])
            if user_reviews:
                yearly_user_reviews[year].extend([int(x) for x in user_reviews])

        # Calculate yearly averages
        yearly_averages = {}
        for year, reviews in yearly_user_reviews.items():
            if reviews:
                yearly_averages[year] = statistics.mean(reviews)
            else:
                yearly_averages[year] = 5.0  # Default fallback value
        
        for movie in movies:
            movie_year = movies[movie].get("year", "unknown")
            movies[movie].pop('year', None)
            movies[movie].pop('cerimony-date', None)
            movies[movie].pop('name', None)

            winner_val = movies[movie]['winner'].lower()
            if winner_val == 'falso' or winner_val == 'false':
                movies[movie]['class'] = 'Loser'
            elif winner_val == 'verdadeiro' or winner_val == 'true':
                movies[movie]['class'] = 'Winner'
            else:
                movies[movie]['class'] = 'Loser'
            movies[movie].pop('winner')

            # Handle both 'user-review' and 'user-reviews' keys for compatibility
            user_reviews = movies[movie].get("user-review", []) or movies[movie].get("user-reviews", [])
            if not user_reviews:
                # Use yearly average for this movie's year
                if movie_year in yearly_averages:
                    avg_rating = yearly_averages[movie_year]
                else:
                    avg_rating = (
                        sum(yearly_averages.values()) / len(yearly_averages)
                        if yearly_averages
                        else 5.0
                    )
                # Create a list with the average value repeated to simulate reviews
                user_reviews = [int(avg_rating)] * 5
            else:
                user_reviews = list(map(int, user_reviews))
            
            critic_review = [int(x) // 10.0 for x in (movies[movie].get("critic-review", []) or movies[movie].get("critic-reviews", []))]
            movies[movie]['user-mean'] = round(
                statistics.mean(user_reviews), 2)
            movies[movie]['user-stdev'] = round(
                statistics.stdev(user_reviews) if len(user_reviews) > 1 else 0, 2)
            movies[movie]['user-median'] = round(
                statistics.median(user_reviews), 2)
            movies[movie]['user-mode'] = round(
                statistics.mode(user_reviews), 2)
            movies[movie]['user-percentile-25'] = round(
                np.percentile(user_reviews, 25), 2)
            movies[movie]['user-percentile-75'] = round(
                np.percentile(user_reviews, 75), 2)
            # Remove the review keys that were used
            movies[movie].pop("user-review", None)
            movies[movie].pop("user-reviews", None)

            movies[movie]['critic-mean'] = round(
                statistics.mean(critic_review), 2)
            movies[movie]['critic-stdev'] = round(
                statistics.stdev(critic_review) if len(critic_review) > 1 else 0, 2)
            movies[movie]['critic-median'] = round(
                statistics.median(critic_review), 2)
            movies[movie]['critic-mode'] = round(
                statistics.mode(critic_review), 2)
            movies[movie]['critic-percentile-25'] = round(
                np.percentile(critic_review, 25), 2)
            movies[movie]['critic-percentile-75'] = round(
                np.percentile(critic_review, 75), 2)
            movies[movie].pop("critic-review", None)
            movies[movie].pop("critic-reviews", None)

        df = pd.DataFrame.from_dict(movies, orient='index')
        print(df.loc[df['class'] == 'Winner'][['critic-mean', 'user-mean']])
        ros = RandomOverSampler(random_state=80)
        x_train, x_test, y_train, y_test = train_test_split(
            df.drop('class', axis=1), df['class'], test_size=0.2, stratify=df['class'], random_state=80)
        # print(y_test)
        # print((y_train == 'Loser').sum())
        # print((y_train == 'Winner').sum())
        x_resampled, y_resampled = ros.fit_resample(
            x_train, y_train)
        # print((y_resampled == 'Loser').sum())
        # print((y_resampled == 'Winner').sum())
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
        plt.suptitle(opt + ' naive bayes')
        plt.show()

        proba = model.predict_proba(x_test)

        proba_y = pd.concat([pd.DataFrame(index=x_test.index.values,
                                          data=proba, columns=labels), y_test], axis=1)

        print(proba_y)


def knn(opt, detail):
    k = 0

    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    if (opt == 'oscars'):
        k = 14
    elif (opt == 'golden_globe'):
        k = 5
    elif (opt == 'the_game_awards' or opt == 'the_game_awards_2024'):
        k = 10
    else:
        k = 15

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
        plt.suptitle(opt + ' knn')
        plt.show()

        proba = model.predict_proba(x_test)

        proba_y = pd.concat([pd.DataFrame(index=x_test.index.values,
                                          data=proba, columns=labels), y_test], axis=1)

        print(proba_y)


def randomForest(opt, detail):
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)
    min_leaf = 0

    if (opt == 'oscars'):
        min_leaf = 15
    elif (opt == 'the_game_awards' or opt == 'the_game_awards_2024'):
        min_leaf = 5
    else:
        min_leaf = 3

    model = RandomForestClassifier(
        criterion='entropy', random_state=80, min_samples_leaf=min_leaf, min_samples_split=2)
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
        plt.suptitle(opt + ' random forest')
        plt.show()

        proba = model.predict_proba(x_test)

        proba_y = pd.concat([pd.DataFrame(index=x_test.index.values,
                                          data=proba, columns=labels), y_test], axis=1)

        print(proba_y)


if __name__ == '__main__':
    details = True
    # for name in ['oscars', 'golden_globe', 'golden_globe_comedy', 'the_game_awards', 'the_game_awards_2024']:
    for name in ['the_game_awards']:
        print('\nResultados do ' + name)
        # generateTrainAndTest(name)
        print('\nNaive Bayes')
        naiveBayes(name, details)
        print('\nKNN')
        knn(name, details)
        print('\nRandom Forest')
        randomForest(name, details)