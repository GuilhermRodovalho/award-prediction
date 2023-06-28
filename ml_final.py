import pandas as pd
import json
from sklearn.discriminant_analysis import unique_labels
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import statistics


def generateDataframes(json_input):
    movies = json.load(json_input)
    for movie in movies:
        movies[movie].pop('year')
        movies[movie].pop('cerimony-date')

        if movies[movie]['winner'].lower() == 'falso' or movies[movie]['winner'].lower() == 'false':
            movies[movie]['class'] = 'Loser'
        else:
            movies[movie]['class'] = 'Winner'
        movies[movie].pop('winner')

        user_reviews = list(map(int, movies[movie]['user-review']))
        critic_review = [
            int(x)//10.0 for x in movies[movie]['critic-review']]
        movies[movie]['user-mean'] = round(
            statistics.mean(user_reviews), 2)
        movies[movie]['user-stdev'] = round(
            statistics.stdev(user_reviews), 2)
        movies[movie]['user-median'] = round(
            statistics.median(user_reviews), 2)
        movies[movie]['user-mode'] = round(
            statistics.mode(user_reviews), 2)
        movies[movie]['user-percentile-25'] = round(
            np.percentile(user_reviews, 25), 2)
        movies[movie]['user-percentile-75'] = round(
            np.percentile(user_reviews, 75), 2)
        movies[movie].pop('user-review')

        movies[movie]['critic-mean'] = round(
            statistics.mean(critic_review), 2)
        movies[movie]['critic-stdev'] = round(
            statistics.stdev(critic_review), 2)
        movies[movie]['critic-median'] = round(
            statistics.median(critic_review), 2)
        movies[movie]['critic-mode'] = round(
            statistics.mode(critic_review), 2)
        movies[movie]['critic-percentile-25'] = round(
            np.percentile(critic_review, 25), 2)
        movies[movie]['critic-percentile-75'] = round(
            np.percentile(critic_review, 75), 2)
        movies[movie].pop('critic-review')

    df = pd.DataFrame.from_dict(movies, orient='index')

    return df


def generateTrainAndTest(opt):
    train_filename = ''
    test_filename = ''
    if opt == 'oscars':
        train_filename = 'oscar_movies_data.json'
        test_filename = 'oscar_movies_2023_data.json'
    elif opt == 'golden_globe':
        train_filename = 'golden_globe_movies_data.json'
        test_filename = 'golden_globe_movies_2023_data.json'
    elif opt == 'golden_globe_comedy':
        train_filename = 'golden_globe_movies_comedy_data.json'
        test_filename = 'golden_globe_movies_comedy_2023_data.json'
    else:
        print('Erro')
        exit(1)

    with open('data/'+train_filename, 'r') as json_train, open('data/'+test_filename) as json_test:
        df_train = generateDataframes(json_train)
        df_test = generateDataframes(json_test)

        ros = RandomOverSampler(random_state=80)
        x_resampled, y_resampled = ros.fit_resample(
            df_train.drop('class', axis=1), df_train['class'])

        return x_resampled, df_test.drop('class', axis=1), y_resampled, df_test['class']


def naiveBayes(opt):
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    model = GaussianNB()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    proba = model.predict_proba(x_test)

    labels = unique_labels(y_test, y_pred)

    proba_y = pd.DataFrame(index=x_test.index.values,
                           data=proba, columns=labels)
    proba_y['Loser'] = round(proba_y['Loser'], 4)
    proba_y['Winner'] = round(proba_y['Winner'], 4)

    print(proba_y)


def knn(opt):
    k = 0
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    if (opt == 'oscars'):
        k = 14
    elif (opt == 'golden_globe'):
        k = 5
    else:
        k = 15

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    labels = unique_labels(y_test, y_pred)

    proba = model.predict_proba(x_test)
    proba_y = pd.DataFrame(index=x_test.index.values,
                           data=proba, columns=labels)
    proba_y['Loser'] = round(proba_y['Loser'], 4)
    proba_y['Winner'] = round(proba_y['Winner'], 4)
    print(proba_y)


def randomForest(opt):
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    min_leaf = 0

    if (opt == 'oscars'):
        min_leaf = 15
    else:
        min_leaf = 3

    model = RandomForestClassifier(
        criterion='entropy', random_state=80, min_samples_leaf=min_leaf, min_samples_split=2)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    labels = unique_labels(y_test, y_pred)

    proba = model.predict_proba(x_test)

    proba_y = pd.DataFrame(index=x_test.index.values,
                           data=proba, columns=labels)
    proba_y['Loser'] = round(proba_y['Loser'], 4)
    proba_y['Winner'] = round(proba_y['Winner'], 4)

    print(proba_y)


if __name__ == '__main__':
    for name in ['oscars', 'golden_globe', 'golden_globe_comedy']:
        print('\nResultados do ' + name)
        print('\nNaive Bayes')
        naiveBayes(name)
        print('\nKNN')
        knn(name)
        print('\nRandom Forest')
        randomForest(name)
