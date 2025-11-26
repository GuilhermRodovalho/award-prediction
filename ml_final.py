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
import os

# Create plots directory if it doesn't exist
os.makedirs("plots", exist_ok=True)


def generateTrainAndTestTemporal(opt, train_years=None, test_years=None):
    """
    Split data by year instead of random split.

    Args:
        opt: Dataset name
        train_years: List of years for training (e.g., ['2014', '2015', ...])
        test_years: List of years for testing (e.g., ['2023', '2024'])
    """
    # Load main historical data
    filename = ""
    if opt == "oscars":
        filename = "oscar_movies_data.json"
    elif opt == "golden_globe":
        filename = "golden_globe_movies_data.json"
    elif opt == "golden_globe_comedy":
        filename = "golden_globe_movies_comedy_data.json"
    elif opt == "the_game_awards":
        filename = "the_game_awards_data.json"
    else:
        print("Erro")
        exit(1)

    with open("data/" + filename, "r") as json_file:
        movies = json.load(json_file)

    # Load 2024 data if testing Game Awards and 2024 is in test_years
    if opt == "the_game_awards" and test_years and "2024" in test_years:
        try:
            with open("data/the_game_awards_2024_data.json", "r") as json_file_2024:
                movies_2024 = json.load(json_file_2024)
                movies.update(movies_2024)
        except FileNotFoundError:
            print("Warning: the_game_awards_2024_data.json not found")

    # Store years before processing
    movie_years = {}
    for movie in movies:
        movie_years[movie] = movies[movie].get("year", "unknown")

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

    # Second pass: process movies with missing user reviews filled
    for movie in movies:
        movie_year = movies[movie].get("year", "unknown")
        movies[movie].pop("year", None)
        movies[movie].pop("cerimony-date", None)
        movies[movie].pop("name", None)

        if (
            movies[movie]["winner"].lower() == "falso"
            or movies[movie]["winner"].lower() == "false"
        ):
            movies[movie]["class"] = "Loser"
        else:
            movies[movie]["class"] = "Winner"
        movies[movie].pop("winner")

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
        movies[movie]["user-mean"] = round(statistics.mean(user_reviews), 2)
        movies[movie]["user-stdev"] = round(
            statistics.stdev(user_reviews) if len(user_reviews) > 1 else 0, 2
        )
        movies[movie]["user-median"] = round(statistics.median(user_reviews), 2)
        movies[movie]["user-mode"] = round(statistics.mode(user_reviews), 2)
        movies[movie]["user-percentile-25"] = round(
            np.percentile(user_reviews, 25), 2
        )
        movies[movie]["user-percentile-75"] = round(
            np.percentile(user_reviews, 75), 2
        )
        movies[movie].pop("user-review", None)
        movies[movie].pop("user-reviews", None)

        movies[movie]["critic-mean"] = round(statistics.mean(critic_review), 2)
        movies[movie]["critic-stdev"] = round(
            statistics.stdev(critic_review) if len(critic_review) > 1 else 0, 2
        )
        movies[movie]["critic-median"] = round(statistics.median(critic_review), 2)
        movies[movie]["critic-mode"] = round(statistics.mode(critic_review), 2)
        movies[movie]["critic-percentile-25"] = round(
            np.percentile(critic_review, 25), 2
        )
        movies[movie]["critic-percentile-75"] = round(
            np.percentile(critic_review, 75), 2
        )
        movies[movie].pop("critic-review", None)
        movies[movie].pop("critic-reviews", None)

    df = pd.DataFrame.from_dict(movies, orient="index")

    # Split by year
    train_mask = df.index.map(lambda x: movie_years.get(x, "unknown") in train_years)
    test_mask = df.index.map(lambda x: movie_years.get(x, "unknown") in test_years)

    x_train = df[train_mask].drop("class", axis=1)
    y_train = df[train_mask]["class"]
    x_test = df[test_mask].drop("class", axis=1)
    y_test = df[test_mask]["class"]

    print("\n📊 Divisão Temporal:")
    print(f"TREINO: Anos {train_years} - {len(x_train)} jogos")
    print(f"  - Winners: {(y_train == 'Winner').sum()}")
    print(f"  - Losers: {(y_train == 'Loser').sum()}")
    print(f"TESTE: Anos {test_years} - {len(x_test)} jogos")
    print(f"  - Winners: {(y_test == 'Winner').sum()}")
    print(f"  - Losers: {(y_test == 'Loser').sum()}")

    # Apply RandomOverSampler to training data
    ros = RandomOverSampler(random_state=80)
    x_resampled, y_resampled = ros.fit_resample(x_train, y_train)

    return x_resampled, x_test, y_resampled, y_test


def generateTrainAndTest(opt):
    filename = ""
    if opt == "oscars":
        filename = "oscar_movies_data.json"
    elif opt == "golden_globe":
        filename = "golden_globe_movies_data.json"
    elif opt == "golden_globe_comedy":
        filename = "golden_globe_movies_comedy_data.json"
    elif opt == "the_game_awards":
        filename = "the_game_awards_data.json"
    else:
        print("Erro")
        exit(1)

    with open("data/" + filename, "r") as json_file:
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

        # Second pass: process movies with missing user reviews filled
        for movie in movies:
            movie_year = movies[movie].get("year", "unknown")
            movies[movie].pop("year", None)
            movies[movie].pop("cerimony-date", None)
            movies[movie].pop("name", None)

            if (
                movies[movie]["winner"].lower() == "falso"
                or movies[movie]["winner"].lower() == "false"
            ):
                movies[movie]["class"] = "Loser"
            else:
                movies[movie]["class"] = "Winner"
            movies[movie].pop("winner")

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
            movies[movie]["user-mean"] = round(statistics.mean(user_reviews), 2)
            movies[movie]["user-stdev"] = round(
                statistics.stdev(user_reviews) if len(user_reviews) > 1 else 0, 2
            )
            movies[movie]["user-median"] = round(statistics.median(user_reviews), 2)
            movies[movie]["user-mode"] = round(statistics.mode(user_reviews), 2)
            movies[movie]["user-percentile-25"] = round(
                np.percentile(user_reviews, 25), 2
            )
            movies[movie]["user-percentile-75"] = round(
                np.percentile(user_reviews, 75), 2
            )
            movies[movie].pop("user-review", None)
            movies[movie].pop("user-reviews", None)

            movies[movie]["critic-mean"] = round(statistics.mean(critic_review), 2)
            movies[movie]["critic-stdev"] = round(
                statistics.stdev(critic_review) if len(critic_review) > 1 else 0, 2
            )
            movies[movie]["critic-median"] = round(statistics.median(critic_review), 2)
            movies[movie]["critic-mode"] = round(statistics.mode(critic_review), 2)
            movies[movie]["critic-percentile-25"] = round(
                np.percentile(critic_review, 25), 2
            )
            movies[movie]["critic-percentile-75"] = round(
                np.percentile(critic_review, 75), 2
            )
            movies[movie].pop("critic-review", None)
            movies[movie].pop("critic-reviews", None)

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
        plt.savefig(f"plots/{opt}_naive_bayes_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

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
        plt.savefig(f"plots/{opt}_knn_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

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
        plt.savefig(f"plots/{opt}_random_forest_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()

        proba = model.predict_proba(x_test)

        proba_y = pd.concat(
            [
                pd.DataFrame(index=x_test.index.values, data=proba, columns=labels),
                y_test,
            ],
            axis=1,
        )

        print(proba_y)


def predictFutureYears(opt, train_years, test_years):
    """
    Train on historical years and predict/evaluate on future years.
    """
    print("=" * 80)
    print(f"🎯 PREVISÃO PARA {opt.upper()}")
    print("=" * 80)

    x_train, x_test, y_train, y_test = generateTrainAndTestTemporal(opt, train_years, test_years)

    # Store test games with their actual results
    test_games = x_test.index.values
    actual_results = y_test.values

    # Dictionary to store all predictions
    all_predictions = {}

    # Naive Bayes
    print("\n" + "=" * 80)
    print("📊 NAIVE BAYES")
    print("=" * 80)
    model_nb = GaussianNB()
    model_nb.fit(x_train, y_train)
    y_pred_nb = model_nb.predict(x_test)
    proba_nb = model_nb.predict_proba(x_test)

    acc_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Accuracy: {acc_nb:.2%}")

    if len(set(y_test)) > 1 and len(set(y_pred_nb)) > 1:
        f1_nb = f1_score(y_test, y_pred_nb, average="binary", pos_label="Winner")
        precision_nb = precision_score(y_test, y_pred_nb, average="binary", pos_label="Winner")
        recall_nb = recall_score(y_test, y_pred_nb, average="binary", pos_label="Winner")
        print(f"Precision: {precision_nb:.2%}")
        print(f"Recall: {recall_nb:.2%}")
        print(f"F1 Score: {f1_nb:.2%}")

    all_predictions['Naive Bayes'] = {'pred': y_pred_nb, 'proba': proba_nb}

    # KNN
    print("\n" + "=" * 80)
    print("📊 K-NEAREST NEIGHBORS")
    print("=" * 80)
    k = 14 if opt == "oscars" else (5 if opt == "golden_globe" else 15)
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(x_train, y_train)
    y_pred_knn = model_knn.predict(x_test)
    proba_knn = model_knn.predict_proba(x_test)

    acc_knn = accuracy_score(y_test, y_pred_knn)
    print(f"Accuracy: {acc_knn:.2%}")

    if len(set(y_test)) > 1 and len(set(y_pred_knn)) > 1:
        f1_knn = f1_score(y_test, y_pred_knn, average="binary", pos_label="Winner")
        precision_knn = precision_score(y_test, y_pred_knn, average="binary", pos_label="Winner")
        recall_knn = recall_score(y_test, y_pred_knn, average="binary", pos_label="Winner")
        print(f"Precision: {precision_knn:.2%}")
        print(f"Recall: {recall_knn:.2%}")
        print(f"F1 Score: {f1_knn:.2%}")

    all_predictions['KNN'] = {'pred': y_pred_knn, 'proba': proba_knn}

    # Random Forest
    print("\n" + "=" * 80)
    print("📊 RANDOM FOREST")
    print("=" * 80)
    min_leaf = 15 if opt == "oscars" else 3
    model_rf = RandomForestClassifier(
        criterion="entropy",
        random_state=80,
        min_samples_leaf=min_leaf,
        min_samples_split=2,
    )
    model_rf.fit(x_train, y_train)
    y_pred_rf = model_rf.predict(x_test)
    proba_rf = model_rf.predict_proba(x_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    print(f"Accuracy: {acc_rf:.2%}")

    if len(set(y_test)) > 1 and len(set(y_pred_rf)) > 1:
        f1_rf = f1_score(y_test, y_pred_rf, average="binary", pos_label="Winner")
        precision_rf = precision_score(y_test, y_pred_rf, average="binary", pos_label="Winner")
        recall_rf = recall_score(y_test, y_pred_rf, average="binary", pos_label="Winner")
        print(f"Precision: {precision_rf:.2%}")
        print(f"Recall: {recall_rf:.2%}")
        print(f"F1 Score: {f1_rf:.2%}")

    all_predictions['Random Forest'] = {'pred': y_pred_rf, 'proba': proba_rf}

    # Print detailed predictions
    print("\n" + "=" * 80)
    print("🎮 PREVISÕES DETALHADAS")
    print("=" * 80)

    for i, game in enumerate(test_games):
        actual = actual_results[i]
        symbol = "🏆" if actual == "Winner" else "  "

        print(f"\n{symbol} {game}")
        print(f"   Real: {actual}")

        for model_name, preds in all_predictions.items():
            pred = preds['pred'][i]
            proba = preds['proba'][i]

            # Get class labels
            if model_name == 'Naive Bayes':
                classes = model_nb.classes_
            elif model_name == 'KNN':
                classes = model_knn.classes_
            else:
                classes = model_rf.classes_

            winner_idx = list(classes).index('Winner') if 'Winner' in classes else 0
            winner_prob = proba[winner_idx] * 100

            status = "✓" if pred == actual else "✗"
            print(f"   {model_name:15s}: {pred:6s} ({winner_prob:.1f}% Winner) {status}")


if __name__ == "__main__":
    details = True
    # for name in ["oscars", "golden_globe", "golden_globe_comedy"]:
    for name in ["the_game_awards"]:
        print("\nResultados do " + name)
        # generateTrainAndTest(name)
        print("\nNaive Bayes")
        naiveBayes(name, details)
        print("\nKNN")
        knn(name, details)
        print("\nRandom Forest")
        randomForest(name, details)
