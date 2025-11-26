#!/usr/bin/env python3
"""
Script para predição de Game Awards comparando diferentes tipos de features.

Este script treina modelos separadamente usando:
1. Apenas features de críticas
2. Apenas features de usuários
3. Combinação de ambas

Salva matrizes de confusão e logs em pastas separadas.
"""

import pandas as pd
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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
from datetime import datetime


def setup_results_directories(feature_type):
    """
    Cria estrutura de diretórios para salvar resultados.

    Args:
        feature_type: Tipo de feature ('criticas', 'usuarios', 'combinacao')
    """
    base_dir = f"resultados/{feature_type}"
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def filter_features_by_type(df, feature_type):
    """
    Filtra features do dataframe baseado no tipo.

    Args:
        df: DataFrame com todas as features
        feature_type: 'criticas', 'usuarios', 'combinacao', ou 'ponderado_90_10'

    Returns:
        DataFrame filtrado ou ponderado
    """
    if feature_type == "criticas":
        # Apenas features de críticas
        critic_cols = [col for col in df.columns if col.startswith("critic-")]
        return df[critic_cols]
    elif feature_type == "usuarios":
        # Apenas features de usuários
        user_cols = [col for col in df.columns if col.startswith("user-")]
        return df[user_cols]
    elif feature_type == "ponderado_90_10":
        # Ponderação: 90% críticas, 10% usuários
        # Cria novas features ponderadas
        df_weighted = pd.DataFrame(index=df.index)

        # Aplica peso de 0.9 nas features de críticas
        critic_cols = [col for col in df.columns if col.startswith("critic-")]
        for col in critic_cols:
            df_weighted[col] = df[col] * 0.9

        # Aplica peso de 0.1 nas features de usuários
        user_cols = [col for col in df.columns if col.startswith("user-")]
        for col in user_cols:
            df_weighted[col] = df[col] * 0.1

        return df_weighted
    else:  # combinacao
        # Todas as features
        return df


def generateTrainAndTestTemporal(opt, train_years, test_years, feature_type):
    """
    Split data by year and filter by feature type.

    Args:
        opt: Dataset name
        train_years: List of years for training
        test_years: List of years for testing
        feature_type: 'criticas', 'usuarios', 'combinacao', ou 'ponderado_90_10'
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
            yearly_averages[year] = 5.0

    # Second pass: process movies
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
            if movie_year in yearly_averages:
                avg_rating = yearly_averages[movie_year]
            else:
                avg_rating = (
                    sum(yearly_averages.values()) / len(yearly_averages)
                    if yearly_averages
                    else 5.0
                )
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

    # Filter features based on type
    features_df = filter_features_by_type(df.drop("class", axis=1), feature_type)

    # Split by year
    train_mask = df.index.map(lambda x: movie_years.get(x, "unknown") in train_years)
    test_mask = df.index.map(lambda x: movie_years.get(x, "unknown") in test_years)

    x_train = features_df[train_mask]
    y_train = df[train_mask]["class"]
    x_test = features_df[test_mask]
    y_test = df[test_mask]["class"]

    # Apply RandomOverSampler to training data
    ros = RandomOverSampler(random_state=80)
    x_resampled, y_resampled = ros.fit_resample(x_train, y_train)

    return x_resampled, x_test, y_resampled, y_test


def save_confusion_matrix(y_test, y_pred, model_name, output_dir):
    """
    Salva matriz de confusão como imagem.

    Args:
        y_test: Valores reais
        y_pred: Valores preditos
        model_name: Nome do modelo
        output_dir: Diretório de saída
    """
    from sklearn.discriminant_analysis import unique_labels

    labels = unique_labels(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_dir, f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

    return filepath


def predictFutureYearsComparative(opt, train_years, test_years, feature_type):
    """
    Treina e avalia modelos para um tipo específico de feature.

    Args:
        opt: Nome do dataset
        train_years: Anos para treinamento
        test_years: Anos para teste
        feature_type: 'criticas', 'usuarios', ou 'combinacao'
    """
    # Setup output directory
    output_dir = setup_results_directories(feature_type)

    # Open log file
    log_file = os.path.join(output_dir, 'metrics_log.txt')

    with open(log_file, 'w', encoding='utf-8') as log:
        # Header
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log.write("=" * 80 + "\n")
        log.write(f"🎮 THE GAME AWARDS - ANÁLISE: {feature_type.upper()}\n")
        log.write(f"📅 Executado em: {timestamp}\n")
        log.write("=" * 80 + "\n\n")

        log.write(f"📚 Anos de treinamento: {train_years[0]}-{train_years[-1]}\n")
        log.write(f"🔮 Anos de teste: {', '.join(test_years)}\n\n")

        # Generate data
        x_train, x_test, y_train, y_test = generateTrainAndTestTemporal(
            opt, train_years, test_years, feature_type
        )

        # Log data split info
        log.write("=" * 80 + "\n")
        log.write("📊 DIVISÃO DOS DADOS\n")
        log.write("=" * 80 + "\n")
        log.write(f"TREINO: {len(x_train)} jogos\n")
        log.write(f"  - Winners: {(y_train == 'Winner').sum()}\n")
        log.write(f"  - Losers: {(y_train == 'Loser').sum()}\n")
        log.write(f"TESTE: {len(x_test)} jogos\n")
        log.write(f"  - Winners: {(y_test == 'Winner').sum()}\n")
        log.write(f"  - Losers: {(y_test == 'Loser').sum()}\n")
        log.write(f"Features utilizadas: {list(x_train.columns)}\n\n")

        # Store test games
        test_games = x_test.index.values
        actual_results = y_test.values
        all_predictions = {}

        # ===== NAIVE BAYES =====
        log.write("=" * 80 + "\n")
        log.write("📊 NAIVE BAYES\n")
        log.write("=" * 80 + "\n")

        model_nb = GaussianNB()
        model_nb.fit(x_train, y_train)
        y_pred_nb = model_nb.predict(x_test)
        proba_nb = model_nb.predict_proba(x_test)

        acc_nb = accuracy_score(y_test, y_pred_nb)
        log.write(f"Accuracy: {acc_nb:.4f} ({acc_nb:.2%})\n")

        if len(set(y_test)) > 1 and len(set(y_pred_nb)) > 1:
            f1_nb = f1_score(y_test, y_pred_nb, average="binary", pos_label="Winner")
            precision_nb = precision_score(y_test, y_pred_nb, average="binary", pos_label="Winner")
            recall_nb = recall_score(y_test, y_pred_nb, average="binary", pos_label="Winner")
            log.write(f"Precision: {precision_nb:.4f} ({precision_nb:.2%})\n")
            log.write(f"Recall: {recall_nb:.4f} ({recall_nb:.2%})\n")
            log.write(f"F1 Score: {f1_nb:.4f} ({f1_nb:.2%})\n")

        # Save confusion matrix
        cm_path = save_confusion_matrix(y_test, y_pred_nb, "Naive Bayes", output_dir)
        log.write(f"Matriz de confusão salva em: {cm_path}\n\n")

        all_predictions['Naive Bayes'] = {'pred': y_pred_nb, 'proba': proba_nb, 'model': model_nb}

        # ===== KNN =====
        log.write("=" * 80 + "\n")
        log.write("📊 K-NEAREST NEIGHBORS\n")
        log.write("=" * 80 + "\n")

        k = 14 if opt == "oscars" else (5 if opt == "golden_globe" else 15)
        log.write(f"K = {k}\n")

        model_knn = KNeighborsClassifier(n_neighbors=k)
        model_knn.fit(x_train, y_train)
        y_pred_knn = model_knn.predict(x_test)
        proba_knn = model_knn.predict_proba(x_test)

        acc_knn = accuracy_score(y_test, y_pred_knn)
        log.write(f"Accuracy: {acc_knn:.4f} ({acc_knn:.2%})\n")

        if len(set(y_test)) > 1 and len(set(y_pred_knn)) > 1:
            f1_knn = f1_score(y_test, y_pred_knn, average="binary", pos_label="Winner")
            precision_knn = precision_score(y_test, y_pred_knn, average="binary", pos_label="Winner")
            recall_knn = recall_score(y_test, y_pred_knn, average="binary", pos_label="Winner")
            log.write(f"Precision: {precision_knn:.4f} ({precision_knn:.2%})\n")
            log.write(f"Recall: {recall_knn:.4f} ({recall_knn:.2%})\n")
            log.write(f"F1 Score: {f1_knn:.4f} ({f1_knn:.2%})\n")

        cm_path = save_confusion_matrix(y_test, y_pred_knn, "KNN", output_dir)
        log.write(f"Matriz de confusão salva em: {cm_path}\n\n")

        all_predictions['KNN'] = {'pred': y_pred_knn, 'proba': proba_knn, 'model': model_knn}

        # ===== RANDOM FOREST =====
        log.write("=" * 80 + "\n")
        log.write("📊 RANDOM FOREST\n")
        log.write("=" * 80 + "\n")

        min_leaf = 15 if opt == "oscars" else 3
        log.write(f"min_samples_leaf = {min_leaf}\n")

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
        log.write(f"Accuracy: {acc_rf:.4f} ({acc_rf:.2%})\n")

        if len(set(y_test)) > 1 and len(set(y_pred_rf)) > 1:
            f1_rf = f1_score(y_test, y_pred_rf, average="binary", pos_label="Winner")
            precision_rf = precision_score(y_test, y_pred_rf, average="binary", pos_label="Winner")
            recall_rf = recall_score(y_test, y_pred_rf, average="binary", pos_label="Winner")
            log.write(f"Precision: {precision_rf:.4f} ({precision_rf:.2%})\n")
            log.write(f"Recall: {recall_rf:.4f} ({recall_rf:.2%})\n")
            log.write(f"F1 Score: {f1_rf:.4f} ({f1_rf:.2%})\n")

        cm_path = save_confusion_matrix(y_test, y_pred_rf, "Random Forest", output_dir)
        log.write(f"Matriz de confusão salva em: {cm_path}\n\n")

        all_predictions['Random Forest'] = {'pred': y_pred_rf, 'proba': proba_rf, 'model': model_rf}

        # ===== PREVISÕES DETALHADAS =====
        log.write("=" * 80 + "\n")
        log.write("🎮 PREVISÕES DETALHADAS POR JOGO\n")
        log.write("=" * 80 + "\n\n")

        for i, game in enumerate(test_games):
            actual = actual_results[i]
            symbol = "🏆" if actual == "Winner" else "  "

            log.write(f"{symbol} {game}\n")
            log.write(f"   Real: {actual}\n")

            for model_name, preds in all_predictions.items():
                pred = preds['pred'][i]
                proba = preds['proba'][i]
                classes = preds['model'].classes_

                winner_idx = list(classes).index('Winner') if 'Winner' in classes else 0
                winner_prob = proba[winner_idx] * 100

                status = "✓" if pred == actual else "✗"
                log.write(f"   {model_name:15s}: {pred:6s} ({winner_prob:5.1f}% Winner) {status}\n")

            log.write("\n")

        # Summary
        log.write("=" * 80 + "\n")
        log.write("📈 RESUMO COMPARATIVO\n")
        log.write("=" * 80 + "\n")
        log.write(f"Naive Bayes  - Accuracy: {acc_nb:.2%}\n")
        log.write(f"KNN          - Accuracy: {acc_knn:.2%}\n")
        log.write(f"Random Forest- Accuracy: {acc_rf:.2%}\n")
        log.write("=" * 80 + "\n")

    print(f"\n✅ Resultados salvos em: {output_dir}/")
    print(f"   - Log de métricas: metrics_log.txt")
    print(f"   - Matrizes de confusão: *_confusion_matrix.png")


if __name__ == "__main__":
    # Define years
    train_years = [str(year) for year in range(2014, 2023)]  # 2014-2022
    test_years = ['2023', '2024']

    print("\n" + "=" * 80)
    print("🎮 THE GAME AWARDS - ANÁLISE COMPARATIVA")
    print("=" * 80)
    print(f"\n📚 Treinando com dados históricos: {train_years[0]}-{train_years[-1]}")
    print(f"🔮 Testando/prevendo anos: {', '.join(test_years)}")
    print("\n" + "=" * 80)

    # Run for each feature type
    feature_types = [
        ("criticas", "apenas avaliações de críticos"),
        ("usuarios", "apenas avaliações de usuários"),
        ("combinacao", "combinação de críticos e usuários"),
        ("ponderado_90_10", "ponderado 90% críticos + 10% usuários (oficial TGA)")
    ]

    for feature_type, description in feature_types:
        print(f"\n{'=' * 80}")
        print(f"🔬 Executando análise: {description.upper()}")
        print(f"{'=' * 80}")

        predictFutureYearsComparative("the_game_awards", train_years, test_years, feature_type)

    print("\n" + "=" * 80)
    print("✅ ANÁLISE COMPLETA!")
    print("=" * 80)
    print("\n📁 Resultados organizados em:")
    print("   - resultados/criticas/")
    print("   - resultados/usuarios/")
    print("   - resultados/combinacao/")
    print("   - resultados/ponderado_90_10/")
    print("\n")
