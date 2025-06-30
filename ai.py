import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def load_data():
    """Carrega os dados dos arquivos JSON."""
    with open("data/the_game_awards_data.json", "r") as f:
        historical_data = json.load(f)

    with open("data/the_game_awards_2024_data.json", "r") as f:
        data_2024 = json.load(f)

    return historical_data, data_2024


def extract_features(game_data):
    """Extrai características relevantes dos dados dos jogos."""
    features = []

    for game_key, game_info in game_data.items():
        # Extrair avaliações de usuários e converter para numérico
        user_reviews = game_info.get("user-reviews", [])
        if user_reviews:
            # Converter explicitamente para float
            user_reviews_numeric = [float(review) for review in user_reviews]
            user_avg = np.mean(user_reviews_numeric)
            user_std = (
                np.std(user_reviews_numeric) if len(user_reviews_numeric) > 1 else 0
            )
            user_count = len(user_reviews_numeric)
        else:
            user_avg = 5.0  # Valor padrão
            user_std = 0.0
            user_count = 0

        # Extrair avaliações de críticos e converter para numérico
        critic_reviews = game_info.get("critic-reviews", [])
        if critic_reviews:
            # Converter explicitamente para float
            critic_reviews_numeric = [float(review) for review in critic_reviews]
            critic_avg = (
                np.mean(critic_reviews_numeric) / 10.0
            )  # Normalizar para escala 0-10
            critic_std = (
                np.std(critic_reviews_numeric) / 10.0
                if len(critic_reviews_numeric) > 1
                else 0
            )
            critic_count = len(critic_reviews_numeric)
        else:
            critic_avg = 7.0  # Valor padrão
            critic_std = 0.0
            critic_count = 0

        # Extrair outras informações
        year = int(game_info.get("year", 0))
        rating_diff = critic_avg - user_avg

        # Feature para representar se tem ou não avaliações
        has_critic_reviews = 1 if critic_reviews else 0
        has_user_reviews = 1 if user_reviews else 0

        features.append(
            {
                "name": game_info.get("name", "Unknown"),
                "user_avg": user_avg,
                "user_std": user_std,
                "user_count": user_count,
                "critic_avg": critic_avg,
                "critic_std": critic_std,
                "critic_count": critic_count,
                "rating_diff": rating_diff,
                "has_critic_reviews": has_critic_reviews,
                "has_user_reviews": has_user_reviews,
                "year": year,
                "winner": 1 if game_info.get("winner", "FALSO") == "VERDADEIRO" else 0,
            }
        )

    return pd.DataFrame(features)


def train_models(df):
    """Treina os modelos de KNN, Naive Bayes e Random Forest."""
    # Separar features e target
    X = df.drop(["name", "winner", "year"], axis=1)
    y = df["winner"]

    # Dividir em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Normalizar as features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Definir e treinar modelos
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        # Treinar modelo
        model.fit(X_train_scaled, y_train)

        # Validação cruzada
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

        # Avaliação no conjunto de teste
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {
            "model": model,
            "cv_scores": cv_scores,
            "accuracy": accuracy,
            "report": classification_report(y_test, y_pred),
            "conf_matrix": confusion_matrix(y_test, y_pred),
        }

        print(f"\n{name} - Acurácia: {accuracy:.4f}")
        print(f"CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    return models, scaler, results, X_test, y_test


def predict_winners(models, scaler, df_2024):
    """Faz previsões para os jogos do ano atual."""
    # Preparar dados para previsão
    X_2024 = df_2024.drop(["name", "winner", "year"], axis=1)
    X_2024_scaled = scaler.transform(X_2024)

    # Fazer previsões com cada modelo
    predictions = {}
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_2024_scaled)
            win_probs = probs[:, 1]
        else:
            # Fallback para modelos sem predict_proba
            win_probs = model.predict(X_2024_scaled).astype(float)

        predictions[name] = win_probs

    # Organizar resultados
    results_df = pd.DataFrame({"Game": df_2024["name"]})
    for model_name, probs in predictions.items():
        results_df[model_name] = probs

    results_df["Average"] = results_df[[m for m in predictions.keys()]].mean(axis=1)
    results_df = results_df.sort_values("Average", ascending=False)

    return predictions, results_df


def plot_feature_importance(model, feature_names):
    """Plota a importância das features para o Random Forest."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()


def plot_predictions(predictions_df):
    """Visualiza as previsões para os jogos de 2024."""
    top_games = predictions_df.head(10)

    # Preparar dados para gráfico
    model_cols = [
        col for col in predictions_df.columns if col not in ["Game", "Average"]
    ]
    melted_df = pd.melt(
        top_games,
        id_vars=["Game"],
        value_vars=model_cols,
        var_name="Model",
        value_name="Probability",
    )

    plt.figure(figsize=(14, 8))
    sns.barplot(x="Game", y="Probability", hue="Model", data=melted_df)
    plt.title("Probabilidade de Vitória por Modelo (Top 10 Jogos)")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("prediction_comparison.png")
    plt.show()


def plot_rating_comparison(df):
    """Visualiza a relação entre avaliações de críticos e usuários."""
    plt.figure(figsize=(12, 8))

    sns.scatterplot(
        data=df,
        x="critic_avg",
        y="user_avg",
        hue="winner",
        size="critic_count",
        sizes=(20, 200),
        alpha=0.7,
    )

    plt.title("Relação entre Avaliações de Críticos e Usuários")
    plt.xlabel("Avaliação Média dos Críticos (0-10)")
    plt.ylabel("Avaliação Média dos Usuários (0-10)")

    # Adicionar linha onde críticos = usuários
    lims = [0, 10]
    plt.plot(lims, lims, "k--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("ratings_comparison.png")
    plt.show()


def plot_model_performance(results):
    """Visualiza o desempenho comparativo dos modelos."""
    models = list(results.keys())
    accuracies = [results[model]["accuracy"] for model in models]
    cv_scores = [results[model]["cv_scores"].mean() for model in models]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(models))
    width = 0.35

    plt.bar(x - width / 2, accuracies, width, label="Test Accuracy")
    plt.bar(x + width / 2, cv_scores, width, label="CV Accuracy")

    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Performance Comparison")
    plt.xticks(x, models)
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig("model_performance.png")
    plt.show()


def plot_rating_distributions(df):
    """Visualiza a distribuição das avaliações por status de vencedor."""
    plt.figure(figsize=(16, 8))

    # Distribuição das avaliações dos críticos
    plt.subplot(1, 2, 1)
    sns.histplot(
        data=df, x="critic_avg", hue="winner", kde=True, bins=15, element="step"
    )
    plt.title("Distribuição das Avaliações de Críticos")
    plt.xlabel("Média (0-10)")
    plt.legend(["Não Vencedor", "Vencedor"])

    # Distribuição das avaliações dos usuários
    plt.subplot(1, 2, 2)
    sns.histplot(data=df, x="user_avg", hue="winner", kde=True, bins=15, element="step")
    plt.title("Distribuição das Avaliações de Usuários")
    plt.xlabel("Média (0-10)")

    plt.tight_layout()
    plt.savefig("rating_distributions.png")
    plt.show()


def plot_confusion_matrices(results):
    """Plota as matrizes de confusão para todos os modelos."""
    models = list(results.keys())
    n_models = len(models)

    # Determinar o layout da figura (linhas e colunas)
    if n_models <= 3:
        rows, cols = 1, n_models
    else:
        rows = (n_models + 1) // 2
        cols = 2

    plt.figure(figsize=(15, 5 * rows))

    for i, model_name in enumerate(models):
        plt.subplot(rows, cols, i + 1)

        cm = results[model_name]["conf_matrix"]

        # Plotar a matriz de confusão usando seaborn
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Não Vencedor", "Vencedor"],
            yticklabels=["Não Vencedor", "Vencedor"],
        )

        plt.title(f"Matriz de Confusão - {model_name}")
        plt.ylabel("Valor Real")
        plt.xlabel("Valor Previsto")

    plt.tight_layout()
    plt.savefig("confusion_matrices.png")
    plt.show()


def main():
    # Carregar e processar dados
    historical_data, data_2024 = load_data()

    # Extrair features
    historical_df = extract_features(historical_data)
    df_2024 = extract_features(data_2024)

    # Visualizar distribuições e relações nos dados
    plot_rating_distributions(historical_df)
    plot_rating_comparison(historical_df)

    # Treinar modelos
    models, scaler, results, X_test, y_test = train_models(historical_df)

    # Visualizar desempenho dos modelos
    plot_model_performance(results)
    plot_confusion_matrices(results)

    # Visualizar importância das features (apenas Random Forest)
    plot_feature_importance(
        models["Random Forest"],
        historical_df.drop(["name", "winner", "year"], axis=1).columns,
    )

    # Fazer previsões para 2024
    predictions, prediction_df = predict_winners(models, scaler, df_2024)

    # Visualizar previsões
    plot_predictions(prediction_df)

    # Mostrar os 5 jogos com maior probabilidade de vitória
    print("\nTop 5 jogos com maior probabilidade de vitória:")
    print(prediction_df[["Game", "Average"]].head(5))

    return prediction_df


if __name__ == "__main__":
    results = main()
