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
    elif opt == "the_game_awards":
        filename = "the_game_awards_data.json"
    else:
        print("Erro: Opção de dataset inválida.")
        exit(1)

    filepath = "data/" + filename
    try:
        with open(filepath, "r") as json_file:
            movies_raw = json.load(json_file)
    except FileNotFoundError:
        print(f"Erro: Arquivo {filepath} não encontrado.")
        # Retornar DataFrames vazios estruturados para evitar crash massivo
        # Idealmente, as funções chamadoras deveriam esperar None ou levantar exceção
        X_empty_df = pd.DataFrame()
        y_empty_series = pd.Series(dtype="object")
        return X_empty_df, X_empty_df, y_empty_series, y_empty_series
    except json.JSONDecodeError:
        print(f"Erro: Falha ao decodificar JSON do arquivo {filepath}.")
        X_empty_df = pd.DataFrame()
        y_empty_series = pd.Series(dtype="object")
        return X_empty_df, X_empty_df, y_empty_series, y_empty_series

    if not movies_raw:
        print(
            f"Aviso: O arquivo JSON {filepath} está vazio ou não contém dados de filmes/jogos."
        )
        cols = [
            "user-mean",
            "user-stdev",
            "user-median",
            "user-mode",
            "user-percentile-25",
            "user-percentile-75",
            "critic-mean",
            "critic-stdev",
            "critic-median",
            "critic-mode",
            "critic-percentile-25",
            "critic-percentile-75",
        ]
        X_empty = pd.DataFrame(columns=cols)
        y_empty = pd.Series(name="class", dtype=str)
        dummy_y = (
            pd.Series([0] * len(X_empty), index=X_empty.index, name="class")
            if y_empty.empty
            else y_empty
        )  # type: ignore
        if X_empty.empty or X_empty.shape[1] == 0:
            X_empty = pd.DataFrame(
                {"dummy_feature": np.zeros(len(dummy_y))}, index=dummy_y.index
            )

        x_train, x_test, y_train, y_test = train_test_split(
            X_empty,
            dummy_y,
            test_size=0.2,
            random_state=80,
            stratify=dummy_y if len(dummy_y.unique()) > 1 else None,
        )
        return x_train, x_test, y_train, y_test

    stat_feature_names = [
        "user-mean",
        "user-stdev",
        "user-median",
        "user-mode",
        "user-percentile-25",
        "user-percentile-75",
        "critic-mean",
        "critic-stdev",
        "critic-median",
        "critic-mode",
        "critic-percentile-25",
        "critic-percentile-75",
    ]
    all_calculated_stats_collection = {key: [] for key in stat_feature_names}
    partially_processed_movies = {}

    for movie_id, movie_data_raw in movies_raw.items():
        current_movie = movie_data_raw.copy()
        current_movie.pop("year", None)
        current_movie.pop("cerimony-date", None)
        winner_status = current_movie.get("winner", "").lower()
        current_movie["class"] = (
            "Winner" if winner_status not in ["falso", "false"] else "Loser"
        )
        current_movie.pop("winner", None)

        user_reviews_str = current_movie.get("user-reviews", [])
        user_reviews = [
            int(r)
            for r in user_reviews_str
            if isinstance(r, (int, str)) and str(r).strip().isdigit()
        ]

        if user_reviews:
            mean_val = round(statistics.mean(user_reviews), 2)
            current_movie["user-mean"] = mean_val
            all_calculated_stats_collection["user-mean"].append(mean_val)
            current_movie["user-median"] = round(statistics.median(user_reviews), 2)
            all_calculated_stats_collection["user-median"].append(
                current_movie["user-median"]
            )
            try:
                current_movie["user-mode"] = round(statistics.mode(user_reviews), 2)
            except statistics.StatisticsError:
                current_movie["user-mode"] = mean_val
            all_calculated_stats_collection["user-mode"].append(
                current_movie["user-mode"]
            )
            current_movie["user-percentile-25"] = round(
                np.percentile(user_reviews, 25), 2
            )
            all_calculated_stats_collection["user-percentile-25"].append(
                current_movie["user-percentile-25"]
            )
            current_movie["user-percentile-75"] = round(
                np.percentile(user_reviews, 75), 2
            )
            all_calculated_stats_collection["user-percentile-75"].append(
                current_movie["user-percentile-75"]
            )
            current_movie["user-stdev"] = (
                round(statistics.stdev(user_reviews), 2)
                if len(user_reviews) >= 2
                else 0.0
            )
            all_calculated_stats_collection["user-stdev"].append(
                current_movie["user-stdev"]
            )
        else:
            for stat in [
                "user-mean",
                "user-stdev",
                "user-median",
                "user-mode",
                "user-percentile-25",
                "user-percentile-75",
            ]:
                current_movie[stat] = np.nan
        current_movie.pop("user-reviews", None)

        critic_reviews_str = current_movie.get("critic-reviews", [])
        critic_reviews_int = [
            int(r)
            for r in critic_reviews_str
            if isinstance(r, (int, str)) and str(r).strip().isdigit()
        ]
        critic_review_values = [x / 10.0 for x in critic_reviews_int]

        if critic_review_values:
            mean_val = round(statistics.mean(critic_review_values), 2)
            current_movie["critic-mean"] = mean_val
            all_calculated_stats_collection["critic-mean"].append(mean_val)
            current_movie["critic-median"] = round(
                statistics.median(critic_review_values), 2
            )
            all_calculated_stats_collection["critic-median"].append(
                current_movie["critic-median"]
            )
            try:
                current_movie["critic-mode"] = round(
                    statistics.mode(critic_review_values), 2
                )
            except statistics.StatisticsError:
                current_movie["critic-mode"] = mean_val
            all_calculated_stats_collection["critic-mode"].append(
                current_movie["critic-mode"]
            )
            current_movie["critic-percentile-25"] = round(
                np.percentile(critic_review_values, 25), 2
            )
            all_calculated_stats_collection["critic-percentile-25"].append(
                current_movie["critic-percentile-25"]
            )
            current_movie["critic-percentile-75"] = round(
                np.percentile(critic_review_values, 75), 2
            )
            all_calculated_stats_collection["critic-percentile-75"].append(
                current_movie["critic-percentile-75"]
            )
            current_movie["critic-stdev"] = (
                round(statistics.stdev(critic_review_values), 2)
                if len(critic_review_values) >= 2
                else 0.0
            )
            all_calculated_stats_collection["critic-stdev"].append(
                current_movie["critic-stdev"]
            )
        else:
            for stat in [
                "critic-mean",
                "critic-stdev",
                "critic-median",
                "critic-mode",
                "critic-percentile-25",
                "critic-percentile-75",
            ]:
                current_movie[stat] = np.nan
        current_movie.pop("critic-reviews", None)
        partially_processed_movies[movie_id] = current_movie

    global_fill_values = {}
    default_fill_value_for_globals = 0.0
    for stat_key, collected_values in all_calculated_stats_collection.items():
        if collected_values:
            strategy = (
                statistics.median
                if "mode" in stat_key or "median" in stat_key
                else statistics.mean
            )
            global_fill_values[stat_key] = round(strategy(collected_values), 2)
        else:
            global_fill_values[stat_key] = default_fill_value_for_globals

    final_processed_movies_dict = {}
    for movie_id, movie_data in partially_processed_movies.items():
        filled_movie_data = movie_data.copy()
        for stat_key in stat_feature_names:
            if pd.isna(
                filled_movie_data.get(stat_key)
            ):  # Use .get() in case a stat key was somehow missed
                filled_movie_data[stat_key] = global_fill_values[stat_key]
        final_processed_movies_dict[movie_id] = filled_movie_data

    df = pd.DataFrame.from_dict(final_processed_movies_dict, orient="index")

    # AQUI ESTÁ A MODIFICAÇÃO: Converter o índice (nomes dos jogos) em uma coluna chamada 'game_name'
    df = df.reset_index().rename(columns={"index": "game_name"})

    if df.empty:
        print("Erro: DataFrame vazio após o processamento dos dados (2).")
        # Retornar DataFrames vazios estruturados
        X_empty_df = pd.DataFrame(
            columns=stat_feature_names if stat_feature_names else ["dummy_feature"]
        )
        y_empty_series = pd.Series(name="class", dtype=str)
        # Garantir que os dataframes tenham pelo menos uma linha para train_test_split
        if X_empty_df.empty:
            X_empty_df = pd.DataFrame({"dummy_feature": [0]})
        if y_empty_series.empty:
            y_empty_series = pd.Series(["Loser"], name="class")

        # Preencher X_empty_df com uma linha de NaNs/zeros se estiver vazio mas tiver colunas
        if X_empty_df.shape[0] == 0 and not X_empty_df.columns.empty:
            X_empty_df = pd.DataFrame(
                np.nan, index=[0], columns=X_empty_df.columns
            ).fillna(0)
        elif (
            X_empty_df.shape[0] == 0 and X_empty_df.columns.empty
        ):  # Caso de X_empty_df realmente vazio
            X_empty_df = pd.DataFrame({"dummy_feature": [0]})
            if y_empty_series.empty:
                y_empty_series = pd.Series(
                    ["Loser"] * len(X_empty_df), name="class", index=X_empty_df.index
                )

        return train_test_split(
            X_empty_df,
            y_empty_series,
            test_size=0.2,
            random_state=80,
            stratify=y_empty_series if len(y_empty_series.unique()) > 1 else None,
        )

    # Modificar esta linha para excluir 'game_name' também
    X = df.drop(["class", "game_name"], axis=1, errors="ignore")
    y = df["class"]

    if X.empty or X.shape[1] == 0:
        print("Erro: Não há features (X) para treinar após o processamento.")
        X_dummy = pd.DataFrame(
            {"dummy_feature": np.zeros(len(y))}, index=y.index if not y.empty else None
        )
        y_dummy = y if not y.empty else pd.Series(dtype=str)
        if y_dummy.empty:
            y_dummy = pd.Series(
                ["Loser"] * len(X_dummy), name="class", index=X_dummy.index
            )

        return train_test_split(
            X_dummy,
            y_dummy,
            test_size=0.2,
            random_state=80,
            stratify=y_dummy
            if len(y_dummy.unique()) > 1 and len(y_dummy) > 1
            else None,
        )

    ros = RandomOverSampler(random_state=80)
    stratify_option = y if not y.empty and len(y.unique()) > 1 else None

    # Garantir que y não seja vazio para train_test_split
    if y.empty:
        print(
            "Aviso: Target 'y' está vazio antes do train_test_split. Retornando X original e y dummy."
        )
        # Criar um y dummy para evitar falha no train_test_split, embora isso não seja ideal para treinamento.
        y_dummy_tts = pd.Series(["Loser"] * len(X), index=X.index, name="class")
        # Não é possível estratificar com y dummy de uma classe
        x_train, x_test, y_train, y_test = train_test_split(
            X, y_dummy_tts, test_size=0.2, random_state=80, stratify=None
        )
        # Não tentar oversample com y_train dummy
        return x_train, x_test, y_train, y_test

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=stratify_option, random_state=80
    )

    # --- SEÇÃO PARA RANDOMOVERSAMPLER ---
    if y_train.empty or len(y_train.unique()) < 2:
        if y_train.empty:
            print(
                "Aviso: y_train está vazio após train_test_split. RandomOverSampler não será aplicado."
            )
        else:  # len(y_train.unique()) é 0 ou 1
            print(
                f"Aviso: y_train tem {len(y_train.unique())} classe(s) após train_test_split. RandomOverSampler não será aplicado."
            )
        x_resampled, y_resampled = x_train, y_train
    else:
        # y_train não está vazio e tem pelo menos 2 classes.
        # RandomOverSampler pode, em teoria, lidar com uma classe minoritária de 1 amostra.
        min_minority_samples_needed = 1  # Para RandomOverSampler.

        if x_train.empty:  # Checagem adicional para x_train vazio
            print(
                "Aviso: x_train está vazio. Não é possível fazer oversampling. Usando dados de treino originais."
            )
            x_resampled, y_resampled = x_train, y_train
        elif y_train.value_counts().min() < min_minority_samples_needed:
            # Esta condição só seria verdadeira se min_minority_samples_needed > 1 E a classe minoritária tivesse menos.
            # Ou se a classe minoritária tivesse 0 amostras (mas len(y_train.unique()) < 2 já teria coberto)
            print(
                f"Aviso: A classe minoritária em y_train tem {y_train.value_counts().min()} amostras, menos que o limiar de {min_minority_samples_needed}. RandomOverSampler não será aplicado."
            )
            x_resampled, y_resampled = x_train, y_train
        else:
            try:
                print(
                    f"Valores em y_train antes do RandomOverSampler: {y_train.value_counts().to_dict()}"
                )
                x_resampled, y_resampled = ros.fit_resample(x_train, y_train)
                print(
                    f"Valores em y_resampled depois do RandomOverSampler: {y_resampled.value_counts().to_dict()}"
                )
            except ValueError as e:
                print(
                    f"Erro (ValueError) durante fit_resample com RandomOverSampler: {e}. Usando dados de treino originais."
                )
                x_resampled, y_resampled = x_train, y_train
            except Exception as e:  # Captura genérica para outros problemas
                print(
                    f"Erro inesperado durante fit_resample com RandomOverSampler: {e}. Usando dados de treino originais."
                )
                x_resampled, y_resampled = x_train, y_train

    if not x_resampled.empty:
        # Converter DataFrame para numérico e substituir valores não-numéricos por 0
        for col in x_resampled.columns:
            try:
                x_resampled[col] = pd.to_numeric(x_resampled[col], errors="coerce")
                x_resampled[col] = x_resampled[col].fillna(0)
            except Exception as e:
                print(f"Erro ao converter coluna {col}: {e}")

        # Garantir que o índice seja numérico
        x_resampled = x_resampled.reset_index(drop=True)

    if not x_test.empty:
        # Converter DataFrame de teste para numérico também
        for col in x_test.columns:
            try:
                x_test[col] = pd.to_numeric(x_test[col], errors="coerce")
                x_test[col] = x_test[col].fillna(0)
            except Exception as e:
                print(f"Erro ao converter coluna de teste {col}: {e}")

        # Garantir que o índice seja numérico
        x_test = x_test.reset_index(drop=True)

    # Resetar índice das variáveis target também
    if not y_resampled.empty:
        y_resampled = y_resampled.reset_index(drop=True)

    if not y_test.empty:
        y_test = y_test.reset_index(drop=True)

    return x_resampled, x_test, y_resampled, y_test


def naiveBayes(opt, detail):
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    if (
        x_train.empty
        or x_test.empty
        or y_train.empty
        or y_test.empty
        or x_train.shape[0] < 1
        or x_test.shape[0] < 1
    ):
        print("NB: Dados insuficientes para treinar/testar. Pulando.")
        return

    model = GaussianNB()
    try:
        model.fit(x_train, y_train)
    except ValueError as e:
        print(f"NB: Erro ao treinar modelo: {e}. Pulando.")
        return

    y_pred = model.predict(x_test)

    accuray = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="binary", pos_label="Winner", zero_division=0)
    precision = precision_score(
        y_pred, y_test, average="binary", pos_label="Winner", zero_division=0
    )
    recall = recall_score(
        y_pred, y_test, average="binary", pos_label="Winner", zero_division=0
    )

    print("Accuracy:", accuray)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    if detail is True and not y_test.empty:
        labels = unique_labels(y_test, y_pred)
        if (
            not list(labels)
            or len(labels) < 2
            and not np.array_equal(np.unique(y_test), np.unique(y_pred))
        ):
            # Se labels for vazio ou tiver apenas uma classe mas y_test/y_pred discordam
            # isso pode indicar um problema (ex: todas as predições são uma classe, y_test tem outra)
            # No entanto, ConfusionMatrixDisplay pode lidar com isso se os labels corretos forem passados.
            # Usar os labels de y_test como fallback se a combinação for problemática.
            labels_for_cm = (
                np.unique(y_test)
                if len(np.unique(y_test)) >= 1
                else ["Loser", "Winner"]
            )
            labels = labels_for_cm  # type: ignore

        if not list(labels):  # Ainda vazio?
            print("NB: Não é possível gerar matriz de confusão (labels vazios).")
            return

        try:
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot()
            plt.suptitle(opt + " naive bayes")
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_test)
                proba_df_cols = (
                    labels if len(labels) == proba.shape[1] else model.classes_
                )  # type: ignore
                if (
                    len(proba_df_cols) != proba.shape[1]
                ):  # Fallback se model.classes_ também não bater
                    proba_df_cols = [f"Class_{i}" for i in range(proba.shape[1])]

                proba_y = pd.concat(
                    [
                        pd.DataFrame(
                            index=x_test.index.values, data=proba, columns=proba_df_cols
                        ),
                        y_test.reset_index(drop=True),
                    ],
                    axis=1,
                )
                print(proba_y)
        except Exception as e:
            print(f"NB: Erro ao gerar detalhes (matriz/proba): {e}")


def knn(opt, detail):
    k = 0
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    if (
        x_train.empty
        or x_test.empty
        or y_train.empty
        or y_test.empty
        or x_train.shape[0] < 1
        or x_test.shape[0] < 1
    ):
        print("KNN: Dados insuficientes para treinar/testar. Pulando.")
        return

    if opt == "oscars":
        k = 14
    elif opt == "golden_globe":
        k = 5
    else:
        k = 15

    if k > x_train.shape[0]:
        k = max(1, x_train.shape[0])
        print(f"KNN: Aviso: k ajustado para {k}.")

    if x_train.shape[0] < k:
        print(f"KNN: n_samples ({x_train.shape[0]}) < n_neighbors ({k}). Pulando.")
        return

    model = KNeighborsClassifier(n_neighbors=k)
    try:
        model.fit(x_train, y_train)
    except ValueError as e:
        print(f"KNN: Erro ao treinar: {e}. Pulando.")
        return
    y_pred = model.predict(x_test)

    accuray = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="binary", pos_label="Winner", zero_division=0)
    precision = precision_score(
        y_pred, y_test, average="binary", pos_label="Winner", zero_division=0
    )
    recall = recall_score(
        y_pred, y_test, average="binary", pos_label="Winner", zero_division=0
    )

    print("Accuracy:", accuray)  # type: ignore
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    if detail is True and not y_test.empty:
        labels = unique_labels(y_test, y_pred)
        if (
            not list(labels)
            or len(labels) < 2
            and not np.array_equal(np.unique(y_test), np.unique(y_pred))
        ):
            labels_for_cm = (
                np.unique(y_test)
                if len(np.unique(y_test)) >= 1
                else ["Loser", "Winner"]
            )
            labels = labels_for_cm  # type: ignore

        if not list(labels):
            print("KNN: Não é possível gerar matriz de confusão (labels vazios).")
            return
        try:
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot()
            plt.suptitle(opt + " knn")
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_test)
                proba_df_cols = (
                    labels if len(labels) == proba.shape[1] else model.classes_
                )  # type: ignore
                if len(proba_df_cols) != proba.shape[1]:
                    proba_df_cols = [f"Class_{i}" for i in range(proba.shape[1])]

                proba_y = pd.concat(
                    [
                        pd.DataFrame(
                            index=x_test.index.values, data=proba, columns=proba_df_cols
                        ),
                        y_test.reset_index(drop=True),
                    ],
                    axis=1,
                )
                print(proba_y)
        except Exception as e:
            print(f"KNN: Erro ao gerar detalhes (matriz/proba): {e}")


def randomForest(opt, detail):
    x_train, x_test, y_train, y_test = generateTrainAndTest(opt)

    if (
        x_train.empty
        or x_test.empty
        or y_train.empty
        or y_test.empty
        or x_train.shape[0] < 1
        or x_test.shape[0] < 1
    ):
        print("RF: Dados insuficientes para treinar/testar. Pulando.")
        return

    min_leaf = 15 if opt == "oscars" else 3

    if min_leaf > x_train.shape[0] // 2 and x_train.shape[0] > 1:
        old_min_leaf = min_leaf
        min_leaf = max(1, x_train.shape[0] // 2)
        print(
            f"RF: Aviso: min_samples_leaf ajustado de {old_min_leaf} para {min_leaf}."
        )
    elif x_train.shape[0] <= 1:
        min_leaf = 1

    min_split = max(2, min_leaf * 2)
    if (
        min_split > x_train.shape[0] and x_train.shape[0] >= 2
    ):  # min_samples_split não pode ser > n_samples
        min_split = x_train.shape[0]
    elif (
        x_train.shape[0] < 2
    ):  # Se só 1 amostra, min_split deve ser 1, embora RF precise de mais.
        min_split = 1  # RF provavelmente falhará ou terá desempenho ruim.

    model = RandomForestClassifier(
        criterion="entropy",
        random_state=80,
        min_samples_leaf=min_leaf,
        min_samples_split=min_split,
    )

    if (
        x_train.shape[0] < model.min_samples_split
        or x_train.shape[0] < model.min_samples_leaf
        or x_train.shape[0] < 2
    ):  # RF geralmente precisa de pelo menos 2 amostras.
        print(
            f"RF: dataset de treino muito pequeno ({x_train.shape[0]} amostras) para os parâmetros. Pulando."
        )
        return

    try:
        model.fit(x_train, y_train)
    except ValueError as e:
        print(f"RF: Erro ao treinar modelo: {e}. Pulando.")
        return
    y_pred = model.predict(x_test)

    accuray = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average="binary", pos_label="Winner", zero_division=0)
    precision = precision_score(
        y_pred, y_test, average="binary", pos_label="Winner", zero_division=0
    )
    recall = recall_score(
        y_pred, y_test, average="binary", pos_label="Winner", zero_division=0
    )

    print("Accuracy:", accuray)  # type: ignore
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    if detail is True and not y_test.empty:
        labels = unique_labels(y_test, y_pred)
        if (
            not list(labels)
            or len(labels) < 2
            and not np.array_equal(np.unique(y_test), np.unique(y_pred))
        ):
            labels_for_cm = (
                np.unique(y_test)
                if len(np.unique(y_test)) >= 1
                else ["Loser", "Winner"]
            )
            labels = labels_for_cm  # type: ignore

        if not list(labels):
            print("RF: Não é possível gerar matriz de confusão (labels vazios).")
            return
        try:
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
            disp.plot()
            plt.suptitle(opt + " random forest")
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_test)
                proba_df_cols = (
                    labels if len(labels) == proba.shape[1] else model.classes_
                )  # type: ignore
                if (
                    hasattr(model, "classes_") and len(model.classes_) != proba.shape[1]
                ):  # Fallback
                    proba_df_cols = [f"Class_{i}" for i in range(proba.shape[1])]
                elif not hasattr(model, "classes_") and len(labels) != proba.shape[1]:
                    proba_df_cols = [f"Class_{i}" for i in range(proba.shape[1])]

                proba_y = pd.concat(
                    [
                        pd.DataFrame(
                            index=x_test.index.values, data=proba, columns=proba_df_cols
                        ),
                        y_test.reset_index(drop=True),
                    ],
                    axis=1,
                )
                print(proba_y)
        except Exception as e:
            print(f"RF: Erro ao gerar detalhes (matriz/proba): {e}")


def run_classifier(model_func, name, detail, classifier_name):
    print(f"\n--- {classifier_name} para {name} ---")
    try:
        model_func(name, detail)
    except ValueError as e:
        print(f"FALHA (ValueError) ao executar {classifier_name} para {name}: {e}")
    except Exception as e:
        import traceback

        print(f"Um ERRO INESPERADO ocorreu com {classifier_name} para {name}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    details = True
    datasets = ["the_game_awards"]

    for name in datasets:
        print(
            f"\n=================================================\nPROCESSANDO DATASET: {name.upper()}\n================================================="
        )
        run_classifier(naiveBayes, name, details, "Naive Bayes")
        input()
        run_classifier(knn, name, details, "KNN")
        input()
        run_classifier(randomForest, name, details, "Random Forest")
        input()
