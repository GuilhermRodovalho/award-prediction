import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
import ast


def plotHistogram():
    with open('oscar_movies_statistics.json', 'r') as json_file:
        movies = json.load(json_file)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

        medias = []
        medianas = []
        stdev = []

        for movie in movies:
            medias.append(movies[movie]['critic-mean'])
            medianas.append(movies[movie]['critic-median'])
            stdev.append(movies[movie]['critic-stdev'])

        ax1.hist(medias, bins=20, color='blue')
        ax2.hist(medianas, bins=20, color='green')
        ax3.hist(stdev, bins=20, color='red')

        ax1.set_title('Média')
        ax2.set_title('Mediana')
        ax3.set_title('Desvio padrão')
        ax1.set_xlabel('Valores')
        ax2.set_xlabel('Valores')
        ax3.set_xlabel('Valores')
        ax1.set_ylabel('Frequência')

        plt.show()


def plotHistogramByClass(class_):
    with open('oscar_movies_statistics.json', 'r') as json_file:
        movies = json.load(json_file)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

        medias = []
        medianas = []
        stdev = []

        for movie in movies:
            if movies[movie]['winner'] == class_:
                medias.append(movies[movie]['critic-mean'])
                medianas.append(movies[movie]['critic-median'])
                stdev.append(movies[movie]['critic-stdev'])

        ax1.hist(medias, bins=20, color='blue')
        ax2.hist(medianas, bins=20, color='green')
        ax3.hist(stdev, bins=20, color='red')

        ax1.set_title('Média')
        ax2.set_title('Mediana')
        ax3.set_title('Desvio padrão')
        ax1.set_xlabel('Valores')
        ax2.set_xlabel('Valores')
        ax3.set_xlabel('Valores')
        ax1.set_ylabel('Frequência')
        # Show the plot
        plt.show()


def plotScatter():
    with open('golden_globe_movies_statistics.json', 'r') as json_file:
        movies = json.load(json_file)
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(5, 10))

        medias_critic = ([], [])
        medianas_critic = ([], [])
        stdev_critic = ([], [])

        medias_user = ([], [])
        medianas_user = ([], [])
        stdev_user = ([], [])

        for movie in movies:
            if movies[movie]['winner'] == "True":
                medias_critic[0].append(movies[movie]['critic-mean'])
                medianas_critic[0].append(movies[movie]['critic-median'])
                stdev_critic[0].append(movies[movie]['critic-stdev'])

                medias_user[0].append(movies[movie]['user-mean'])
                medianas_user[0].append(movies[movie]['user-median'])
                stdev_user[0].append(movies[movie]['user-stdev'])
            else:
                medias_critic[1].append(movies[movie]['critic-mean'])
                medianas_critic[1].append(movies[movie]['critic-median'])
                stdev_critic[1].append(movies[movie]['critic-stdev'])

                medias_user[1].append(movies[movie]['user-mean'])
                medianas_user[1].append(movies[movie]['user-median'])
                stdev_user[1].append(movies[movie]['user-stdev'])

        ax1.scatter(medias_critic[0], medianas_critic[0], label='Ganhador')
        ax1.scatter(medias_critic[1], medianas_critic[1], label="Perdedor")

        ax2.scatter(medias_critic[0], stdev_critic[0])
        ax2.scatter(medias_critic[1], stdev_critic[1])

        ax3.scatter(medianas_critic[0], stdev_critic[0])
        ax3.scatter(medianas_critic[1], stdev_critic[1])

        # ax1.scatter(medias_user[0], medianas_user[0], label='Ganhador')
        # ax1.scatter(medias_user[1], medianas_user[1], label="Perdedor")

        # ax2.scatter(medias_user[0], stdev_user[0])
        # ax2.scatter(medias_user[1], stdev_user[1])

        # ax3.scatter(medianas_user[0], stdev_user[0])
        # ax3.scatter(medianas_user[1], stdev_user[1])

        ax1.legend()

        ax1.set_title('Críticos')
        # ax1.set_title('Usuários')

        ax1.set_xlabel('Média')
        ax1.set_ylabel('Mediana')

        ax2.set_xlabel('Média')
        ax2.set_ylabel('Desvio Padrão')

        ax3.set_xlabel('Mediana')
        ax3.set_ylabel('Desvio Padrão')

        plt.show()


def generateFeatures():
    with open('oscar_movies_data.json', 'r') as json_file, open('oscar_movies_statistics.json', 'w') as output_file:
        movies = json.load(json_file)
        for movie in movies:
            user_reviews = list(map(int, movies[movie]['user-review']))
            critic_review = list(map(int, movies[movie]['critic-review']))
            movies[movie]['user-mean'] = statistics.mean(user_reviews)
            movies[movie]['user-stdev'] = statistics.stdev(user_reviews)
            movies[movie]['user-median'] = statistics.median(user_reviews)
            movies[movie].pop('user-review')

            movies[movie]['critic-mean'] = statistics.mean(critic_review)
            movies[movie]['critic-stdev'] = statistics.stdev(critic_review)
            movies[movie]['critic-median'] = statistics.median(critic_review)
            movies[movie].pop('critic-review')

        json.dump(movies, output_file)


if __name__ == '__main__':
    plotScatter()
