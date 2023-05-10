import json
import statistics
import matplotlib.pyplot as plt
import numpy as np


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


def plotHistogramByClass(class_, type_):
    with open('golden_globe_movies_statistics.json', 'r') as json_file:
        movies = json.load(json_file)
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)
              ) = plt.subplots(2, 3, figsize=(10, 5))

        medias = []
        medianas = []
        stdev = []
        modas = []
        perc_25 = []
        perc_75 = []

        if type_ == 'critic':
            for movie in movies:
                if movies[movie]['class'] == class_:
                    medias.append(movies[movie]['critic-mean'])
                    medianas.append(movies[movie]['critic-median'])
                    stdev.append(movies[movie]['critic-stdev'])
                    modas.append(movies[movie]['critic-mode'])
                    perc_25.append(movies[movie]['critic-percentile-25'])
                    perc_75.append(movies[movie]['critic-percentile-75'])
        else:
            for movie in movies:
                if movies[movie]['class'] == class_:
                    medias.append(movies[movie]['user-mean'])
                    medianas.append(movies[movie]['user-median'])
                    stdev.append(movies[movie]['user-stdev'])
                    modas.append(movies[movie]['user-mode'])
                    perc_25.append(movies[movie]['user-percentile-25'])
                    perc_75.append(movies[movie]['user-percentile-75'])

        ax1.hist(medias, bins=20, color='blue')
        ax2.hist(medianas, bins=20, color='green')
        ax3.hist(stdev, bins=20, color='red')
        ax4.hist(modas, bins=20, color='yellow')
        ax5.hist(perc_25, bins=20, color='grey')
        ax6.hist(perc_75, bins=20, color='purple')

        ax1.set_title('Média')
        ax2.set_title('Mediana')
        ax3.set_title('Desvio padrão')
        ax4.set_title('Moda')
        ax5.set_title('Percentil 25%')
        ax6.set_title('Percentil 75%')
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


def plot3DScatter():
    with open('oscar_movies_statistics.json', 'r') as json_file:
        movies = json.load(json_file)

        fig = plt.figure()
        ax = plt.axes(projection='3d')

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

        ax.scatter3D(medias_critic[0], medianas_critic[0],
                     stdev_critic[0], label='Ganhador')
        ax.scatter3D(medias_critic[1], medianas_critic[1],
                     stdev_critic[1], label="Perdedor")

        # ax.scatter3D(medias_user[0], medianas_user[0],
        #              stdev_user[0], label='Ganhador')
        # ax.scatter3D(medias_user[1], medianas_user[1],
        #              stdev_user[1], label="Perdedor")

        ax.legend()

        ax.set_title('Críticos')
        # ax.set_title('Usuários')

        ax.set_xlabel('Média')
        ax.set_ylabel('Mediana')
        ax.set_zlabel('Desvio Padrão')

        plt.show()


def generateFeatures():
    with open('golden_globe_movies_2023_data.json', 'r') as json_file, open('golden_globe_movies_2023_statistics.json', 'w') as output_file:
        movies = json.load(json_file)
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

        json.dump(movies, output_file)


if __name__ == '__main__':
    plotHistogramByClass('Winner', 'user')
    # generateFeatures()
