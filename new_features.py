import json
import statistics
import matplotlib.pyplot as plt
import numpy as np
import ast


def plotHistogram():
    with open('oscar_movies_statistics.json', 'r') as json_file:
        movies = json.load(json_file)
        # Create a figure and three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

        medias = []
        medianas = []
        modas = []

        for movie in movies:
            medias.append(movies[movie]['critic-mean'])
            medianas.append(movies[movie]['critic-median'])
            modas.append(movies[movie]['critic-stdev'])

        # Plot the histograms
        ax1.hist(medias, bins=20, color='blue')
        ax2.hist(medianas, bins=20, color='green')
        ax3.hist(modas, bins=20, color='red')

        # Add titles and axis labels
        ax1.set_title('Média')
        ax2.set_title('Mediana')
        ax3.set_title('Desvio padrão')
        ax1.set_xlabel('Valores')
        ax2.set_xlabel('Valores')
        ax3.set_xlabel('Valores')
        ax1.set_ylabel('Frequência')
        # Show the plot
        plt.show()


def plotHistogramByClass(class_):
    with open('oscar_movies_statistics.json', 'r') as json_file:
        movies = json.load(json_file)
        # Create a figure and three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))

        medias = []
        medianas = []
        modas = []

        for movie in movies:
            if movies[movie]['winner'] == class_:
                medias.append(movies[movie]['critic-mean'])
                medianas.append(movies[movie]['critic-median'])
                modas.append(movies[movie]['critic-stdev'])

        # Plot the histograms
        ax1.hist(medias, bins=20, color='blue')
        ax2.hist(medianas, bins=20, color='green')
        ax3.hist(modas, bins=20, color='red')

        # Add titles and axis labels
        ax1.set_title('Média')
        ax2.set_title('Mediana')
        ax3.set_title('Desvio padrão')
        ax1.set_xlabel('Valores')
        ax2.set_xlabel('Valores')
        ax3.set_xlabel('Valores')
        ax1.set_ylabel('Frequência')
        # Show the plot
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
    plotHistogramByClass("True")
