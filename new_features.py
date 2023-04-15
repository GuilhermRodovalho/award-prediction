import json
import statistics

if __name__ == '__main__':
    with open('golden_globe_movies_data.json', 'r') as json_file, open('golden_globe_movies_statistics.json', 'w') as output_file:
        movies = json.load(json_file)
        for movie in movies:
            user_reviews = list(map(int, movies[movie]['user-review']))
            critic_review = list(map(int, movies[movie]['critic-review']))
            movies[movie]['user-mean'] = statistics.mean(user_reviews)
            movies[movie]['user-mode'] = statistics.mode(user_reviews)
            movies[movie]['user-median'] = statistics.median(user_reviews)
            movies[movie].pop('user-review')

            movies[movie]['critic-mean'] = statistics.mean(critic_review)
            movies[movie]['critic-mode'] = statistics.mode(critic_review)
            movies[movie]['critic-median'] = statistics.median(critic_review)
            movies[movie].pop('critic-review')

        json.dump(movies, output_file)
