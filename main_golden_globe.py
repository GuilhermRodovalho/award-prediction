# main.py

from datetime import datetime
from bs4 import BeautifulSoup
import csv
import requests
import re
import sys
import json


def get_reviews(url, release_year, cerimony_date, critic=False):
    reviews_list = []
    session = requests.Session()
    session.headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.69'}
    res = session.get(url)
    soup = BeautifulSoup(res.content, 'html.parser')
    date_str = soup.find("span", class_='release_date').contents[3].get_text()
    date_obj = date_obj = datetime.strptime(date_str, '%B %d, %Y').date()
    if (int(release_year) != date_obj.year):
        raise AssertionError("Anos diferentes - "+url)
    if (critic):
        reviews = soup.find("div", class_='critic_reviews').find_all(
            'div', class_='review')
    else:
        reviews = soup.find("div", class_='user_reviews').find_all(
            'div', class_='review')
    for review in reviews:
        if ('ad_unit' in review.get('class')):
            continue
        review_score = review.find('div', class_='metascore_w').get_text()
        if (critic):
            reviews_list.append(review_score)
        else:
            review_date_str = review.find('span', class_='date').get_text()
            review_date_obj = datetime.strptime(
                review_date_str, '%b %d, %Y').date()
            if (cerimony_date > review_date_obj.strftime("%d/%m/%Y")):
                reviews_list.append(review_score)
    try:
        next_page_flipper = soup.find('div', class_='page_flipper')
        if next_page_flipper:
            next_page = next_page_flipper.find(
                'span', class_='flipper next').find('a')
            if next_page:
                next_page_url = 'https://www.metacritic.com' + \
                    next_page['href']
                reviews_list.extend(get_reviews(
                    next_page_url, release_year, cerimony_date, critic))
        return reviews_list
    except Exception as e:
        pass
    finally:
        pass


def processaArquivoUserReview(film, film_year, oscar_date):
    film_string = film.lower().replace(" ", "-")
    film_string = re.sub(r'[^a-zA-Z0-9\-]+', '', film_string)
    try:
        return get_reviews(
            f'https://www.metacritic.com/movie/{film_string}/user-reviews?sort-by=date&num_items=100', film_year, oscar_date)
    except Exception as e:
        try:
            film_string = film_string + "-" + film_year
            return get_reviews(
                f'https://www.metacritic.com/movie/{film_string}/user-reviews?sort-by=date&num_items=100', film_year, oscar_date)
        except Exception as e:
            print(film_string+' não encontrado')


def processaArquivoCriticReview(film, film_year, oscar_date):
    film_string = film.lower().replace(" ", "-")
    film_string = re.sub(r'[^a-zA-Z0-9\-]+', '', film_string)
    try:
        return get_reviews(
            f'https://www.metacritic.com/movie/{film_string}/critic-reviews?sort-by=date&num_items=100', film_year, oscar_date, True)
    except Exception as e:
        try:
            film_string = film_string + "-" + film_year
            return get_reviews(
                f'https://www.metacritic.com/movie/{film_string}/critic-reviews?sort-by=date&num_items=100', film_year, oscar_date, True)
        except Exception as e:
            print(film_string+' não encontrado')


if __name__ == '__main__':
    with open('golden_globe_2023.csv') as csv_file, open('golden_globe_movies_2023_data.json', 'w') as movies_file:
        movies_dict = dict()
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        try:
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    print("Processing: " + row[2])
                    user = processaArquivoUserReview(row[2], row[0], row[1])
                    critic = processaArquivoCriticReview(
                        row[2], row[0], row[1])
                    movies_dict[row[2]] = {
                        'user-review': user, 'critic-review': critic, 'year': row[0], 'cerimony-date': row[1], 'winner': row[3]}
            json.dump(movies_dict, movies_file)
        except Exception as e:
            print(e)
