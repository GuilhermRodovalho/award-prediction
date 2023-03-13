# main.py

from datetime import datetime
from bs4 import BeautifulSoup
import requests
import re
import sys


def get_reviews(url, release_year, cerimony_year):
    reviews_list = {"reviews": [
    ], 'url': 'url'}
    session = requests.Session()
    session.headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.69'}
    res = session.get(url)
    soup = BeautifulSoup(res.content, 'html.parser')
    date_str = soup.find("span", class_='release_date').contents[3].get_text()
    date_obj = date_obj = datetime.strptime(date_str, '%B %d, %Y').date()
    if (release_year != date_obj.year):
        raise Exception("Anos diferentes - "+url)
    reviews = soup.find("div", class_='user_reviews').find_all(
        'div', class_='review')
    count = 0
    for review in reviews:
        review_date_str = review.find('span', class_='date').get_text()
        review_date_obj = datetime.strptime(
            review_date_str, '%b %d, %Y').date()
        if (cerimony_year == review_date_obj.year):
            count += 1
    next_page = soup.find('div', class_='page_flipper').find(
        'span', class_='flipper next').find('a')
    if next_page:
        next_page_url = 'https://www.metacritic.com' + next_page['href']
        reviews_list['reviews'].extend(get_reviews(
            next_page_url, release_year, cerimony_year)['reviews'])
    return reviews_list


if __name__ == '__main__':
    second_attemp = False

    filename = "log.txt"
    file_obj = open(filename, "w")
    sys.stdout = file_obj

    film = '@\'Whiplash'
    film_string = film.lower().replace(" ", "-")
    film_string = re.sub(r'[^a-zA-Z0-9\-]+', '', film_string)
    try:
        get_reviews(
            f'https://www.metacritic.com/movie/{film_string}/user-reviews?sort-by=date&num_items=100', 2014, 2015)
        print(film_string + " adicionado com sucesso")
    except AttributeError as e:
        try:
            print(film_string+' não encontrado, tentando adicionar o ano ao final')
            film_string = film_string + "-" + '2014'
            get_reviews(
                f'https://www.metacritic.com/movie/{film_string}/user-reviews?sort-by=date&num_items=100', 2014, 2015)
        except Exception as e:
            print(film_string+' não encontrado')
    except Exception as e:
        print(film_string+' com ano incorreto, tentando novamente')
    finally:
        file_obj.close()
