from datetime import datetime
import time
from bs4 import BeautifulSoup
import csv
import requests
import re
import json
from typing import List, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

SEGUNDOS_PARA_ESPERAR = 1


class MetacriticScraper:
    """
    Classe responsável por buscar e processar avaliações de filmes no Metacritic.
    """

    BASE_URL = "https://www.metacritic.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.69"
            )
        }
        # Inicializa o driver do Selenium com opções headless
        options = Options()
        options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=options)

    def get_reviews(self, url: str, release_year: str, cerimony_date: str, critic: bool = False) -> List[str]:
        """
        Obtém as avaliações de um filme a partir de uma URL do Metacritic.

        Args:
            url (str): URL da página de avaliações do filme.
            release_year (str): Ano de lançamento do filme.
            cerimony_date (str): Data da cerimônia do Oscar.
            critic (bool): Indica se as avaliações são de críticos (True) ou usuários (False).

        Returns:
            List[str]: Lista de avaliações.
        """

        movie_details_url = url[:url.rfind("/")] + "/details"
        try:
            response = self.session.get(movie_details_url)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Erro ao acessar {movie_details_url}: {e}")

        soup = BeautifulSoup(response.content, "html.parser")
        release_date = self._extract_release_date(soup)

        if int(release_year) != release_date.year:
            raise ValueError(f"Anos diferentes: {release_year} vs {release_date.year} para {url}")

        reviews_list = []
        try:
            print(url)
            # Usar Selenium para carregar a página principal das reviews
            self.driver.get(url)
            # Pega a quantidade de reviews do produto
            quantidade_de_reviews = int(self.driver.find_element(value="c-pageProductReviews_text", by=By.CLASS_NAME).text.split(" ")[1].replace(",", ""))

            # Scroll down até o final para carregar todas as reviews
            for _ in range(quantidade_de_reviews // 50): # Cada página tem 50 reviews
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                # espera carregar a página
                time.sleep(SEGUNDOS_PARA_ESPERAR)

            soup = BeautifulSoup(self.driver.page_source, "html.parser")
        except Exception as e:
            raise RuntimeError(f"Erro ao acessar {url} via Selenium: {e}")

        reviews = self._extract_reviews(soup, cerimony_date, critic)
        reviews_list.extend(reviews)

        next_page_url = self._get_next_page_url(soup)
        if next_page_url:
            reviews_list.extend(self.get_reviews(next_page_url, release_year, cerimony_date, critic))

        return reviews_list

    def _extract_release_date(self, soup: BeautifulSoup) -> datetime:
        """
        Extrai a data de lançamento do filme.

        Args:
            soup (BeautifulSoup): Objeto BeautifulSoup da página.

        Returns:
            datetime: Data de lançamento.
        """
        try:
            date_str = (
                soup.find("div", class_="c-gameDetails_ReleaseDate")
                .find_all("span")[1]
                .get_text()
            )
            return datetime.strptime(date_str, "%b %d, %Y").date()
        except (AttributeError, IndexError, ValueError):
            raise RuntimeError("Não foi possível extrair a data de lançamento.")

    def _extract_reviews(self, soup: BeautifulSoup, cerimony_date: str, critic: bool) -> List[str]:
        """
        Extrai as avaliações da página.

        Args:
            soup (BeautifulSoup): Objeto BeautifulSoup da página.
            cerimony_date (str): Data da cerimônia do Oscar.
            critic (bool): Indica se as avaliações são de críticos (True) ou usuários (False).

        Returns:
            List[str]: Lista de avaliações.
        """
        reviews = []
        cerimony_date_obj = datetime.strptime(cerimony_date, "%d/%m/%Y").date()

        print(f"Found {len(soup.find_all('div', class_='c-siteReview'))} reviews")
        with open("soup.html", "w") as f:
            f.write(str(soup))

        for review in soup.find_all("div", class_="c-siteReview"):
            if "ad_unit" in review.get("class", []):
                continue
            try:
                score = review.find("div", class_="c-siteReviewScore").find("span").text
            except AttributeError:
                score = None
            if not score:  # score is not being found
                continue

            if not critic:
                date_str = review.find("div", class_="c-siteReviewHeader_reviewDate").text.strip()
                if date_str:
                    review_date = datetime.strptime(date_str, "%b %d, %Y").date()
                    if review_date > cerimony_date_obj:
                        continue

            reviews.append(score)

        return reviews

    def _get_next_page_url(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Verifica se existe uma próxima página de avaliações.

        Args:
            soup (BeautifulSoup): Objeto BeautifulSoup da página.

        Returns:
            Optional[str]: URL da próxima página, se existir.
        """
        next_page = soup.find("div", class_="page_flipper")
        if next_page:
            link = next_page.find("span", class_="flipper next")
            if link and link.find("a"):
                return self.BASE_URL + link.find("a")["href"]
        return None


def slugify_film_title(title: str) -> str:
    """
    Converte o título do filme em um slug apropriado para URLs.

    Args:
        title (str): Título do filme.

    Returns:
        str: Slug do filme.
    """
    return re.sub(r"[^a-zA-Z0-9\-]+", "", title.lower().replace(" ", "-"))


def process_reviews(scraper: MetacriticScraper, film: str, film_year: str, oscar_date: str, critic: bool) -> List[str]:
    """
    Processa as avaliações de um filme, tentando diferentes variações de slug.

    Args:
        scraper (MetacriticScraper): Instância do scraper.
        film (str): Nome do filme.
        film_year (str): Ano de lançamento do filme.
        oscar_date (str): Data da cerimônia do Oscar.
        critic (bool): Indica se são avaliações de críticos.

    Returns:
        List[str]: Lista de avaliações.
    """
    film_slug = slugify_film_title(film)
    paths = [
        f"/game/{film_slug}/{'critic-reviews' if critic else 'user-reviews'}?sort-by=date&num_items=100",
        f"/game/{film_slug}-{film_year}/{'critic-reviews' if critic else 'user-reviews'}?sort-by=date&num_items=100",
    ]

    for path in paths:
        try:
            return scraper.get_reviews(scraper.BASE_URL + path, film_year, oscar_date, critic)
        except Exception as e:
            print(f"Erro ao processar {film_slug}: {e}")

    print(f"{film_slug} não encontrado")
    return []


def main():
    """
    Função principal que processa o arquivo CSV e gera um arquivo JSON com os dados dos filmes.
    """
    scraper = MetacriticScraper()
    try:
        with open("./csv/the_game_awards.csv") as csv_file, open("the_game_awards_data.json", "w") as games_file:
            games_dict = {}
            csv_reader = csv.reader(csv_file, delimiter=";")
            next(csv_reader)  # Ignora o cabeçalho

            for row in csv_reader:
                print(f"Processando: {row[2]}")
                user_reviews = process_reviews(scraper, row[2], row[0], row[1], critic=False)
                critic_reviews = process_reviews(scraper, row[2], row[0], row[1], critic=True)
                games_dict[row[2]] = {
                    "user-reviews": user_reviews,
                    "critic-reviews": critic_reviews,
                    "year": row[0],
                    "cerimony-date": row[1],
                    "winner": row[3],
                }

            json.dump(games_dict, games_file, indent=4)
    except Exception as e:
        print(f"Erro na execução principal: {e}")


if __name__ == "__main__":
    main()
