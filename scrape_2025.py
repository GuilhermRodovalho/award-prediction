from datetime import date, datetime
from bs4 import BeautifulSoup
import csv
import requests
import re
import json
from typing import List, Optional, Dict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import os

class MetacriticScraper:
    """
    Classe responsável por buscar e processar avaliações de itens no Metacritic.
    """

    BASE_URL = "https://www.metacritic.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
            )
        }
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("log-level=3")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"user-agent={self.session.headers['User-Agent']}")

        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(45)

    def _get_available_platforms(self, game_base_url: str) -> List[str]:
        """
        Extrai os slugs das plataformas disponíveis para um jogo.
        """
        platform_discovery_url = f"{game_base_url}/user-reviews/"
        platforms: List[str] = []
        print(f"Buscando plataformas em: {platform_discovery_url}")

        try:
            self.driver.get(platform_discovery_url)

            WebDriverWait(self.driver, 15).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "select[name='Platforms']")
                )
            )

            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")

            platform_select_tag = soup.find("select", {"name": "Platforms"})
            if platform_select_tag:
                for option in platform_select_tag.find_all("option"):
                    value = option.get("value")
                    if value and value.strip():
                        platforms.append(value)

                if platforms:
                    print(f"Plataformas encontradas: {platforms}")
                    return platforms

        except TimeoutException:
            try:
                page_source_on_timeout = self.driver.page_source
                soup_on_timeout = BeautifulSoup(page_source_on_timeout, "html.parser")
                single_platform_indicator = soup_on_timeout.select_one(
                    ".c-gameHeader_platform > a, .c-gameHeader_platform > span"
                )
                if single_platform_indicator:
                    platform_name = single_platform_indicator.get_text(strip=True)
                    print(
                        f"Indicador de plataforma única encontrado: '{platform_name}'"
                    )
            except Exception:
                pass

        except Exception as e:
            print(f"Erro ao buscar plataformas: {type(e).__name__} - {e}")

        if not platforms:
            print(f"Nenhuma plataforma específica encontrada para {game_base_url}")
        return platforms

    def _get_reviews_for_specific_url(
        self,
        page_url: str,
        cerimony_date: str,
        critic: bool,
    ) -> List[str]:
        """
        Obtém as avaliações de uma URL específica com carregamento eficiente.
        """
        reviews_list = []
        current_url_to_scrape = page_url
        processed_urls_for_this_call = set()

        while (
            current_url_to_scrape
            and current_url_to_scrape not in processed_urls_for_this_call
        ):
            processed_urls_for_this_call.add(current_url_to_scrape)
            print(f"Raspando reviews de: {current_url_to_scrape}")
            try:
                self.driver.get(current_url_to_scrape)

                # Determinar quantidade total de reviews
                try:
                    reviews_text_element = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located(
                            (By.CLASS_NAME, "c-pageProductReviews_text")
                        )
                    )

                    match = re.search(
                        r"of\s+([\d,]+)|([\d,]+)\s+(User|Critic) Review",
                        reviews_text_element.text,
                    )

                    if match:
                        num_str = match.group(1) or match.group(2)
                        quantidade_de_reviews = int(num_str.replace(",", ""))
                        print(f"Total de reviews esperadas: {quantidade_de_reviews}")
                    else:
                        # Se não encontrar o padrão, considerar que não há paginação
                        print("Padrão de contagem de reviews não encontrado")
                        quantidade_de_reviews = 0
                except Exception:
                    print("Não foi possível determinar número de reviews")
                    quantidade_de_reviews = 0

                # Carregar todas as reviews na página com estratégia adaptativa
                last_review_count = 0
                no_change_count = 0
                max_no_change = 6  # Parar após 6 tentativas sem mudança

                while True:
                    # Rolar a página para carregar mais reviews
                    self.driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight);"
                    )
                    time.sleep(1)  # Aguardar carregamento de conteúdo

                    # Verificar número atual de reviews
                    current_reviews = len(
                        self.driver.find_elements(By.CLASS_NAME, "c-siteReview")
                    )

                    # Verificar se já carregamos todas as reviews esperadas
                    if current_reviews >= quantidade_de_reviews:
                        print(
                            f"Todas as {quantidade_de_reviews} reviews carregadas ({current_reviews})"
                        )
                        break

                    # Verificar se houve progresso no carregamento
                    if current_reviews == last_review_count:
                        no_change_count += 1
                        if no_change_count >= max_no_change:
                            print(
                                f"Carregamento estabilizado em {current_reviews} reviews"
                            )
                            break
                    else:
                        no_change_count = 0

                    last_review_count = current_reviews

                # Extrair reviews do HTML carregado
                soup = BeautifulSoup(self.driver.page_source, "html.parser")
                page_reviews = self._extract_reviews_from_soup(
                    soup, cerimony_date, critic
                )
                reviews_list.extend(page_reviews)
                print(f"Extraídas {len(page_reviews)} reviews desta página")

                # Verificar se há próxima página
                next_page_relative_url = self._get_next_page_url_from_soup(soup)
                if next_page_relative_url:
                    if not next_page_relative_url.startswith("http"):
                        current_url_to_scrape = self.BASE_URL + next_page_relative_url
                    else:
                        current_url_to_scrape = next_page_relative_url
                    print(f"Próxima página encontrada: {current_url_to_scrape}")
                else:
                    current_url_to_scrape = None

            except TimeoutException:
                print(f"Timeout ao carregar {current_url_to_scrape}")
                current_url_to_scrape = None
            except Exception as e:
                print(
                    f"Erro ao raspar {current_url_to_scrape}: {type(e).__name__} - {e}"
                )
                current_url_to_scrape = None

        return reviews_list

    def get_reviews_for_all_platforms(
        self,
        game_slug: str,
        release_year: str,
        cerimony_date: str,
        critic: bool = False,
    ) -> List[str]:
        """
        Obtém as avaliações de um item para todas as plataformas disponíveis.
        """
        base_game_url = f"{self.BASE_URL}/game/{game_slug}"
        details_url = f"{base_game_url}/details"

        review_type = "critic" if critic else "user"
        print(
            f"Processando jogo: {game_slug} (Ano: {release_year}, Tipo: {review_type})"
        )

        try:
            response = self.session.get(details_url, timeout=20)
            # Don't raise for status here immediately to allow graceful fail if 404 to try other slugs
            if response.status_code == 404:
                 print(f"URL não encontrada: {details_url}")
                 return [] # Return empty to trigger next slug attempt
            
            response.raise_for_status()
            details_soup = BeautifulSoup(response.content, "html.parser")
            
            # Data de lançamento
            try:
                release_date_obj = self._extract_release_date(details_soup)
                # if int(release_year) != release_date_obj.year:
                #     print(
                #         f"AVISO: Ano de lançamento diverge. CSV: {release_year}, Metacritic: {release_date_obj.year}"
                #     )
            except Exception as e:
                print(f"Aviso: Não foi possível validar data de lançamento: {e}")

        except Exception as e:
            print(f"Erro ao acessar detalhes ou validar data: {e}")
            if "404" in str(e):
                return []

        all_reviews_collected = []
        platform_slugs = self._get_available_platforms(base_game_url)
        review_type_path = "critic-reviews" if critic else "user-reviews"

        if not platform_slugs:
            print(f"Tentando URL genérica para {game_slug}")
            generic_reviews_url = (
                f"{base_game_url}/{review_type_path}/?sort-by=date&num_items=100"
            )
            all_reviews_collected.extend(
                self._get_reviews_for_specific_url(
                    generic_reviews_url, cerimony_date, critic
                )
            )
        else:
            print(
                f"Coletando reviews para '{game_slug}' nas plataformas: {platform_slugs}"
            )
            for platform_slug_value in platform_slugs:
                platform_specific_url = f"{base_game_url}/{review_type_path}/?platform={platform_slug_value}&sort-by=date&num_items=100"
                print(f"Coletando para plataforma: {platform_slug_value}")
                reviews_for_platform = self._get_reviews_for_specific_url(
                    platform_specific_url, cerimony_date, critic
                )
                all_reviews_collected.extend(reviews_for_platform)
                print(
                    f"Coletadas {len(reviews_for_platform)} reviews para {platform_slug_value}"
                )

        print(f"Total de reviews para '{game_slug}': {len(all_reviews_collected)}")
        return all_reviews_collected

    def _extract_release_date(self, soup: BeautifulSoup) -> date:
        """Extrai a data de lançamento do conteúdo da página de detalhes."""
        try:
            release_date_span = soup.find(
                "span", string=re.compile(r"Release Date:", re.IGNORECASE)
            )
            if release_date_span and release_date_span.find_next_sibling("span"):
                date_str = release_date_span.find_next_sibling("span").get_text(
                    strip=True
                )
            elif soup.find("div", class_="c-gameDetails_ReleaseDate"):
                date_str_element = soup.find(
                    "div", class_="c-gameDetails_ReleaseDate"
                ).find_all("span")
                if len(date_str_element) > 1:
                    date_str = date_str_element[1].get_text(strip=True)
                else:
                    raise AttributeError("Estrutura de data de lançamento incompleta")
            elif soup.find("li", class_=re.compile(r"release.?date", re.I)):
                date_li = soup.find("li", class_=re.compile(r"release.?date", re.I))
                if date_li:
                    date_span_in_li = date_li.find(
                        "span", class_="g-text-bold"
                    ) or date_li.find(
                        "span", class_=lambda x: x and "u-text-uppercase" not in x
                    )
                    if date_span_in_li:
                        date_str = date_span_in_li.get_text(strip=True)
                    else:
                        date_str_raw = date_li.get_text(strip=True)
                        date_match = re.search(
                            r"(\w{3}\s\d{1,2},\s\d{4})", date_str_raw
                        )
                        if date_match:
                            date_str = date_match.group(1)
                        else:
                            raise AttributeError("Formato de data não reconhecido")
                else:
                    raise AttributeError("Elemento de data não encontrado")
            else:
                raise AttributeError("Elemento de data não encontrado")

            return datetime.strptime(date_str, "%b %d, %Y").date()
        except Exception as e:
            raise RuntimeError(f"Não foi possível extrair a data de lançamento: {e}")

    def _extract_reviews_from_soup(
        self, soup: BeautifulSoup, cerimony_date: str, critic: bool
    ) -> List[str]:
        """Extrai as avaliações do HTML (soup) de uma página."""
        reviews = []
        cerimony_date_obj = datetime.strptime(cerimony_date, "%d/%m/%Y").date()

        review_elements = soup.find_all("div", class_=re.compile(r"\bc-siteReview\b"))

        for review_div in review_elements:
            if "ad_unit" in review_div.get("class", []):
                continue

            score_element = review_div.find(
                "div", class_=re.compile(r"\bc-siteReviewScore\b")
            )
            score = None
            if score_element:
                score_span = score_element.find("span")
                if score_span:
                    score = score_span.text.strip()

            if not score:
                continue

            if not critic:
                date_str_element = review_div.find(
                    "div", class_=re.compile(r"c-siteReviewHeader_reviewDate|c-siteReview_reviewDate")
                )
                if not date_str_element:
                    continue

                date_str = date_str_element.text.strip()
                try:
                    review_date = datetime.strptime(date_str, "%b %d, %Y").date()
                    if review_date > cerimony_date_obj:
                        continue
                except ValueError:
                    continue

            try:
                int(score)
                reviews.append(score)
            except ValueError:
                pass

        return reviews

    def _get_next_page_url_from_soup(self, soup: BeautifulSoup) -> Optional[str]:
        """Verifica se existe uma próxima página de avaliações no HTML (soup)."""
        next_page_link = soup.select_one(
            "a.c-pager_next[href], a.c-content_pager_anchor_next[href]"
        )
        if next_page_link and next_page_link.get("href"):
            href = next_page_link["href"]
            if "disabled" not in next_page_link.get("class", []):
                return href

        legacy_next_page = soup.find("div", class_="page_flipper")
        if legacy_next_page:
            link_span = legacy_next_page.find("span", class_="flipper next")
            if link_span and link_span.find("a"):
                href = link_span.find("a")["href"]
                return href

        return None

    def close(self):
        """Fecha o driver do Selenium."""
        if self.driver:
            self.driver.quit()
            print("Driver do Selenium fechado.")


def slugify_title(title: str) -> str:
    """Converte o título em um slug apropriado para URLs."""
    slug = title.lower()
    slug = slug.replace("'", "")
    slug = slug.replace(":", "")
    slug = slug.replace("&", "and")
    slug = slug.replace("ē", "e")
    slug = slug.replace("é", "e")
    slug = slug.replace("ö", "o")
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug


def process_game_reviews(
    scraper: MetacriticScraper,
    item_name: str,
    year: str,
    cerimony_date: str,
    critic: bool,
) -> List[str]:
    """
    Processa as avaliações de um jogo, tentando diferentes variações de slug.
    """
    review_type = "críticos" if critic else "usuários"
    slug_base = slugify_title(item_name)
    potential_slugs = [slug_base, f"{slug_base}-{year}"]

    # Specific fix for Hades 2 -> Hades II
    if "hades-2" in slug_base:
        potential_slugs.append(slug_base.replace("hades-2", "hades-ii"))

    if year in item_name:
        item_name_no_year = item_name.replace(f"({year})", "").replace(year, "").strip()
        if item_name_no_year and slugify_title(item_name_no_year) != slug_base:
            potential_slugs.append(slugify_title(item_name_no_year))

    potential_slugs = list(dict.fromkeys(s for s in potential_slugs if s))

    for title_slug_variant in potential_slugs:
        print(
            f"Tentando slug: {title_slug_variant} para '{item_name}' (reviews de {review_type})"
        )
        try:
            reviews = scraper.get_reviews_for_all_platforms(
                title_slug_variant, year, cerimony_date, critic
            )
            if reviews: # Only return if we found reviews
                return reviews
            
            # If reviews list is empty, it might be that the page exists but no reviews, 
            # OR the page doesn't exist (handled by get_reviews... returning []).
            # We continue trying other slugs if result is empty, unless we are sure the page was found but empty.
            # For simplicity, let's assume if we find reviews we stop. If not, we keep trying slugs.

        except Exception as e:
            print(
                f"Erro ao processar '{title_slug_variant}' para {review_type}: {e}. Tentando próximo slug."
            )

    print(
        f"Nenhuma review de {review_type} encontrada para '{item_name}' após tentar: {potential_slugs}"
    )
    return []


def main():
    scraper = MetacriticScraper()
    all_games_data: Dict[str, Dict] = {}

    output_json_path = "./data/the_game_awards_2025_data.json"
    
    games_list = [
        "Clair Obscur: Expedition 33",
        "Death Stranding 2: On the beach",
        "Donkey Kong Bananza",
        "Hades 2",
        "Hollow Knight: Silksong",
        "Kingdom Come: Deliverance 2"
    ]

    # Load existing data if available to append/merge (optional, but safer to write new file as requested)
    # The user said "Não sobrescreva nenhum dado que já está salvo".
    # Writing to a new file ensures this.
    
    try:
        ceremony_d = "31/12/2025" # Future date to capture all reviews
        game_year = "2025"

        for i, game_name in enumerate(games_list):
            print(f"\n[{i + 1}/{len(games_list)}] Processando: '{game_name}' (Ano: {game_year})")

            user_reviews = process_game_reviews(
                scraper, game_name, game_year, ceremony_d, critic=False
            )

            critic_reviews = process_game_reviews(
                scraper, game_name, game_year, ceremony_d, critic=True
            )

            game_key = f"{game_name}"
            all_games_data[game_key] = {
                "name": game_name,
                "user-reviews": user_reviews,
                "critic-reviews": critic_reviews,
                "year": game_year,
                "cerimony-date": ceremony_d,
                "winner": "Unknown", # Future prediction
            }
            time.sleep(1)

        import os

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

        with open(output_json_path, "w", encoding="utf-8") as games_file:
            json.dump(all_games_data, games_file, indent=4, ensure_ascii=False)

        print(f"\nProcessamento concluído. Dados salvos em {output_json_path}")

    except Exception as e:
        print(f"Erro fatal: {type(e).__name__} - {e}")
    finally:
        scraper.close()


if __name__ == "__main__":
    main()
