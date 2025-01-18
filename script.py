import json
from bs4 import BeautifulSoup
import re

# Carregar o HTML do arquivo
with open("soup.html", "r", encoding="utf-8") as file:
    html_content = file.read()

# Usar BeautifulSoup para processar o HTML
soup = BeautifulSoup(html_content, "html.parser")

# Encontrar o script que contém os dados do `window.__NUXT__`
script_tags = soup.find_all("script")
nuxt_data = None

for script in script_tags:
    if script.string and "window.__NUXT__" in script.string:
        with open("nuxt_data.json", "w") as nuxt_file:
            nuxt_file.write(script.string)
        # Extrair o conteúdo bruto de `window.__NUXT__`
        nuxt_raw = script.string
        nuxt_json_start = nuxt_raw.find("{")
        raw_json = nuxt_raw[nuxt_json_start:]

        # Corrigir propriedades não entre aspas duplas (se necessário)
        corrected_json = re.sub(r"(?<!\\)'", '"', raw_json)  # Substituir aspas simples por duplas
        corrected_json = re.sub(r"(\w+):", r'"\1":', corrected_json)  # Garantir chaves com aspas duplas
        
        # Carregar JSON corrigido
        nuxt_data = json.loads(corrected_json)
        break

if nuxt_data is None:
    print("Dados do window.__NUXT__ não encontrados.")
    exit()

# Acessar as notas dos filmes na estrutura do `window.__NUXT__`
reviews = nuxt_data.get("data", [{}])[0].get("k", {}).get("components", [])

# Extrair as informações desejadas
movie_scores = []
for review in reviews:
    try:
        score = review.get("score")  # Ajustar o caminho conforme a estrutura real
        user = review.get("user")  # Ajustar o caminho conforme necessário
        comment = review.get("comment")  # Ajustar o caminho conforme necessário
        if score:
            movie_scores.append({
                "user": user,
                "score": score,
                "comment": comment
            })
    except AttributeError:
        continue

# Exibir os resultados
for movie in movie_scores:
    print(f"Usuário: {movie['user']}, Nota: {movie['score']}, Comentário: {movie['comment']}")
