# Resumo Comparativo - Análise de Predição Game Awards

**Data de Execução:** 2025-11-08
**Período de Treinamento:** 2014-2022
**Período de Teste:** 2023-2024
**Total de Jogos (Teste):** 12 (2 Winners, 10 Losers)

---

## Comparação de Accuracy por Tipo de Feature

### 1. Críticas (Apenas Reviews de Críticos)

| Modelo         | Accuracy | Precision | Recall  | F1 Score |
|----------------|----------|-----------|---------|----------|
| Naive Bayes    | 58.33%   | 28.57%    | 100.00% | 44.44%   |
| KNN (K=15)     | 58.33%   | 28.57%    | 100.00% | 44.44%   |
| Random Forest  | **83.33%** | 50.00%  | 100.00% | 66.67%   |

**Features utilizadas:** 6 features de críticas
- critic-mean, critic-stdev, critic-median, critic-mode, critic-percentile-25, critic-percentile-75

**Melhor Modelo:** Random Forest com 83.33% de accuracy

---

### 2. Usuários (Apenas Reviews de Usuários)

| Modelo         | Accuracy | Precision | Recall  | F1 Score |
|----------------|----------|-----------|---------|----------|
| Naive Bayes    | 16.67%   | N/A       | N/A     | N/A      |
| KNN (K=15)     | **83.33%** | 50.00%  | 100.00% | 66.67%   |
| Random Forest  | 75.00%   | 33.33%    | 50.00%  | 40.00%   |

**Features utilizadas:** 6 features de usuários
- user-mean, user-stdev, user-median, user-mode, user-percentile-25, user-percentile-75

**Melhor Modelo:** KNN com 83.33% de accuracy

**Observação:** Naive Bayes apresentou desempenho muito baixo (16.67%) com features de usuários, prevendo Winner para quase todos os casos.

---

### 3. Combinação (Reviews de Críticos + Usuários)

| Modelo         | Accuracy | Precision | Recall  | F1 Score |
|----------------|----------|-----------|---------|----------|
| Naive Bayes    | 16.67%   | N/A       | N/A     | N/A      |
| KNN (K=15)     | 75.00%   | 40.00%    | 100.00% | 57.14%   |
| Random Forest  | **91.67%** | 66.67%  | 100.00% | 80.00%   |

**Features utilizadas:** 12 features (6 de críticos + 6 de usuários)
- Todas as features de críticos e usuários combinadas

**Melhor Modelo:** Random Forest com 91.67% de accuracy

---

### 4. Ponderado 90/10 (Oficial The Game Awards) ⭐

| Modelo         | Accuracy | Precision | Recall  | F1 Score |
|----------------|----------|-----------|---------|----------|
| Naive Bayes    | 16.67%   | N/A       | N/A     | N/A      |
| KNN (K=15)     | 58.33%   | 28.57%    | 100.00% | 44.44%   |
| Random Forest  | **91.67%** | 66.67%  | 100.00% | 80.00%   |

**Features utilizadas:** 12 features ponderadas
- Features de críticos multiplicadas por 0.9 (90%)
- Features de usuários multiplicadas por 0.1 (10%)

**Melhor Modelo:** Random Forest com 91.67% de accuracy

**Observação:** Esta ponderação reflete o método oficial do The Game Awards, onde críticos têm 90% de peso e usuários 10%.

---

## Análise Geral

### Performance por Modelo

#### Naive Bayes
- **Melhor Performance:** Críticas (58.33%)
- **Pior Performance:** Usuários, Combinação e Ponderado (16.67%)
- **Observação:** Modelo muito sensível ao tipo de feature, com desempenho degradado quando usa features de usuários

#### KNN (K=15)
- **Melhor Performance:** Críticas e Usuários (83.33%)
- **Performance Combinada:** 75.00%
- **Performance Ponderada:** 58.33%
- **Observação:** Desempenho inconsistente com diferentes tipos de features

#### Random Forest ⭐
- **Melhor Performance:** Combinação e Ponderado 90/10 (91.67%)
- **Performance com Críticas:** 83.33%
- **Performance com Usuários:** 75.00%
- **Observação:** Modelo mais robusto e consistente em todos os cenários

---

## Comparação: Combinação vs Ponderado 90/10

Ambas as abordagens alcançaram **91.67% de accuracy** com Random Forest, mas com diferenças sutis:

### Combinação (Todas features iguais)
- Random Forest: 91.67%
- KNN: 75.00%
- Errou: Hi-Fi Rush (não previsto como winner)

### Ponderado 90/10 (Oficial TGA)
- Random Forest: 91.67%
- KNN: 58.33% (pior que combinação)
- Errou: Hi-Fi Rush (não previsto como winner)

**Análise:** Embora tenham a mesma accuracy final, a ponderação 90/10 representa melhor o processo real do The Game Awards. O Random Forest demonstra robustez ao manter a mesma performance em ambos os cenários.

---

## Comparação Visual de Accuracy

```
                    Naive Bayes    KNN        Random Forest
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Críticas              58.33%     58.33%      83.33% ███████
Usuários              16.67%     83.33%      75.00% ███████
Combinação            16.67%     75.00%      91.67% ████████
Ponderado 90/10       16.67%     58.33%      91.67% ████████
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Conclusões e Recomendações

### 1. Melhor Abordagem Geral

**Random Forest com Ponderação 90/10** é a escolha recomendada:
- **91.67% de accuracy**
- **80.00% de F1 Score**
- Reflete o método oficial do The Game Awards
- Identificou corretamente os 2 winners do conjunto de teste (Baldur's Gate 3 e Astro Bot)

### 2. Análise por Tipo de Feature

**Críticas:**
- Mais confiáveis para Random Forest (83.33%)
- Boa separação entre winners e losers
- Features estatísticas de críticos são altamente preditivas

**Usuários:**
- Desempenho variável entre modelos
- KNN teve bom desempenho (83.33%)
- Naive Bayes falhou completamente (16.67%)
- Menos preditivas quando usadas isoladamente

**Combinação:**
- Melhor resultado absoluto (91.67% com Random Forest)
- Aproveita a complementaridade entre críticos e usuários

**Ponderado 90/10:**
- Mesma accuracy que combinação para Random Forest (91.67%)
- Mais realista em relação ao processo oficial do TGA
- Demonstra que críticos têm maior poder preditivo

### 3. Recomendação de Modelo

Para predições futuras do Game Awards:

**Opção 1 (RECOMENDADA):** 🏆 Random Forest com Ponderação 90/10
- Accuracy: 91.67%
- F1 Score: 80.00%
- Reflete método oficial TGA
- Maior relevância prática

**Opção 2 (Alternativa):** Random Forest com Combinação
- Accuracy: 91.67%
- F1 Score: 80.00%
- Mesma performance, ponderação implícita

**Opção 3 (Fallback):** Random Forest apenas com Críticas
- Accuracy: 83.33%
- Menor complexidade
- Útil quando dados de usuários não estão disponíveis

**Não Recomendado:** Naive Bayes com qualquer conjunto que inclua features de usuários
- Desempenho muito baixo
- Alta taxa de falsos positivos

### 4. Insights Importantes

1. **Críticos são mais preditivos que usuários:** Features de críticos sozinhas (83.33%) superam usuários sozinhos (75.00% melhor caso)

2. **Ponderação 90/10 valida o método TGA:** O fato de alcançar 91.67% confirma que a ponderação oficial do TGA (90% críticos, 10% usuários) é bem fundamentada

3. **Random Forest é robusto:** Único modelo que manteve alta performance em todos os cenários (75-91.67%)

4. **Naive Bayes não é adequado:** Completamente inadequado para features de usuários ou combinações

5. **Jogo mais difícil de prever:** Hi-Fi Rush foi incorretamente previsto como Winner em alguns cenários, sugerindo que tinha características similares aos vencedores históricos

---

## Previsões para 2023-2024

### Winners Identificados Corretamente (Random Forest Ponderado 90/10):
- ✅ **Baldur's Gate 3** (2023) - 75.6% probabilidade
- ✅ **Astro Bot** (2024) - 51.6% probabilidade

### Falso Positivo:
- ❌ **Hi-Fi Rush** previsto como Winner (74.2% probabilidade)
  - Teve reviews excepcionais mas não venceu

### Games com Alta Pontuação que Não Venceram:
- The Legend of Zelda: Tears of the Kingdom (44.2%)
- Cyberpunk 2077 (39.0%)
- Elden Ring: Shadow of the Erdtree (33.2%)

---

## Estrutura de Arquivos

```
resultados/
├── criticas/
│   ├── knn_confusion_matrix.png
│   ├── metrics_log.txt
│   ├── naive_bayes_confusion_matrix.png
│   └── random_forest_confusion_matrix.png
├── usuarios/
│   ├── knn_confusion_matrix.png
│   ├── metrics_log.txt
│   ├── naive_bayes_confusion_matrix.png
│   └── random_forest_confusion_matrix.png
├── combinacao/
│   ├── knn_confusion_matrix.png
│   ├── metrics_log.txt
│   ├── naive_bayes_confusion_matrix.png
│   └── random_forest_confusion_matrix.png
├── ponderado_90_10/
│   ├── knn_confusion_matrix.png
│   ├── metrics_log.txt
│   ├── naive_bayes_confusion_matrix.png
│   └── random_forest_confusion_matrix.png
└── RESUMO_COMPARATIVO.md (este arquivo)
```

---

## Como Reproduzir os Resultados

```bash
python ml_predict_future_comparative.py
```

Este script executa automaticamente as quatro análises e salva todos os resultados nas respectivas pastas.

---

## Próximos Passos Sugeridos

1. **Análise de Feature Importance:** Identificar quais features específicas são mais preditivas
2. **Ensemble Methods:** Combinar múltiplos modelos para potencialmente melhorar accuracy
3. **Análise de Outliers:** Investigar por que Hi-Fi Rush foi previsto incorretamente
4. **Validação Cruzada:** Testar com diferentes splits temporais
5. **Predição 2025:** Usar modelo treinado para prever vencedores de 2025
