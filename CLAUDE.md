# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
- `poetry install` - Install all project dependencies
- `poetry shell` - Activate the virtual environment

### Code Quality
- `ruff check` - Lint code with Ruff
- `ruff format` - Format code with Ruff

### Running the Prediction System
- `python ml_final.py` - Run ML predictions for award winners using historical data
- `python main_oscars.py` - Run the Metacritic scraper to collect movie/game data
- `python ai.py` - Run comprehensive ML analysis with visualizations
- `python plots.py` - Generate various data visualization plots

## Project Architecture

### Data Collection Pipeline
The project uses a sophisticated data collection system centered around the `MetacriticScraper` class in `main_oscars.py`:

- **Web Scraping**: Selenium-based scraper that extracts user and critic reviews from Metacritic
- **Multi-platform Support**: Automatically detects and scrapes reviews from all available game platforms
- **Date Filtering**: Reviews are filtered based on ceremony dates to avoid data leakage
- **Robust URL Generation**: Uses intelligent slug generation and fallback mechanisms for finding game pages

### Data Storage Structure
- **CSV Files** (`csv/`): Raw award nomination data organized by ceremony type
- **JSON Files** (`data/`): Processed review data with extracted features
- **Data Flow**: CSV → Metacritic scraping → JSON with reviews → Feature extraction → ML models

### Machine Learning Pipeline

#### Feature Engineering (`ml_final.py` and `ai.py`)
The system extracts statistical features from review data:
- **User Reviews**: Mean, standard deviation, median, mode, 25th/75th percentiles
- **Critic Reviews**: Same statistical measures, normalized to 0-10 scale
- **Derived Features**: Rating differences, review counts, availability flags

#### Model Implementation
Three ML models are implemented with hyperparameter optimization:
- **Naive Bayes**: `GaussianNB` with probability outputs
- **K-Nearest Neighbors**: Optimized K values per dataset (K=5-15)
- **Random Forest**: Entropy criterion with tuned min_samples_leaf parameters

#### Data Handling
- **Class Imbalance**: Uses `RandomOverSampler` to handle winner/loser imbalance
- **Normalization**: `StandardScaler` for KNN model
- **Cross-validation**: 5-fold CV for model evaluation

### Award Categories Supported
- **Oscars**: Historical and 2023 predictions
- **Golden Globe**: Regular and comedy categories
- **The Game Awards**: Gaming industry predictions

### Visualization System (`ai.py`)
Comprehensive plotting functionality:
- Feature importance analysis for Random Forest
- Rating distribution comparisons
- Model performance metrics
- Confusion matrices for all models
- Prediction probability visualizations

## Data Files

### Input Data
- Award nomination CSV files with columns: `year_game`, `year_ceremony`, `game_name`, `winner`
- Ceremony date format: `DD/MM/YYYY`

### Output Data
- JSON files containing processed review data with statistical features
- PNG visualization files for analysis results

## Configuration Notes

- **Selenium Setup**: Headless Chrome driver with specific options for web scraping
- **API Rate Limiting**: Built-in delays between requests to respect Metacritic's servers
- **Error Handling**: Comprehensive exception handling for network issues and data parsing

## Performance Considerations

- The scraping process is time-intensive due to rate limiting and page load times
- Models are optimized for small datasets typical in award prediction scenarios
- Feature extraction includes extensive statistical analysis for robust prediction features