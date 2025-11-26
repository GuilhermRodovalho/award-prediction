#!/usr/bin/env python3
"""
Script to predict Game Awards winners using historical data.

This script trains models on historical data (2014-2022) and tests
on recent years (2023, 2024) to evaluate predictive performance.
"""

from ml_final import predictFutureYears

if __name__ == "__main__":
    # Define years for training and testing
    train_years = [str(year) for year in range(2014, 2023)]  # 2014-2022
    test_years = ['2023', '2024']

    print("\n" + "=" * 80)
    print("🎮 THE GAME AWARDS - PREDIÇÃO TEMPORAL")
    print("=" * 80)
    print(f"\n📚 Treinando com dados históricos: {train_years[0]}-{train_years[-1]}")
    print(f"🔮 Testando/prevendo anos: {', '.join(test_years)}")
    print("\n" + "=" * 80)

    # Run predictions
    predictFutureYears("the_game_awards", train_years, test_years)

    print("\n" + "=" * 80)
    print("✅ ANÁLISE COMPLETA!")
    print("=" * 80)
