"""Command-line entry point for movie rating training and prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.movie_rating import load_dataset, predict_rating, train_evaluate

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "Section14_Q4_Movie_Rating_CICD.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and predict movie ratings (0–10).")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to movie-rating CSV")
    parser.add_argument("--genre", default="Drama")
    parser.add_argument("--duration", type=float, default=145, help="Duration in minutes")
    parser.add_argument("--budget", type=float, default=50, help="Budget in crore INR")
    parser.add_argument("--popularity", type=float, default=70, help="Popularity index")
    parser.add_argument("--year", type=int, default=2025, help="Release year")
    parser.add_argument("--platform", default="Streaming", choices=["Streaming", "Theatre", "Both"])
    parser.add_argument("--reviews", type=int, default=10000, help="Review count")
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    args = parser.parse_args()

    data = load_dataset(args.data)
    model, metrics = train_evaluate(data)
    movie = {
        "Genre": args.genre,
        "Duration_Min": args.duration,
        "Budget_Crore_INR": args.budget,
        "Popularity_Index": args.popularity,
        "Release_Year": args.year,
        "Release_Platform": args.platform,
        "Review_Count": args.reviews,
    }
    result = {
        "model": "random_forest_regressor",
        "prediction_rating_10": predict_rating(model, movie),
        "input": movie,
        "holdout_metrics": metrics,
        "dataset_rows": int(len(data)),
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
