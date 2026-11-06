from pathlib import Path

from app.movie_rating import load_dataset, predict_rating, train_evaluate

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "Section14_Q4_Movie_Rating_CICD.csv"


def test_dataset_is_loaded_and_missing_features_are_supported():
    data = load_dataset(DATASET)
    assert len(data) == 480
    assert data["Rating_10"].between(0, 10).all()
    assert data["Genre"].isna().any()  # Pipeline imputes source-data gaps.


def test_training_and_prediction_are_valid():
    model, metrics = train_evaluate(load_dataset(DATASET))
    rating = predict_rating(
        model,
        {
            "Genre": "Drama",
            "Duration_Min": 145,
            "Budget_Crore_INR": 50,
            "Popularity_Index": 70,
            "Release_Year": 2025,
            "Release_Platform": "Streaming",
            "Review_Count": 10000,
        },
    )
    assert 0 <= rating <= 10
    assert metrics["test_rows"] == 96
    assert metrics["train_rows"] == 384
    assert metrics["mae"] >= 0
