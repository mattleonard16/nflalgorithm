from pathlib import Path

from utils.model_provenance import build_provenance, describe_bundle_provenance


def test_build_provenance_records_training_window() -> None:
    provenance = build_provenance(
        season_week_tuples=[(2024, 1), (2024, 2)],
        feature_cols=["age"],
        training_rows=10,
        private_model_path=Path("models/weekly/receiving_yards_model.joblib"),
    )

    assert provenance["training_rows"] == 10
    assert provenance["season_week_tuples"] == [[2024, 1], [2024, 2]]
    assert describe_bundle_provenance({"provenance": provenance}).startswith("trained_at=")
    assert describe_bundle_provenance({}) == "no provenance"
