"""Train and evaluate the synthetic forward cash-flow model."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from synthetic_data import FEATURE_COLUMNS, create_training_data


def train_model(sample_count: int = 2400, seed: int = 42):
    """Fit on one synthetic split and report performance on held-out profiles."""
    data = create_training_data(sample_count, seed)
    x_train, x_test, y_train, y_test = train_test_split(
        data[FEATURE_COLUMNS],
        data["future_surplus_persisted"],
        test_size=0.25,
        random_state=seed,
        stratify=data["future_surplus_persisted"],
    )

    model = RandomForestClassifier(
        n_estimators=240,
        min_samples_leaf=5,
        max_features=0.8,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    report = {
        "held_out_profiles": int(len(y_test)),
        "positive_rate": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, probabilities >= 0.5)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "feature_importance": dict(
            sorted(
                zip(FEATURE_COLUMNS, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
    }
    return model, report


if __name__ == "__main__":
    _, evaluation = train_model()
    print("CoverIQ synthetic forward cash-flow model")
    print("Held-out synthetic profiles:", evaluation["held_out_profiles"])
    print("ROC AUC:", round(evaluation["roc_auc"], 3))
    print("Accuracy:", round(evaluation["accuracy"], 3))
    print("Brier score:", round(evaluation["brier_score"], 3))
    print("This is a synthetic benchmark, not real-customer validation.")
