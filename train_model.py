"""Train and evaluate the simulated premium-affordability model."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from synthetic_data import FEATURE_COLUMNS, LABEL_COLUMN, create_training_data


def train_model(sample_count: int = 2400, seed: int = 42):
    """Fit on simulated histories and evaluate on held-out fictional customers."""
    data = create_training_data(sample_count, seed)
    train, test = train_test_split(
        data,
        test_size=0.25,
        random_state=seed,
        stratify=data[LABEL_COLUMN],
    )

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1500, class_weight="balanced", random_state=seed),
            ),
        ]
    )
    model.fit(train[FEATURE_COLUMNS], train[LABEL_COLUMN])

    y_true = test[LABEL_COLUMN]
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    predictions = probabilities >= 0.5
    majority_class = int(train[LABEL_COLUMN].mode().iloc[0])
    baseline = [majority_class] * len(test)

    coefficients = model.named_steps["classifier"].coef_[0]
    feature_weights = dict(
        sorted(
            zip(FEATURE_COLUMNS, (float(value) for value in coefficients)),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )
    )
    report = {
        "training_profiles": int(len(train)),
        "held_out_profiles": int(len(test)),
        "positive_rate": float(y_true.mean()),
        "baseline_accuracy": float(accuracy_score(y_true, baseline)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=[0, 1],
            target_names=["reserve_not_maintained", "reserve_maintained"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
        "feature_weights": feature_weights,
        "training_data_note": (
            "Training and evaluation use simulated transaction histories. "
            "Metrics describe performance on this simulator, not real customers."
        ),
    }
    return model, report


if __name__ == "__main__":
    _, evaluation = train_model()
    print("CoverIQ premium affordability model · simulated data")
    print(f"Training profiles: {evaluation['training_profiles']:,}")
    print(f"Held-out profiles: {evaluation['held_out_profiles']:,}")
    print(f"Baseline accuracy: {evaluation['baseline_accuracy']:.3f}")
    print(f"Model accuracy:    {evaluation['accuracy']:.3f}")
    print(f"Balanced accuracy: {evaluation['balanced_accuracy']:.3f}")
    print(f"ROC-AUC:           {evaluation['roc_auc']:.3f}")
    print(f"Brier score:       {evaluation['brier_score']:.3f}")
    print("Top feature weights:")
    for name, weight in list(evaluation["feature_weights"].items())[:6]:
        print(f"  {name:<34} {weight:+.3f}")
    print(evaluation["training_data_note"])
