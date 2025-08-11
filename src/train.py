import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_experiment("genetic_rf_tuning")

def run_experiment(n_estimators, max_depth):
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        # Predict and log accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)

        # Log the trained model
        mlflow.sklearn.log_model(model, "model")

        # Feature importance plot
        feature_importance = pd.Series(
            model.feature_importances_, index=data.feature_names
        )
        feature_importance.sort_values().plot(kind="barh", figsize=(8, 10))
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")

        print(f"Run: n_estimators={n_estimators}, max_depth={max_depth} → Accuracy={accuracy:.4f}")

# Try multiple combinations
for n in [10, 50, 100]:
    for depth in [3, 5, None]:
        run_experiment(n, depth)
