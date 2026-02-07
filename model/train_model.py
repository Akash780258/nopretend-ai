import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "data/processed/activity_features.csv"
MODEL_PATH = "model/productivity_model.pkl"


def main():
    print("Loading processed dataset...")

    df = pd.read_csv(DATA_PATH)

    # Separate features and target
    X = df.drop("label", axis=1)
    y = df["label"]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)

    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    ))

    # Save model + encoder
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "label_encoder": label_encoder
            },
            f
        )

    print("\nModel saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
