import pandas as pd
import pickle
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DATA_PATH = "data/processed/activity_features.csv"
MODEL_PATH = "model/productivity_model.pkl"


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)
    X = df.drop("label", axis=1)
    y = df["label"]

    # Load model
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    label_encoder = saved["label_encoder"]

    y_encoded = label_encoder.transform(y)

    # Predictions
    y_pred = model.predict(X)

    # Confusion Matrix
    cm = confusion_matrix(y_encoded, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix – NoPretend AI")
    plt.show()

    # Feature Importance
    importances = model.feature_importances_
    feature_names = X.columns

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df)


if __name__ == "__main__":
    main()
