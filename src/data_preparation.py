
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_and_split_data():

    # Find the main project folder
    project_root = Path(__file__).resolve().parent.parent

    # Load the dataset
    data_path = project_root / "dataset" / "data.csv"
    df = pd.read_csv(data_path, sep=";")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert the target into a binary classification problem
    df["Target_Binary"] = df["Target"].apply(
        lambda x: "Dropout" if x == "Dropout" else "Non-Dropout"
    )

    # Remove second semester features
    second_sem_columns = [
        "Curricular units 2nd sem (credited)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (evaluations)",
        "Curricular units 2nd sem (approved)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 2nd sem (without evaluations)"
    ]

    df = df.drop(columns=second_sem_columns)

    # Separate features and target
    X = df.drop(columns=["Target", "Target_Binary"])
    y = df["Target_Binary"]

    # Split the data into training and testing sets
    return train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
