
import streamlit as st
import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="mRNA Safety Intelligence",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Moderna mRNA Vaccine Safety Intelligence")
st.caption("Educational AI prototype | Demonstration only")

st.warning(
    "This project is for academic demonstration only. "
    "It is not a clinically validated system and must not "
    "be used to make medical or vaccine dosage decisions."
)

# --------------------------------------------------
# 2. FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).parent

data_path = BASE_DIR / "data" / "patient_records.csv"
output_dir = BASE_DIR / "outputs"

# --------------------------------------------------
# 3. DATASET DASHBOARD
# --------------------------------------------------

st.header("📊 Dataset Overview")

try:
    df = pd.read_csv(data_path)

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", len(df))
    col2.metric("Dataset Features", len(df.columns))
    col3.metric(
        "Missing Values",
        int(df.isna().sum().sum())
    )

    # Dataset explorer
    st.subheader("🔎 Interactive Dataset Explorer")

    selected_column = st.selectbox(
        "Choose a column to explore",
        df.columns.tolist()
    )

    search_text = st.text_input(
        "Search for a value in the selected column"
    )

    filtered_df = df.copy()

    if search_text:
        filtered_df = df[
            df[selected_column]
            .astype(str)
            .str.contains(
                search_text,
                case=False,
                na=False,
                regex=False
            )
        ]

    st.write(f"Matching records: {len(filtered_df)}")

    st.dataframe(
        filtered_df,
        width="stretch"
    )

    # Numerical feature charts
    st.subheader("📈 Explore Numerical Features")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:
        selected_numeric = st.selectbox(
            "Choose a numerical feature",
            numeric_columns
        )

        st.bar_chart(df[selected_numeric].dropna())

    # Dataset information
    st.subheader("📋 Dataset Information")

    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(dtype) for dtype in df.dtypes
        ],
        "Missing Values": df.isna().sum().values
    })

    st.dataframe(
        info,
        width="stretch"
    )

    # Target distribution
    st.subheader("📊 Reaction Score Distribution")

    if "Reaction_Score" in df.columns:
        counts = (
            df["Reaction_Score"]
            .value_counts(dropna=False)
            .sort_index()
        )

        st.bar_chart(counts)

    # Project modules
    st.subheader("📚 Project Modules")

    st.write("1. Dataset inspection and exploration")
    st.write("2. Logistic Regression model")
    st.write("3. Random Forest model")
    st.write("4. Self-attention demonstration")
    st.write("5. Model evaluation and saving")

except FileNotFoundError:
    st.error(
        "Dataset not found. Please check that "
        "data/patient_records.csv exists inside project1."
    )

except Exception as error:
    st.error(f"Dataset dashboard error: {error}")


# --------------------------------------------------
# 4. MACHINE LEARNING MODEL EVALUATION
# --------------------------------------------------

st.divider()

st.header("🤖 Machine Learning Model Evaluation")

st.write(
    "Train two baseline models and compare their "
    "performance on a held-out test dataset."
)

st.caption(
    "The results are experimental and do not establish "
    "clinical validity or individual medical risk."
)

if st.button("Train and Compare Models"):

    try:
        df_eval = pd.read_csv(data_path)

        # Remove rows without target labels
        df_eval = df_eval.dropna(
            subset=["Reaction_Score"]
        )

        # Exclude target and patient identifier
        X = df_eval.drop(
            columns=["Reaction_Score", "Patient_ID"],
            errors="ignore"
        )

        y = df_eval["Reaction_Score"]

        if y.nunique() < 2:
            st.error(
                "At least two target classes are required."
            )

        elif y.value_counts().min() < 2:
            st.error(
                "Each target class needs at least two records."
            )

        else:
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=42,
                stratify=y
            )

            # Identify numerical and categorical features
            numeric_cols = X.select_dtypes(
                include="number"
            ).columns.tolist()

            categorical_cols = X.select_dtypes(
                exclude="number"
            ).columns.tolist()

            transformers = []

            if numeric_cols:
                numeric_pipeline = Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ])

                transformers.append(
                    (
                        "numeric",
                        numeric_pipeline,
                        numeric_cols
                    )
                )

            if categorical_cols:
                categorical_pipeline = Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ])

                transformers.append(
                    (
                        "categorical",
                        categorical_pipeline,
                        categorical_cols
                    )
                )

            preprocessing = ColumnTransformer(
                transformers=transformers
            )

            # Define the models
            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000,
                    random_state=42
                ),

                "Random Forest": RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
            }

            results = []

            # Create output folder if it does not exist
            output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # Train, evaluate and save each model
            for name, classifier in models.items():

                model = Pipeline([
                    (
                        "preprocessing",
                        preprocessing
                    ),
                    (
                        "classifier",
                        classifier
                    )
                ])

                # Train model
                model.fit(X_train, y_train)

                # STEP 10: SAVE TRAINED MODEL
                filename = (
                    name.lower().replace(" ", "_")
                    + ".joblib"
                )

                model_path = output_dir / filename

                joblib.dump(
                    model,
                    model_path
                )

                # Generate predictions for evaluation
                predictions = model.predict(X_test)

                # Calculate evaluation metrics
                results.append({
                    "Model": name,
                    "Accuracy": accuracy_score(
                        y_test, predictions
                    ),
                    "Precision (weighted)": precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    ),
                    "Recall (weighted)": recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    ),
                    "F1 Score (weighted)": f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0
                    )
                })

            # Display results
            results_df = pd.DataFrame(results)

            st.subheader("📋 Evaluation Results")

            st.dataframe(
                results_df.style.format({
                    "Accuracy": "{:.2%}",
                    "Precision (weighted)": "{:.2%}",
                    "Recall (weighted)": "{:.2%}",
                    "F1 Score (weighted)": "{:.2%}"
                }),
                width="stretch"
            )

            # Accuracy chart
            st.subheader("📊 Accuracy Comparison")

            st.bar_chart(
                results_df.set_index("Model")[["Accuracy"]]
            )

            # Saved model confirmation
            st.subheader("💾 Saved Models")

            st.success(
                "Both model pipelines were saved successfully."
            )

            st.write(
                f"Saved model directory: `{output_dir}`"
            )

            st.write("- `logistic_regression.joblib`")
            st.write("- `random_forest.joblib`")

            st.caption(
                "These files contain fitted preprocessing and "
                "model pipelines. Only load model files from "
                "trusted sources. These experimental models "
                "are not validated for clinical use."
            )

    except FileNotFoundError:
        st.error(
            "Dataset file not found. Check "
            "data/patient_records.csv."
        )

    except Exception as error:
        st.error(
            f"Model evaluation failed: {error}"
        )

# --------------------------------------------------
# 5. LOAD A SAVED MODEL WITHOUT RETRAINING
# --------------------------------------------------

st.divider()

st.header("📂 Load a Saved Model")

st.write(
    "Load a previously trained model from the outputs folder. "
    "Loading does not train the model again."
)

saved_models = {
    "Logistic Regression": (
        output_dir / "logistic_regression.joblib"
    ),
    "Random Forest": (
        output_dir / "random_forest.joblib"
    )
}

available_models = {
    name: path
    for name, path in saved_models.items()
    if path.is_file()
}

if not available_models:
    st.info(
        "No saved models found. First go to Machine Learning "
        "Model Evaluation and click Train and Compare Models."
    )

else:
    selected_model_name = st.selectbox(
        "Choose a saved model",
        list(available_models.keys())
    )

    if st.button("Load Saved Model"):
        try:
            model_path = available_models[selected_model_name]

            # Load an existing trusted model file.
            # This does not retrain the model.
            loaded_model = joblib.load(model_path)

            st.session_state["loaded_model"] = loaded_model
            st.session_state["loaded_model_name"] = (
                selected_model_name
            )

            st.success(
                f"{selected_model_name} loaded successfully!"
            )

        except Exception as error:
            st.error(f"Could not load the model: {error}")

    if "loaded_model" in st.session_state:
        st.subheader("Loaded Model Details")

        st.write(
            "Model name:",
            st.session_state["loaded_model_name"]
        )

        loaded_model = st.session_state["loaded_model"]

        if hasattr(loaded_model, "named_steps"):
            st.write(
                "Pipeline components:",
                list(loaded_model.named_steps.keys())
            )

        st.success(
            "The saved model is ready to be reused. "
            "No training was performed during loading."
        )

st.caption(
    "Security note: Load only trusted model files that "
    "you created or obtained from a verified source."
)

# --------------------------------------------------
# 6. TEST A SAVED MODEL ON A DATASET ROW
# --------------------------------------------------

st.divider()

st.header("🧪 Saved Model Prediction Demo")

st.write(
    "Select a dataset row and run the loaded model "
    "to demonstrate how a saved model produces an output."
)

if "loaded_model" not in st.session_state:
    st.info(
        "First load a saved model in the section above."
    )

else:
    try:
        demo_df = pd.read_csv(data_path)

        # Use the same input columns as model training
        feature_df = demo_df.drop(
            columns=["Reaction_Score", "Patient_ID"],
            errors="ignore"
        )

        row_number = st.selectbox(
            "Select a dataset row",
            options=list(range(len(demo_df))),
            format_func=lambda i: f"Dataset row {i + 1}"
        )

        st.subheader("Selected Input Features")

        st.dataframe(
            feature_df.iloc[[row_number]],
            width="stretch"
        )

        if st.button("Run Demo Prediction"):
            loaded_model = st.session_state["loaded_model"]

            selected_row = feature_df.iloc[[row_number]]

            prediction = loaded_model.predict(selected_row)

            st.subheader("Prediction Output")

            st.write(
                "Model:",
                st.session_state["loaded_model_name"]
            )

            st.write(
                "Predicted Reaction_Score label:",
                str(prediction[0])
            )

            if pd.notna(
                demo_df.iloc[row_number]["Reaction_Score"]
            ):
                st.write(
                    "Recorded dataset label:",
                    str(
                        demo_df.iloc[row_number]["Reaction_Score"]
                    )
                )

            st.caption(
                "This is a demonstration on an existing dataset row. "
                "It is not an independently validated prediction, "
                "clinical assessment, or dosage recommendation."
            )

    except Exception as error:
        st.error(f"Prediction demo failed: {error}")
