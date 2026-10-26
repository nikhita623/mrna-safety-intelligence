
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------
# PROJECT PATHS AND SETTINGS
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "patient_records.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Reaction_Score"
ID_COLUMN = "Patient_ID"

st.set_page_config(
    page_title="mRNA Safety Intelligence",
    page_icon="🧬",
    layout="wide",
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


df = load_data()

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("🧬 Project Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Home",
        "Dataset Explorer",
        "Self-Attention",
        "Model Evaluation",
    ],
)

st.sidebar.divider()
st.sidebar.caption("Educational prototype only")

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "Home":
    st.title("🧬 mRNA Safety Intelligence")
    st.subheader("Clinical Safety & Dosage Intelligence — Academic Prototype")

    st.write(
        """
        This project demonstrates how a dataset can be explored and how
        machine-learning models can be evaluated on recorded reaction scores.
        It also demonstrates a simplified self-attention mechanism.
        """
    )

    st.warning(
        "Educational use only. This prototype is not clinically validated "
        "and must not be used to determine vaccine dosage, diagnose a patient, "
        "or make medical or patient-safety decisions."
    )

    if df is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Dataset records", len(df))

        with col2:
            st.metric("Dataset columns", len(df.columns))

        with col3:
            st.metric(
                "Missing values",
                int(df.isna().sum().sum()),
            )

        st.markdown("### Dataset preview")
        st.dataframe(df.head(10), width="stretch")

    else:
        st.error(
            "Dataset not found. Make sure this file exists: "
            "data/patient_records.csv"
        )

    st.markdown("### Project modules")
    st.markdown(
        """
        - **Dataset Explorer:** inspect records and missing values.
        - **Self-Attention:** view an illustrative attention heatmap.
        - **Model Evaluation:** train, compare, save, load and test models.
        """
    )

# --------------------------------------------------
# DATASET EXPLORER
# --------------------------------------------------
elif page == "Dataset Explorer":
    st.title("📊 Dataset Explorer")

    if df is None:
        st.error("Could not find data/patient_records.csv.")
        st.stop()

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing cells", int(df.isna().sum().sum()))

    st.markdown("### Search records")
    search_text = st.text_input(
        "Search across all columns",
        placeholder="Enter an ID, condition, or value",
    )

    if search_text:
        mask = df.astype(str).apply(
            lambda column: column.str.contains(
                search_text,
                case=False,
                na=False,
                regex=False,
            )
        ).any(axis=1)
        filtered_df = df.loc[mask]
    else:
        filtered_df = df

    st.write(f"Showing {len(filtered_df)} matching records")
    st.dataframe(filtered_df, width="stretch")

    st.markdown("### Missing values by column")
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        st.success("No missing values were found.")
    else:
        st.bar_chart(missing)

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    if numeric_columns:
        st.markdown("### Explore a numerical column")
        selected_column = st.selectbox(
            "Select a numerical column",
            numeric_columns,
        )

        st.bar_chart(df[selected_column].dropna().value_counts().sort_index())

    st.markdown("### Dataset information")
    st.write("Column data types")
    st.dataframe(
        pd.DataFrame({
            "Column": df.columns,
            "Data type": [str(dtype) for dtype in df.dtypes],
            "Missing values": [int(value) for value in df.isna().sum()],
        }),
        width="stretch",
    )

    if TARGET in df.columns:
        st.markdown("### Recorded reaction-score distribution")
        st.bar_chart(df[TARGET].value_counts().sort_index())

# --------------------------------------------------
# SELF-ATTENTION PAGE
# --------------------------------------------------
elif page == "Self-Attention":
    st.title("🧠 Self-Attention Demonstration")

    st.write(
        """
        This page demonstrates a simplified self-attention mechanism using
        synthetic numerical data. The heatmap is illustrative; it is not a
        validated clinical explanation of vaccine reactions.
        """
    )

    try:
        import torch
        import torch.nn as nn
        import seaborn as sns

        class SelfAttention(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.query = nn.Linear(input_dim, input_dim)
                self.key = nn.Linear(input_dim, input_dim)
                self.value = nn.Linear(input_dim, input_dim)

            def forward(self, x):
                q = self.query(x)
                k = self.key(x)
                v = self.value(x)

                scores = torch.matmul(
                    q,
                    k.transpose(-2, -1),
                ) / (x.shape[-1] ** 0.5)

                weights = torch.softmax(scores, dim=-1)
                output = torch.matmul(weights, v)

                return output, weights

        torch.manual_seed(42)

        sample_count = st.slider(
            "Number of synthetic samples",
            min_value=3,
            max_value=12,
            value=6,
        )

        input_data = torch.randn(sample_count, 4)
        attention_model = SelfAttention(input_dim=4)
        _, attention_weights = attention_model(input_data)

        weights_array = attention_weights.detach().numpy()

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            weights_array,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            xticklabels=[f"Sample {i + 1}" for i in range(sample_count)],
            yticklabels=[f"Sample {i + 1}" for i in range(sample_count)],
            ax=ax,
        )
        ax.set_title("Illustrative Self-Attention Weights")
        ax.set_xlabel("Samples being attended to")
        ax.set_ylabel("Input samples")

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Values show attention weights from synthetic data, not medical risk."
        )

    except ImportError:
        st.error(
            "A required library is missing. Activate your project environment "
            "and run: python -m pip install torch seaborn matplotlib"
        )

# --------------------------------------------------
# MODEL EVALUATION PAGE
# --------------------------------------------------
elif page == "Model Evaluation":
    st.title("🤖 Model Training & Evaluation")

    st.warning(
        "Academic demonstration only. Model scores are based on this dataset "
        "and do not establish clinical validity or safe vaccine dosage."
    )

    if df is None:
        st.error("Could not find data/patient_records.csv.")
        st.stop()

    if TARGET not in df.columns:
        st.error(f"The dataset must contain a '{TARGET}' column.")
        st.stop()

    # Prepare features and target.
    # Do not use the patient ID as a predictive feature.
    usable_df = df.dropna(subset=[TARGET]).copy()

    excluded_columns = [TARGET]
    if ID_COLUMN in usable_df.columns:
        excluded_columns.append(ID_COLUMN)

    X = usable_df.drop(columns=excluded_columns)
    y = usable_df[TARGET]

    if X.empty:
        st.error("No feature columns are available for training.")
        st.stop()

    class_counts = y.value_counts()

    if len(class_counts) < 2 or class_counts.min() < 2:
        st.error(
            "Training requires at least two target classes, with at least "
            "two records in each class."
        )
        st.stop()

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = [
        column for column in X.columns
        if column not in numeric_features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    st.markdown("### Dataset used for training")
    st.write(f"Records with a recorded target: **{len(usable_df)}**")
    st.write(f"Input features: **{len(X.columns)}**")
    st.write("Features used:", ", ".join(X.columns))

    st.markdown("### Train and compare models")

    test_size = st.slider(
        "Test data percentage",
        min_value=20,
        max_value=40,
        value=25,
        step=5,
    )

    if st.button("🚀 Train and Compare Models", type="primary"):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size / 100,
                random_state=42,
                stratify=y,
            )

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
                "Random Forest": RandomForestClassifier(
                    n_estimators=150,
                    random_state=42,
                    class_weight="balanced",
                ),
            }

            results = []
            saved_paths = {}

            progress = st.progress(0)

            for index, (model_name, estimator) in enumerate(models.items()):
                # Each model gets its own preprocessing pipeline.
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("model", estimator),
                    ]
                )

                pipeline.fit(X_train, y_train)
                predictions = pipeline.predict(X_test)

                results.append({
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test, predictions),
                    "Precision (weighted)": precision_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                    "Recall (weighted)": recall_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                    "F1-score (weighted)": f1_score(
                        y_test,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    ),
                })

                filename = (
                    "logistic_regression.joblib"
                    if model_name == "Logistic Regression"
                    else "random_forest.joblib"
                )
                model_path = OUTPUT_DIR / filename
                joblib.dump(pipeline, model_path)
                saved_paths[model_name] = str(model_path)

                progress.progress((index + 1) / len(models))

            results_df = pd.DataFrame(results).set_index("Model")

            st.session_state["evaluation_results"] = results_df
            st.session_state["saved_model_paths"] = saved_paths
            st.session_state["test_size"] = test_size

            st.success("Training and evaluation completed. Models were saved.")

        except Exception as error:
            st.error(f"Training could not be completed: {error}")

    if "evaluation_results" in st.session_state:
        st.markdown("### Model comparison results")

        results_df = st.session_state["evaluation_results"]
        st.dataframe(
            results_df.style.format("{:.3f}"),
            width="stretch",
        )

        st.markdown("### Metric comparison")
        st.bar_chart(results_df)

        st.caption(
            "These metrics measure performance on one held-out split. "
            "Results can vary with different data splits and do not prove "
            "clinical usefulness."
        )

    # --------------------------------------------------
    # LOAD A SAVED MODEL
    # --------------------------------------------------
    st.divider()
    st.markdown("### Load a saved model")

    model_files = {
        "Logistic Regression": OUTPUT_DIR / "logistic_regression.joblib",
        "Random Forest": OUTPUT_DIR / "random_forest.joblib",
    }

    available_models = [
        name for name, path in model_files.items()
        if path.exists()
    ]

    if not available_models:
        st.info("Train the models above first. Saved models will appear here.")
    else:
        selected_model_name = st.selectbox(
            "Choose a saved model",
            available_models,
        )

        if st.button("Load Selected Model"):
            try:
                loaded_pipeline = joblib.load(model_files[selected_model_name])
                st.session_state["loaded_model"] = loaded_pipeline
                st.session_state["loaded_model_name"] = selected_model_name
                st.success(f"{selected_model_name} loaded successfully.")
            except Exception as error:
                st.error(f"Could not load the saved model: {error}")

    # --------------------------------------------------
    # PREDICTION DEMONSTRATION
    # --------------------------------------------------
    st.divider()
    st.markdown("### Saved Model Prediction Demo")

    if "loaded_model" not in st.session_state:
        st.info("Load a saved model above to enable the demonstration.")
    else:
        st.write(
            f"Loaded model: **{st.session_state['loaded_model_name']}**"
        )

        if len(usable_df) == 0:
            st.info("No usable records are available.")
        else:
            row_number = st.number_input(
                "Select a dataset row number",
                min_value=0,
                max_value=len(usable_df) - 1,
                value=0,
                step=1,
            )

            selected_row = usable_df.iloc[int(row_number)]
            sample_features = X.loc[[selected_row.name]]

            st.write("Selected input record")
            st.dataframe(sample_features, width="stretch")

            if st.button("Run Prediction Demo"):
                try:
                    loaded_pipeline = st.session_state["loaded_model"]
                    predicted_label = loaded_pipeline.predict(
                        sample_features
                    )[0]

                    st.metric(
                        "Predicted recorded reaction-score class",
                        str(predicted_label),
                    )
                    st.write(
                        "Recorded dataset label:",
                        str(selected_row[TARGET]),
                    )
                    st.caption(
                        "This is a demonstration against an existing dataset "
                        "row, not a medical assessment or recommendation."
                    )

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

    st.info(
        "Saved model files are stored in the project's outputs folder."
    )
