# Moderna mRNA Clinical Safety & Dosage Intelligence — Educational Prototype

This is a staged student project scaffold for VS Code. It begins with Week 1:
dataset profiling, basic cleaning, exploratory plots, and a comparison of
Logistic Regression and Random Forest classifiers.

## Important safety and data note

The supplied files describe a course case study. The CSV has 500 records and
a binary `Reaction_Score` label. The dataset's provenance and label definition
must be verified before making claims about predictive validity. The values
must not be assumed to represent actual Moderna dosing, real patient outcomes,
or validated clinical guidance. This application is for education only and
must not recommend, prescribe, or change vaccine doses or replace a clinician.

The supplied guideline document explicitly says it is course reference
material, not real clinical or regulatory guidance. Treat it as unverified
example content for a RAG demonstration only.

## Folder layout

- `data/patient_records.csv`: supplied CSV
- `data/mrna_safety_guidelines.txt`: extracted course reference document
- `src/risk_model.py`: Week 1 starter script
- `outputs/`: generated model and evaluation files (created when the script runs)

## Run in VS Code on Windows

Open this folder in VS Code. In Terminal > New Terminal, activate the virtual
environment you created:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies if necessary:

```powershell
python -m pip install -r requirements.txt
```

Run Week 1:

```powershell
python src/risk_model.py
```

The script prints dataset statistics, creates a cleaned inspection copy, saves
three exploratory plots, compares two baseline classifiers, and writes
evaluation reports and a model file under `outputs/`.

## Week 1 caveats

- Missing `Biomarker_Level` values are imputed with the median in the cleaned
  inspection copy and within the training pipeline.
- Missing `Prior_Reaction_History` is treated as an `Unknown` category by the
  training pipeline's most-frequent imputer only if missing values remain.
- `Patient_ID` is excluded from model features.
- The model is evaluated on a held-out stratified split, but the small,
  undocumented case-study dataset is not evidence of real-world clinical
  performance. Do not use the model for patient care or dosage decisions.
