# Heart Disease Prediction

**Goal:** Build, evaluate, and explain a simple machine‑learning model to predict the presence of heart disease using the UCI Cleveland dataset. This project showcases a full ML workflow—from data ingestion and exploratory analysis to model training, evaluation, and interpretation—using clean, modular code and clear visualizations.


| Model               | Accuracy | Precision | Recall | ROC AUC |
|---------------------|---------:|----------:|-------:|--------:|
| LogisticRegression  | 0.869    | 0.833     | 0.893  | 0.958   |
| RandomForest        | 0.885    | 0.839     | 0.929  | **0.962** |
| XGBoost             | 0.902    | 0.867     | 0.929  | 0.953   |


## Quickstart

### 1. Clone the repository and set up a virtual environment

git clone github.com/Mehraan13/heart-disease-prediction  <br>
cd heart-disease-prediction  <br>
python3 -m venv venv venv\Scripts\activate  <br>

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run Pipiline Scripts
python scripts/ingest.py        # Download and save raw data to data/heart.csv  <br>
python scripts/preprocess.py    # Clean, encode, scale, split data <br>
python scripts/train.py         # Train and save Random Forest model <br>
python scripts/evaluate.py      # Evaluate model and generate visual reports <br>

### 4. View Results

Raw Data: data/heart.csv         <br>
Saved Models: models/<br>
Plots and Metrics: reports/<br>
