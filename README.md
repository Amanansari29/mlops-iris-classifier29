# MLOps Iris Classifier

A sample ML project demonstrating data pipelines, feature stores, and experiment
tracking for an Iris classifier.

## Setup

Use Python 3.9 or newer in a virtual environment. Python 3.11 is recommended
for compatibility with the pinned MLflow 2.17.2 dependencies:

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the baseline model:

```bash
python src/train.py
```

## MLflow experiment tracking

Ensure `data/processed/iris_features.csv` exists. If needed, generate it by
running the DVC pipeline:

```bash
dvc repro
```

Start the MLflow tracking server in one terminal:

```bash
mlflow ui --backend-store-uri ./mlflow-tracking --host 127.0.0.1 --port 5000
```

The `mlflow-tracking/` directory holds the local experiment metadata and
artifacts. Keep it separate from `mlruns/`, which may contain artifacts from
older runs; using `mlruns/` as the backend store can make MLflow report
malformed experiments when those artifact directories lack `meta.yaml`.

In a second terminal, activate the same environment, configure the tracking
URI, and run all three model experiments:

```bash
# PowerShell
$env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

# Git Bash
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000

python train_with_mlflow.py
```

Open <http://127.0.0.1:5000> to compare the runs, metrics, confusion-matrix
images, and logged model artifacts. A fresh experiment contains three runs;
subsequent executions append three additional runs so earlier experiment
history is retained.

Register the highest-F1 run and load the staged model:

```bash
# Run these commands from the repository root
python register_best_model.py
python load_registered_model.py
```

The training script uses a fixed, stratified 80/20 holdout split. With the
repository's current 149-row feature dataset, the measured macro F1 scores are:

| Model | Macro F1 |
|---|---:|
| Logistic Regression | 1.0000 |
| Random Forest (50 trees, depth 3) | 0.9666 |
| Random Forest (200 trees, unlimited depth) | 0.9666 |

The scripts select and register the actual highest-scoring run; they do not
force a particular model to win. Rankings can change if the input data or
evaluation procedure changes.
