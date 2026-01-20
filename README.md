# Tox21 Toxicity Prediction (D‑MPNN Ensemble + RDKit descriptors)

A reproducible pipeline and API for predicting toxicity across the 12 Tox21 endpoints using an ensemble of Directed Message Passing Neural Networks (D‑MPNNs) augmented with RDKit global descriptors.

This repository contains code to:
- train an ensemble of D‑MPNN models on a provided Tox21-style CSV,
- evaluate model performance (ROC-AUC),
- run a prediction API (FastAPI / Uvicorn),
- run single-model/ensemble predictions from script.

Key ideas:
- Molecules (SMILES) are converted to RDKit molecules and featurized into atom/bond features + global RDKit descriptors.
- A D‑MPNN processes bond-level messages and produces per-molecule embeddings.
- Global descriptors are concatenated and used by a feed-forward head for multi-task prediction (12 toxicity endpoints).
- An ensemble of models (different random seeds) is used and probabilities are averaged to produce robust predictions.

---

Table of contents
- Features
- Quickstart (Docker / Local)
- Data format
- Training
- Serving / API usage
- Using the predictor programmatically
- How it works (pipeline)
- Evaluation & testing
- Important implementation notes
- Files of interest
- Contributing / Contact

---

Features
- Multi-task toxicity prediction across 12 Tox21 endpoints:
  NR-AhR, NR-AR, NR-AR-LBD, NR-Aromatase, NR-ER, NR-ER-LBD,
  NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53
- Ensemble training (default seeds: 42, 43, 44, 45, 46)
- RDKit-derived global descriptors are used in addition to graph features
- Scaffold split (90/10 train/val) for more realistic generalization
- FastAPI server for easy serving (endpoints: /predict and /metadata)
- Dockerfile for containerized deployment (installs system deps required by RDKit)

---

Quickstart

Prerequisites
- Docker (recommended) OR Python 3.9 and conda (recommended for RDKit).
- GPU recommended for training but CPU is supported for inference.

Using Docker (recommended)
1. Build:
   docker build -t tox21 .

2. Run (exposes port 7860 as in Dockerfile):
   docker run --rm -p 7860:7860 -v $(pwd)/checkpoints:/app/checkpoints tox21

3. The container runs uvicorn as the default command (see Dockerfile). The HTTP API will be available at http://localhost:7860.

Local setup with conda (if not using Docker)
1. Create environment and install RDKit:
   conda create -n tox21 python=3.9 -y
   conda activate tox21
   conda install -c conda-forge rdkit -y

2. Install Python deps:
   pip install -r requirements.txt

3. Run the API (development):
   uvicorn app:app --host 0.0.0.0 --port 8000

---

Data format

The training/validation loader expects a CSV named `tox21.csv` (or path passed to the loader). The CSV must contain:
- a column named `smiles` with canonical SMILES strings
- the 12 task columns (same names as listed above) containing numeric labels (0/1/NaN for unknown)

Example header:
smiles,NR-AhR,NR-AR,NR-AR-LBD,NR-Aromatase,NR-ER,NR-ER-LBD,NR-PPAR-gamma,SR-ARE,SR-ATAD5,SR-HSE,SR-MMP,SR-p53

Place `tox21.csv` at the repo root or pass its path to the data loader.

---

Training

Basic training command (example):
- (After environment setup)
- python train.py --epochs 50 --batch-size 32 --lr 1e-3

Notes:
- The training script uses a scaffold split (90/10 train/val).
- Ensemble seeds used by default are [42, 43, 44, 45, 46]. The training loop calls train_model for each seed to create an ensemble.
- Global features (RDKit descriptors) are standardized with a scaler; the scaler is saved (scaler.pkl).
- Model checkpoint filenames follow the pattern model_seed_{seed}.pt and are expected by the predictor (predict.py searches for model_seed_*.pt).
- The training loop computes task-wise ROC‑AUC and logs best validation AUCs for each seed; an average ensemble AUC is reported at the end.

If you change hyperparameters, add a command-line option for them or edit parse_args in train.py.

---

Serving / API usage

Start server:
- uvicorn app:app --host 0.0.0.0 --port 8000
(Or use the Dockerfile which runs uvicorn on port 7860.)

Endpoints
- POST /predict
  - Body: JSON with shape { "smiles": ["SMILES1", "SMILES2", ...] }
  - Example:
    curl -X POST "http://localhost:8000/predict" \
      -H "Content-Type: application/json" \
      -d '{"smiles": ["CCO", "c1ccccc1"]}'

  - Response: JSON mapping each input SMILES to a dictionary of task probabilities.
    Example response (illustrative):
    {
      "CCO": {
        "NR-AhR": 0.12,
        "NR-AR": 0.03,
        ...
      },
      "c1ccccc1": { ... }
    }

  Behavior:
  - The API uses the Tox21Predictor class which loads all models matching `model_seed_*.pt` in the model directory and the scaler (`scaler.pkl`).
  - Predictions are produced by averaging sigmoid outputs across ensemble members (consensus).
  - If models are not loaded or checkpoints are missing, the endpoint returns HTTP 503.

- GET /metadata
  - Returns model metadata:
    {
      "model": "D-MPNN Ensemble (RDKit Augmented)",
      "tasks": [...],
      "ensemble_size": <n_models_loaded>,
      "version": "2.0.0",
      "author": "Antigravity",
      "split": "Scaffold (90/10 Train/Val)"
    }

---

Using the predictor programmatically

- Run quick test from predict.py
  python predict.py
  This instantiates Tox21Predictor() and runs a small example list (see main guard in predict.py).

- Example usage in code:
  from predict import Tox21Predictor
  predictor = Tox21Predictor(model_dir="checkpoints")
  results = predictor.predict(["CCO", "c1ccccc1"])
  # results is a dict mapping SMILES -> {task: probability}

---

How it works (high level)

1. Featurization
   - Each SMILES is parsed by RDKit to a Mol.
   - Atom and bond features are built (one-hot encodings and physicochemical flags).
   - Global molecular descriptors (e.g., RDKit Descriptors) are computed and standardized.

2. Graph representation
   - MolGraph builds atom/bond arrays and connectivity (a2b, b2a, etc).
   - BatchMolGraph collates multiple MolGraph instances into a batch for efficient GPU processing.

3. D‑MPNN (model.py)
   - Bond-centric message passing updates bond messages for a fixed number of steps (depth).
   - A readout aggregates information into molecular embeddings.
   - Global descriptors are concatenated to the learned embedding if present.

4. Prediction head
   - The aggregated representation is passed through dense layers to produce logits per task.
   - Sigmoid produces probabilities for each of the 12 tasks.

5. Ensemble
   - Multiple models (different seeds / weights) provide probabilistic estimates; the predictor averages probabilities across the ensemble for the final output.

---

Evaluation & testing

- test_power.py contains a local evaluation script which:
  - loads predictions, computes task-wise ROC-AUC (only if positives are present),
  - prints average ROC-AUC and a simple strength rating.

- Recommended metric: ROC‑AUC per task and average across tasks.

---

Important implementation notes & troubleshooting

- RDKit: RDKit can be tricky to install via pip on some platforms. Use conda with the conda-forge channel when developing locally:
  conda install -c conda-forge rdkit

- Checkpoints: predict.py expects model checkpoints named by pattern `model_seed_*.pt` and a scaler file `scaler.pkl`. Ensure these are present in the working dir or pass model_dir to Tox21Predictor.

- Invalid SMILES: Molecules that fail RDKit parsing are filtered during data loading. For prediction, invalid SMILES will be handled (the predictor logs or returns an informative entry); check console logs if some inputs are missing or return null.

- CPU inference: If CUDA is not available, models are loaded with map_location='cpu'.

- Docker port difference: Dockerfile exposes 7860 and uses `CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]`. If you run uvicorn manually, the default port in app's `__main__` is 8000.

- Reproducibility: Seeds and scaffold-splitting are used to improve reproducibility. Ensemble seeds are listed in train.py (SEEDS = [42, 43, 44, 45, 46]).

---

Files of interest
- app.py — FastAPI server, endpoints, startup model loading
- predict.py — Tox21Predictor class, ensemble loading & predict API
- train.py — training loop, train_model, train_ensemble, collate function
- data.py — data loading, RDKit descriptors, scaffold split, Tox21Dataset
- model.py — MolGraph, BatchMolGraph, D‑MPNN implementation
- test_power.py — evaluation script
- requirements.txt — Python dependencies
- Dockerfile — containerized environment and entrypoint

---

Contributing
- Bug reports, issues, and enhancements are welcome. Open an issue with details and a minimal reproduction where appropriate.
- For code changes: fork, create a feature branch, add tests where relevant, and open a pull request.

---

Contact / Credits
- shushank and antigravity for those cool comments and code. 

---

License
- MIT

---
