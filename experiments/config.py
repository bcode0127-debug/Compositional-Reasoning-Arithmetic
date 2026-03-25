# Configuration for composition analysis (CP1-CP5)

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DATASETS_DIR = PROJECT_ROOT / "datasets"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Model checkpoints
CHECKPOINTS = {
    'transformer_study1': RESULTS_DIR / "transformer/study1/train_best_model.pt",
    'transformer_study2': RESULTS_DIR / "transformer/study2/train_best_model.pt",
    'lstm_study1': RESULTS_DIR / "lstm_baseline/study1/train_best_model.pt",
    'lstm_study2': RESULTS_DIR / "lstm_baseline/study2/train_best_model.pt",
}

# Datasets
DATASETS = {
    'study1_ood': DATASETS_DIR / "study1/ood.json",
    'study1_val': DATASETS_DIR / "study1/val.json",
    'study2_ood': DATASETS_DIR / "study2/ood.json",
    'study2_val': DATASETS_DIR / "study2/val.json",
}

# Analysis output directories
ANALYSIS_DIRS = {
    'cp4_operations': EXPERIMENTS_DIR / "results/operation_breakdown",
    'cp5_traces': EXPERIMENTS_DIR / "results/failure_traces",
}

# Create directories
for d in ANALYSIS_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

ANALYSIS_CONFIG = {
    'device': 'cpu',
    'num_failure_examples': 5,
}