# Mycotoxin Prediction Pipeline

A machine learning pipeline to predict mycotoxin levels (DON concentration) in corn samples using hyperspectral imaging data.

## Project Overview

This project implements a complete machine learning pipeline that:
- Preprocesses hyperspectral imaging data of corn samples
- Builds and optimizes regression models to predict DON concentration
- Provides a user-friendly interface for making predictions
- Includes model interpretability analysis

## Installation

### Prerequisites

This project uses `uv` as the package manager. To install `uv`:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or on Windows PowerShell
irm https://astral.sh/uv/install.ps1 | iex
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/fa-anony-mous/Machine-Learning-Pipeline
cd Machine-Learning-Pipeline 
```

2. Create and activate a virtual environment using `uv`:
```bash
uv venv
uv venv activate
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python main.py
```

This will start both the Streamlit frontend and FastAPI backend concurrently.

## Project Structure

```
├── Data Science & Analytics/
│   ├── EDA.ipynb                # Exploratory data analysis notebook
│   ├── MLE-Assignment.csv       # Original dataset
│   └── outputs/                 # Contains correlation matrices and other EDA outputs
│       ├── X_data.csv           # Processed feature data
│       └── Y_data.csv           # Processed target data
│
├── Machine Learning/
│   ├── ML_trainer.ipynb         # Traditional ML models (ensemble methods)
│   ├── DL_trainer.ipynb         # Deep learning models
│   ├── interpretability.py      # SHAP analysis for model interpretability
│   ├── model_artifacts/         # Saved model weights and parameters
│   └── interpretability_analysis/ # SHAP visualization outputs
│
├── streamlit_app/
│   ├── app.py                   # Streamlit frontend
│   └── backend/                 # FastAPI backend
│       ├── api/                 # API endpoints
│       ├── core/                # Core functionality
│       ├── models/              # Data models
│       └── services/            # Business logic
│
├── main.py                      # Entry point that runs both Streamlit and FastAPI
└── requirements.txt             # Project dependencies
```

## Workflow

1. **Data Preprocessing**: The `EDA.ipynb` notebook performs exploratory data analysis, preprocessing, normalization, and anomaly detection on the hyperspectral data, producing `X_data.csv` and `Y_data.csv`.

2. **Model Training**:
   - `ML_trainer.ipynb`: Implements ensemble models (Random Forest, Gradient Boosting, etc.) and meta-models (Stacking, Voting)
   - `DL_trainer.ipynb`: Implements neural network models
   - Models are evaluated using R², MAE, and MSE metrics

3. **Model Interpretability**: The `interpretability.py` script uses SHAP values to explain model predictions.

4. **Deployment**:
   - FastAPI backend provides prediction endpoints
   - Streamlit frontend offers a user-friendly interface

## Assignment Details

This project was developed as part of a Machine Learning Engineer assignment to test:
- Preprocessing of complex hyperspectral data
- Building and optimizing regression models
- Designing modular, production-ready code

Original assignment details: [Google Doc](https://docs.google.com/document/d/140oJBWeGuWFW9fHJfGOjjTyf9YOXimpzY5c0nOAWUOw/edit?tab=t.0)
