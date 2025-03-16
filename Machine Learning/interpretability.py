import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pickle
import json
import os
from datetime import datetime

# Create results directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"Machine Learning/interpretability_analysis/run_{timestamp}"
os.makedirs(RESULTS_DIR, exist_ok=True)

print(f"SHAP results will be saved to: {RESULTS_DIR}")

# Define the model class
class DirectNonLinearNN(torch.nn.Module):
    """DON prediction neural network model"""
    def __init__(self, input_dim=447, output_range=12.0):
        super(DirectNonLinearNN, self).__init__()
        # A single linear layer
        self.fc = torch.nn.Linear(input_dim, 1)
        self.output_range = output_range
    
    def forward(self, x):
        # Apply linear transformation
        linear_output = self.fc(x)
        
        # Apply sigmoid and scale
        non_linear_output = torch.sigmoid(linear_output) * self.output_range
        
        return non_linear_output

# Load model and scaler
def load_model(model_dir="Machine Learning/model_artifacts"):
    # Load model architecture info
    with open(os.path.join(model_dir, "model_info.json"), "r") as f:
        model_info = json.load(f)
    
    # Create model with same architecture
    model = DirectNonLinearNN(
        input_dim=model_info.get("input_dim", 447),
        output_range=model_info.get("output_range", 12.0)
    )
    
    # Load model weights
    model.load_state_dict(torch.load(
        os.path.join(model_dir, "don_prediction_model.pt"), 
        map_location=torch.device('cpu')
    ))
    model.eval()  # Set to evaluation mode
    
    # Load scaler
    with open(os.path.join(model_dir, "feature_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    
    return model, scaler

# Load data
def load_data(X_path="Machine Learning/X_data.csv", y_path="Machine Learning/Y_data.csv", sample_size=None):
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    
    # Sample a subset if requested
    if sample_size and sample_size < len(X):
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[indices]
        y_sample = y.iloc[indices]
        return X_sample, y_sample
    
    return X, y

# SHAP analysis
def run_shap_analysis(model, scaler, X, feature_names=None, num_display=20):
    # Create a model wrapper for SHAP
    class ShapModelWrapper:
        def __init__(self, model, scaler):
            self.model = model
            self.scaler = scaler
            
        def __call__(self, X):
            # Scale the input features
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
            
            # Get predictions
            with torch.no_grad():
                preds = self.model(X_tensor).numpy()
            
            return preds
    
    model_wrapper = ShapModelWrapper(model, scaler)
    
    # Create the SHAP explainer
    explainer = shap.KernelExplainer(model_wrapper, shap.sample(X.values, 50))
    
    # Calculate SHAP values for the first 100 samples (or fewer if X has fewer rows)
    n_samples = min(100, X.shape[0])
    shap_values = explainer.shap_values(X.values[:n_samples])
    
    # If feature names weren't provided, use column names from X
    if feature_names is None and isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X.values[:n_samples], feature_names=feature_names, show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_summary_plot.png"))
    plt.close()
    
    # Bar plot of mean absolute SHAP values
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X.values[:n_samples], feature_names=feature_names, plot_type="bar", show=False)
    plt.title("Mean Impact on Model Output Magnitude")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_bar_plot.png"))
    plt.close()
    
    # Get the top features by importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Ensure mean_abs_shap is 1-dimensional
    if hasattr(mean_abs_shap, 'shape') and len(mean_abs_shap.shape) > 1:
        mean_abs_shap = mean_abs_shap.flatten()
    
    feature_importance = pd.DataFrame({
        'Feature': feature_names if feature_names else [f'Feature_{i}' for i in range(X.shape[1])],
        'Importance': mean_abs_shap
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Display top features
    print(f"Top {num_display} most important features:")
    print(feature_importance.head(num_display))
    
    # Save to CSV
    feature_importance.to_csv(os.path.join(RESULTS_DIR, "shap_feature_importance.csv"), index=False)
    
    # Create waterfall plots for a few examples
    for i in range(min(5, n_samples)):
        plt.figure(figsize=(10, 6))
        # Fix: Reshape the SHAP values to be 1-dimensional for the waterfall plot
        if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
            if shap_values.shape[1] == 1:
                # If shape is (n, 1), flatten to (n,)
                instance_shap_values = shap_values[i].flatten()
            else:
                instance_shap_values = shap_values[i]
        else:
            instance_shap_values = shap_values[i]
            
        shap.plots.waterfall(shap.Explanation(
            values=instance_shap_values, 
            base_values=explainer.expected_value, 
            data=X.values[i], 
            feature_names=feature_names
        ), show=False)
        plt.title(f"SHAP Waterfall Plot for Instance {i}")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"shap_waterfall_instance_{i}.png"))
        plt.close()
    
    # Create a force plot for the first few samples
    try:
        plt.figure(figsize=(20, 3))
        # Fix: Handle potential shape issues for force plot too
        if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1 and shap_values.shape[1] == 1:
            # If shape is (n, 1), reshape to (n,)
            plot_shap_values = shap_values[:5].reshape(5, -1)
        else:
            plot_shap_values = shap_values[:5]
            
        force_plot = shap.force_plot(
            explainer.expected_value, 
            plot_shap_values, 
            X.values[:5],
            feature_names=feature_names,
            show=False
        )
        shap.save_html(os.path.join(RESULTS_DIR, "shap_force_plot.html"), force_plot)
    except Exception as e:
        print(f"Warning: Could not create force plot: {e}")
    
    return shap_values, feature_importance

# Create a summary report
def create_summary_report(shap_importance):
    with open(os.path.join(RESULTS_DIR, "shap_summary.txt"), "w") as f:
        f.write("DON Prediction Model SHAP Analysis\n")
        f.write("=================================\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Top 20 Features by SHAP Importance:\n")
        for i, (feature, importance) in enumerate(zip(
            shap_importance.head(20)['Feature'], 
            shap_importance.head(20)['Importance']
        )):
            f.write(f"{i+1}. {feature}: {importance:.6f}\n")
        
        f.write("\nConclusion:\n")
        f.write("The SHAP analysis shows the most important features for DON prediction.\n")
        f.write("Features with higher SHAP values have a greater impact on the model's predictions.\n")
        f.write("These insights can be used to understand which factors most influence DON concentration levels.\n")

# Main execution
if __name__ == "__main__":
    # Load model and scaler
    print("Loading model and scaler...")
    model, scaler = load_model()
    
    # Load data
    print("Loading data...")
    X, y = load_data(sample_size=200)
    
    # Run SHAP analysis
    print("Running SHAP analysis...")
    shap_values, shap_importance = run_shap_analysis(model, scaler, X)
    
    # Create summary report
    create_summary_report(shap_importance)
    
    print(f"\nSHAP analysis completed. All results saved to {RESULTS_DIR}")
    print("Files generated:")
    for file in os.listdir(RESULTS_DIR):
        print(f"- {file}")