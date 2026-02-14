"""
Generate all language detection notebooks with correct hyperparameters.

This script creates 5 notebooks (Kannada, Tamil, Malayalam, Telugu, Tulu)
with the exact hyperparameters specified in the FIRE 2025 paper.
"""

import json
import os

# Language configurations
LANGUAGES = {
    'kannada': {
        'name': 'Kannada',
        'code': 'kan',
        'num_labels': 7,
        'train_size': 30910,
        'val_size': 2016,
        'test_size': 2075,
        'macro_f1': 0.8995,
        'macro_precision': 0.8863,
        'macro_recall': 0.9147,
        'accuracy': 0.9681,
        'rank': '7th'
    },
    'tamil': {
        'name': 'Tamil',
        'code': 'tm',
        'num_labels': 8,
        'train_size': 13514,
        'val_size': 1984,
        'test_size': 2006,
        'macro_f1': 0.7434,
        'macro_precision': 0.7696,
        'macro_recall': 0.7353,
        'accuracy': 0.9249,
        'rank': '🥇 1st'
    },
    'malayalam': {
        'name': 'Malayalam',
        'code': 'mal',
        'num_labels': 8,
        'train_size': 25995,
        'val_size': 2008,
        'test_size': 1997,
        'macro_f1': 0.8271,
        'macro_precision': 0.8624,
        'macro_recall': 0.8028,
        'accuracy': 0.8843,
        'rank': '🥇 1st'
    },
    'telugu': {
        'name': 'Telugu',
        'code': 'tl',
        'num_labels': 8,
        'train_size': 6280,
        'val_size': 515,
        'test_size': 494,
        'macro_f1': 0.9515,
        'macro_precision': 0.9446,
        'macro_recall': 0.9590,
        'accuracy': 0.9676,
        'rank': '🥇 1st'
    },
    'tulu': {
        'name': 'Tulu',
        'code': 'tulu',
        'num_labels': 8,
        'train_size': 29524,
        'val_size': 3006,
        'test_size': 3283,
        'macro_f1': 0.8224,
        'macro_precision': 0.8425,
        'macro_recall': 0.8063,
        'accuracy': 0.9028,
        'rank': '🥈 2nd'
    }
}

def create_notebook(lang_key, config):
    """Create a notebook for a specific language."""
    
    lang_name = config['name']
    lang_code = config['code']
    num_labels = config['num_labels']
    
    notebook = {
        "cells": [
            # Title cell
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {lang_name} Language Detection\n",
                    "## FIRE 2025 - CoLI-Dravidian Shared Task\n",
                    "\n",
                    f"This notebook implements word-level language identification for {lang_name} using LaBSE.\n",
                    "\n",
                    "**Paper:** Transformer Driven Word Level Classification of Dravidian Languages (FIRE 2025)\n",
                    "\n",
                    "**Competition Result:**\n",
                    f"- Macro F1: **{config['macro_f1']}**\n",
                    f"- Rank: **{config['rank']}**"
                ]
            },
            # Installation cell
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install required libraries\n",
                    "!pip install transformers==4.10.0\n",
                    "!pip install simpletransformers==0.64.3"
                ]
            },
            # Imports cell
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import torch\n",
                    "from sklearn.preprocessing import LabelEncoder\n",
                    "from sklearn.metrics import confusion_matrix, classification_report\n",
                    "from simpletransformers.classification import ClassificationModel\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns"
                ]
            },
            # Data loading section
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 1. Data Loading"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"# Load training data\n",
                    f"df_train = pd.read_csv('../data/{lang_key}/{lang_code}_train.csv')\n",
                    f"print(f'Training samples: {{len(df_train)}}')\n",
                    "print(f'\\nLabel distribution:\\n{df_train[\"Tag\"].value_counts()}')\n",
                    "df_train.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"# Load validation data\n",
                    f"df_eval = pd.read_csv('../data/{lang_key}/{lang_code}_val.csv')\n",
                    f"print(f'Validation samples: {{len(df_eval)}}')\n",
                    "df_eval.head()"
                ]
            },
            # Label encoding section
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Label Encoding\n",
                    "\n",
                    "Combine train and validation tags for consistent encoding."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Fit label encoder on combined train + eval tags\n",
                    "le = LabelEncoder()\n",
                    "all_tags = pd.concat([df_train['Tag'], df_eval['Tag']]).unique()\n",
                    "le.fit(all_tags)\n",
                    "\n",
                    "# Transform training labels\n",
                    "df_train['Tag'] = le.transform(df_train['Tag'])\n",
                    "\n",
                    "# Display label mapping\n",
                    "label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))\n",
                    "print(f'Label mapping: {label_mapping}')\n",
                    f"print(f'Number of labels: {{len(le.classes_)}} (expected: {num_labels})')"
                ]
            },
            # Model configuration section
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Model Configuration\n",
                    "\n",
                    "**Model:** setu4993/LaBSE (Language-agnostic BERT Sentence Embedding)\n",
                    "\n",
                    "**Hyperparameters (from paper Section 4.1):**\n",
                    "- Training epochs: **10**\n",
                    "- Batch size: **32**\n",
                    "- Optimizer: Adam\n",
                    f"- Number of labels: **{num_labels}** (for {lang_name})"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Clear GPU cache\n",
                    "torch.cuda.empty_cache()\n",
                    "\n",
                    "# Initialize model with paper's hyperparameters\n",
                    "model = ClassificationModel(\n",
                    "    'bert',\n",
                    "    'setu4993/LaBSE',\n",
                    f"    num_labels={num_labels},\n",
                    "    use_cuda=torch.cuda.is_available(),\n",
                    "    args={\n",
                    "        'reprocess_input_data': True,\n",
                    "        'use_cached_eval_features': False,\n",
                    "        'overwrite_output_dir': True,\n",
                    "        'num_train_epochs': 10,\n",
                    "        'train_batch_size': 32,\n",
                    "        'eval_batch_size': 32,\n",
                    "        'save_steps': 500,\n",
                    "        'logging_steps': 100,\n",
                    f"        'output_dir': '../models/{lang_key}/',\n",
                    f"        'best_model_dir': '../models/{lang_key}/best_model/'\n",
                    "    }\n",
                    ")\n",
                    "\n",
                    "print('Model initialized successfully!')"
                ]
            },
            # Training section
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 4. Model Training"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Train the model\n",
                    "model.train_model(df_train)\n",
                    "print('Training completed!')"
                ]
            },
            # Evaluation section
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 5. Model Evaluation"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Transform validation labels\n",
                    "df_eval['Tag'] = le.transform(df_eval['Tag'])\n",
                    "\n",
                    "# Evaluate model\n",
                    "result, model_outputs, wrong_predictions = model.eval_model(df_eval)\n",
                    "print(f'\\nEvaluation Results: {result}')"
                ]
            },
            # Performance analysis section
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 6. Performance Analysis"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Get predictions\n",
                    "y_true = df_eval['Tag'].values\n",
                    "y_pred = np.argmax(model_outputs, axis=1)\n",
                    "\n",
                    "# Confusion Matrix\n",
                    "cm = confusion_matrix(y_true, y_pred)\n",
                    "print('Confusion Matrix:')\n",
                    "print(cm)\n",
                    "\n",
                    "# Plot confusion matrix\n",
                    "plt.figure(figsize=(10, 8))\n",
                    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')\n",
                    "plt.xlabel('Predicted')\n",
                    "plt.ylabel('Actual')\n",
                    f"plt.title('Confusion Matrix - {lang_name}')\n",
                    f"plt.savefig('../results/{lang_key}_confusion_matrix.png', dpi=300, bbox_inches='tight')\n",
                    "plt.show()\n",
                    "\n",
                    "# Classification Report\n",
                    "print('\\nClassification Report:')\n",
                    "print(classification_report(y_true, y_pred, target_names=le.classes_))"
                ]
            },
            # Test predictions section
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["## 7. Test Set Predictions"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"# Load test data\n",
                    f"df_test = pd.read_csv('../data/{lang_key}/{lang_code}_test.csv')\n",
                    f"print(f'Test samples: {{len(df_test)}}')\n",
                    "df_test.head()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Make predictions\n",
                    "predictions, raw_outputs = model.predict(df_test['Word'].astype(str).tolist())\n",
                    "\n",
                    "# Add predictions to dataframe\n",
                    "df_test['Tag'] = predictions\n",
                    "\n",
                    "# Inverse transform to get original labels\n",
                    "df_test['Tag'] = le.inverse_transform(df_test['Tag'])\n",
                    "\n",
                    "# Display sample predictions\n",
                    "print('Sample predictions:')\n",
                    "print(df_test.head(20))"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    f"# Save predictions\n",
                    f"df_test.to_csv('../results/{lang_key}_predictions.csv', index=False)\n",
                    f"print('Predictions saved to ../results/{lang_key}_predictions.csv')"
                ]
            },
            # Results summary
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Results Summary\n",
                    "\n",
                    "**Expected Results (from paper):**\n",
                    f"- Macro F1: **{config['macro_f1']}**\n",
                    f"- Macro Precision: {config['macro_precision']}\n",
                    f"- Macro Recall: {config['macro_recall']}\n",
                    f"- Accuracy: {config['accuracy']}\n",
                    f"- **Competition Rank: {config['rank']}**"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


def main():
    """Generate all notebooks."""
    output_dir = "notebooks"
    os.makedirs(output_dir, exist_ok=True)
    
    for lang_key, config in LANGUAGES.items():
        print(f"Generating {config['name']} notebook...")
        notebook = create_notebook(lang_key, config)
        
        filename = f"{lang_key}_language_detection.ipynb"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(notebook, f, indent=1)
        
        print(f"  ✓ Saved to {filepath}")
    
    print(f"\n✅ Successfully generated {len(LANGUAGES)} notebooks!")
    print("\nNext steps:")
    print("1. Review the generated notebooks")
    print("2. Add your data files to data/ directory")
    print("3. Run the notebooks and verify results")


if __name__ == "__main__":
    main()
