# Word-Level Language Identification for Dravidian Languages

[![FIRE 2025](https://img.shields.io/badge/FIRE-2025-blue)](https://fire.irsi.res.in/)
[![Paper](https://img.shields.io/badge/Paper-Published-green)](link-to-paper)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the implementation for our paper **"Transformer Driven Word Level Classification of Dravidian Languages"** presented at the Forum for Information Retrieval Evaluation (FIRE) 2025.

## 🏆 Competition Results

CoLI-Dravidian @ FIRE 2025 Shared Task on Word-Level Language Identification

| Language  | Macro F1 | Rank | Dataset Size (Train/Val/Test) |
|-----------|----------|------|-------------------------------|
| Telugu    | **0.9515** | **🥇 1st** | 6,280 / 515 / 494 |
| Malayalam | **0.8271** | **🥇 1st** | 25,995 / 2,008 / 1,997 |
| Tamil     | **0.7434** | **🥇 1st** | 13,514 / 1,984 / 2,006 |
| Tulu      | **0.8224** | **🥈 2nd** | 29,524 / 3,006 / 3,283 |
| Kannada   | 0.8995 | 7th | 30,910 / 2,016 / 2,075 |

## 📹 Presentation
   
   [Watch our FIRE 2025 presentation on YouTube]
   https://www.youtube.com/watch?v=m7QtreTPBDM
starts at 2 hours and 8 minutes

## 📖 Abstract

Language detection is the process of automatically identifying the language used in a text, even when that text is not always coherent or grammatically correct. This challenge becomes much tougher when dealing with code-mixed or multilingual text, which is common in linguistically diverse regions like South India.

We propose a high-performance model using **Language-agnostic BERT Sentence Embedding (LaBSE)** for word-level identification of five Dravidian languages: Tamil, Telugu, Malayalam, Kannada, and Tulu.

## 🎯 Key Features

- **Language-Agnostic Approach**: Uses LaBSE pre-trained on 109 languages
- **Strong Performance**: First place in 3/5 languages, second place in 1/5
- **Reproducible**: Complete code with exact hyperparameters from paper
- **Well-Documented**: Clean notebooks with explanations

## 🏗️ Repository Structure

```
fire2025-dravidian-langdetect/
├── notebooks/                  # Jupyter notebooks for each language
│   ├── kannada_language_detection.ipynb
│   ├── tamil_language_detection.ipynb
│   ├── malayalam_language_detection.ipynb
│   ├── telugu_language_detection.ipynb
│   └── tulu_language_detection.ipynb
├── data/                       # Dataset files (add your own)
│   ├── kannada/
│   ├── tamil/
│   ├── malayalam/
│   ├── telugu/
│   └── tulu/
├── models/                     # Saved models (created during training)
├── results/                    # Predictions and evaluation results
├── docs/                       # Additional documentation
├── requirements.txt            # Python dependencies
├── LICENSE                     # License file
└── README.md                   # This file
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/fire2025-dravidian-langdetect.git
cd fire2025-dravidian-langdetect

# Install dependencies
pip install -r requirements.txt
```

**Important:** This project requires specific library versions due to API compatibility:
```bash
pip install transformers==4.10.0
pip install simpletransformers==0.64.3
```

### Data Preparation

1. Download the CoLI-Dravidian @ FIRE 2025 datasets
2. Place them in the `data/` directory following this structure:
   ```
   data/
   ├── kannada/
   │   ├── kan_train.csv
   │   ├── kan_val.csv
   │   └── kan_test.csv
   ├── tamil/
   │   ├── tm_train.csv
   │   ├── tm_val.csv
   │   └── tm_test.csv
   ... (similar for other languages)
   ```

### Running the Notebooks

Open any notebook in Jupyter:

```bash
jupyter notebook notebooks/kannada_language_detection.ipynb
```

Or use Google Colab by uploading the notebooks.

## 🔬 Methodology

### Model Architecture

- **Base Model**: [LaBSE](https://huggingface.co/setu4993/LaBSE) (Language-agnostic BERT Sentence Embedding)
- **Type**: BERT-based transformer with 12 layers, 12 attention heads, 768 hidden units
- **Pre-training**: Multilingual (109 languages) with translation language modeling

### Hyperparameters

As described in the paper (Section 4.1):

| Parameter | Value |
|-----------|-------|
| Training Epochs | 10 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | Default (4e-5) |
| Number of Labels | 7-8 (language-dependent) |

### Training Process

1. **Data Loading**: Load train, validation, and test CSV files
2. **Label Encoding**: Encode language tags using scikit-learn's LabelEncoder
3. **Model Training**: Fine-tune LaBSE on language-specific training data
4. **Evaluation**: Assess performance on validation set
5. **Prediction**: Generate predictions for test set

## 📊 Results

### Detailed Performance Metrics

**Kannada:**
- Weighted Precision: 0.9686
- Weighted Recall: 0.9681
- Weighted F1: 0.9683
- **Macro F1: 0.8995**
- Accuracy: 0.9681

**Tamil:**
- Weighted Precision: 0.9249
- Weighted Recall: 0.9249
- Weighted F1: 0.9242
- **Macro F1: 0.7434**
- Accuracy: 0.9249

**Malayalam:**
- Weighted Precision: 0.8825
- Weighted Recall: 0.8843
- Weighted F1: 0.8818
- **Macro F1: 0.8271**
- Accuracy: 0.8843

**Telugu:**
- Weighted Precision: 0.9689
- Weighted Recall: 0.9676
- Weighted F1: 0.9681
- **Macro F1: 0.9515**
- Accuracy: 0.9676

**Tulu:**
- Weighted Precision: 0.9009
- Weighted Recall: 0.9028
- Weighted F1: 0.9011
- **Macro F1: 0.8224**
- Accuracy: 0.9028

### Error Analysis

Common misclassification patterns:
- **Named Entities**: Model struggles with proper nouns
- **Misspellings**: Non-standard spellings confuse the model
- **Cross-Language Similarity**: Tulu and Kannada share vocabulary
- **Code-mixing**: English words in non-standard scripts

See the paper (Section 6) for detailed error analysis.

## 📄 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{mahibha2025transformer,
  title={Transformer Driven Word Level Classification of Dravidian Languages},
  author={Mahibha, C. Jerin and Robert, Wordson and Shimi, Gersome and Thenmozhi, Durairaj},
  booktitle={Forum for Information Retrieval Evaluation},
  year={2025},
  organization={FIRE}
}
```

## 👥 Authors

- **C. Jerin Mahibha** - Meenakshi Sundararajan Engineering College, Chennai
- **Wordson Robert** - Indian Institute of Science Education and Research, Kolkata
- **Gersome Shimi** - Madras Christian College, Chennai
- **Durairaj Thenmozhi** - Sri Sivasubramaniya Nadar College of Engineering, Chennai

## 📧 Contact

For questions or collaborations, please contact:
- Wordson Robert: wordsonrobert@gmail.com
- C. Jerin Mahibha: jerinmahibha@msec.edu.in

## 🙏 Acknowledgments

- FIRE 2025 organizers for hosting the CoLI-Dravidian shared task
- Dataset contributors
- The Hugging Face team for LaBSE and transformers library

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [FIRE 2025 Website](https://fire.irsi.res.in/)
- [LaBSE Model Card](https://huggingface.co/setu4993/LaBSE)
- [Competition Task Description](link-to-task-description)

---

**Note:** This implementation uses the exact hyperparameters and methodology described in our FIRE 2025 paper. For questions about reproducing results, please refer to the notebooks and ensure you're using the correct library versions specified in `requirements.txt`.
