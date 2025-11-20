# LLM Sentiment Analysis

A comparative study of sentiment classification using **zero-shot** and **few-shot prompting** techniques with multiple large language models (DistilBERT and BART) on restaurant review data from Yelp.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Project Structure](#project-structure)
- [Key Findings](#key-findings)
- [Requirements](#requirements)
- [Acknowledgments](#acknowledgments)

## 🔍 Overview

This project explores advanced prompting techniques for sentiment analysis without traditional fine-tuning. It compares the performance of different LLMs using:

- **Zero-Shot Learning**: Classification without labeled examples
- **Few-Shot Learning**: Classification with a small number of labeled examples
- **Multiple LLMs**: DistilBERT and BART (Facebook's large-scale MNLI model)

## ✨ Features

- 🎯 Comparative analysis of zero-shot vs few-shot prompting
- 🤖 Multiple pre-trained transformer models
- 📊 Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1 Score)
- 📈 Performance visualization and analysis
- 🔄 Balanced dataset sampling (50 positive, 50 negative reviews)

## 📊 Dataset

- **Source**: Yelp restaurant reviews from Arizona
- **Size**: 100 reviews (50 positive, 50 negative)
- **Features**: Review text, star ratings, sentiment labels
- **Domain**: Restaurant and food service reviews

## 🤖 Models Used

### 1. DistilBERT
- **Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Description**: Lighter, faster version of BERT fine-tuned on sentiment analysis
- **Use Case**: Primary sentiment classification model

### 2. BART-MNLI
- **Model**: `facebook/bart-large-mnli`
- **Description**: BART model fine-tuned on Multi-Genre Natural Language Inference
- **Use Case**: Alternative classification approach

## 🚀 Installation

### Prerequisites
```bash
Python 3.7+
pip
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/RaviTeja-Kondeti/LLM-Sentiment-Analysis.git
cd LLM-Sentiment-Analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the Jupyter notebook:
```bash
jupyter notebook sentiment_analysis.ipynb
```

## 💻 Usage

The project is implemented in a Jupyter notebook with the following workflow:

1. **Data Loading**: Import and explore the Yelp restaurant review dataset
2. **Data Preprocessing**: Balance the dataset and prepare for analysis
3. **Zero-Shot Analysis**: Perform sentiment classification without examples
4. **Few-Shot Analysis**: Add labeled examples to improve classification
5. **Multi-Model Comparison**: Compare DistilBERT and BART performance
6. **Evaluation**: Analyze results using multiple metrics

### Code Example

```python
from transformers import pipeline

# Load sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", 
                              model="distilbert-base-uncased-finetuned-sst-2-english")

# Analyze a review
result = sentiment_pipeline("The food was amazing and service was excellent!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

## 📈 Results

### Zero-Shot Learning Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 86% |
| **Precision** | 82% |
| **Recall** | 92% |
| **F1 Score** | 87% |

### Few-Shot Learning Performance

| Metric | Score |
|--------|-------|
| **Accuracy** | 86% |
| **Precision** | 82% |
| **Recall** | 92% |
| **F1 Score** | 87% |

### Model Comparison

- **DistilBERT**: Provides balanced predictions with good accuracy
- **BART-MNLI**: Showed conservative behavior, classifying most examples as negative
- **Best Performer**: DistilBERT demonstrated superior sentiment classification capabilities

## 📁 Project Structure

```
LLM-Sentiment-Analysis/
│
├── sentiment_analysis.ipynb    # Main Jupyter notebook
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── .gitignore                 # Git ignore file
```

## 🔑 Key Findings

1. **Zero-Shot vs Few-Shot**: Both approaches achieved identical performance (86% accuracy), suggesting DistilBERT's pre-training is highly effective for sentiment analysis

2. **Model Selection Matters**: BART-MNLI struggled with positive sentiment detection, highlighting the importance of choosing the right model for the task

3. **Misclassification Patterns**: Errors occurred primarily in:
   - Reviews with mixed sentiments
   - Sarcastic or ironic language
   - Very short reviews lacking context

4. **High Recall**: 92% recall indicates the models excel at identifying positive sentiment

## 📦 Requirements

- pandas
- transformers
- scikit-learn
- torch
- jupyter

*See `requirements.txt` for specific versions*

## 🙏 Acknowledgments

- **Dataset**: Yelp Open Dataset
- **Models**: Hugging Face Transformers library
- **Framework**: PyTorch
- **Tools**: ChatGPT for development assistance

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Ravi Teja Kondeti**
- GitHub: [@RaviTeja-Kondeti](https://github.com/RaviTeja-Kondeti)

---

⭐ If you find this project useful, please consider giving it a star!
