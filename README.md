# Transformer Sentiment Analysis

This project demonstrates how to fine-tune a **Transformer model** (distilbert-base-uncased) for **binary sentiment classification** of user reviews (positive or negative).

## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/PolyakovMA/sentiment-transformer.git
cd sentiment-transformer
pip install -r requirements.txt
```

## 📦 Project Structure

```
sentiment-transformer/
├── data/ # Dataset files
├── notebooks/ # Jupyter notebooks for exploration and experiments
├── src/ # Source code
│ ├── data_preprocessing.py
│ ├── evaluate.py
│ ├── model.py
│ ├── train.py
│ └── utils.py
├── .gitignore # Git ignore file
├── requirements.txt # Python dependencies
└── TSA.py # Main script to run the pipeline
```

## Dependencies
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- scikit-learn
- pandas
- numpy

