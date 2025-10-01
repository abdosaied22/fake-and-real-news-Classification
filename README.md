# Fake and Real News Classification 📰🔍

A machine learning and deep learning project to classify **fake** and **real** news articles.

---

## 🎯 Project Goal

In today’s world where news spreads rapidly, the objective of this project is to build a model that can distinguish between **real** and **fake** news based on text.  
The project experiments with various techniques: from preprocessing, to classical ML models, to deep learning (RNN/GRU/LSTM), and finally fine-tuning Transformer models.

---
![Fake News Detection](https://storage.googleapis.com/kaggle-datasets-images/4831777/8165591/2f455915fd250d05f6709b21b65c6bcc/dataset-cover.jpg?t=2024-04-20-04-53-50)

## 📁 Repository Structure

| File/Folder | Description |
|-------------|-------------|
| `fake-and-real-news-dataset/` | Dataset of real and fake news (e.g. from Kaggle) |
| `imgs/` | Images or visualizations used in report/documentation |
| `model/` | Final trained model or large model-related files |
| `models/` | Different trained models |
| `pkl_files/` | Pickle files for data/preprocessing steps |
| `01.Preprocessing.ipynb` | Text preprocessing steps (cleaning, tokenization, stopword removal, etc.) |
| `02.Classical_ML_models.ipynb` | Training classical ML models (Logistic Regression, SVM, Random Forest, etc.) |
| `03.RNN_GRU_LSTM_models.ipynb` | Neural network models (RNN, GRU, LSTM) |
| `04.Transformer_Fine-tuning.ipynb` | Fine-tuning Transformer models (e.g. BERT, RoBERTa) |
| `home.py` | Simple user interface (e.g. Streamlit/Flask app) to test the model |
| `Report.pdf` | Final project report with results and analysis |
| `README.md` | This documentation file |

---

## 📦 Requirements

To run this project, install the required packages. It is recommended to use a virtual environment:

```bash
python3 -m venv env
source env/bin/activate   # On Linux/macOS
# or on Windows: env\Scripts\activate
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Make sure the dataset is available in `fake-and-real-news-dataset/`.  
2. Run the notebooks in sequence:

   1. `01.Preprocessing.ipynb`  
   2. `02.Classical_ML_models.ipynb`  
   3. `03.RNN_GRU_LSTM_models.ipynb`  
   4. `04.Transformer_Fine-tuning.ipynb`  

3. To launch the interface:

```bash
python home.py
```

Then open the browser at the provided local address (e.g., `http://localhost:XXXX`).

---

## 📊 Results

- Performance comparison between classical ML, neural networks, and Transformers.  
- Fine-tuning Transformer models tends to provide the best accuracy on larger datasets.  
- Full details and visualizations can be found in `Report.pdf`.  

---

## 🔗 References

- Dataset: [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)  
- Word embeddings: [GloVe Embeddings (Stanford NLP)](https://nlp.stanford.edu/projects/glove/)  

---

## 👤 Author

**abdosaied22**

---
