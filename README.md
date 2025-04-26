# 🚀 Fake News Detection Using Word2Vec Embeddings and Random Forest

This project builds a **Fake News Detector** using **Word2Vec embeddings** for text representation combined with a **Random Forest** classifier. It also explores text analysis techniques like **sentiment analysis**, **word clouds**, and **model comparison**.

---

## 📚 Project Overview

- ✅ Load and clean a dataset of real and fake news articles
- ✂️ Preprocess text: lowercasing, stopword removal, lemmatization
- 🧠 Train a **Word2Vec** model to embed words into vectors
- 🔠 Represent entire articles by averaging word vectors
- 🌳 Train a **Random Forest** model for fake news classification
- 📈 Evaluate model performance with classification metrics
- 🔮 Predict labels for unseen validation data
- 📊 Visualize most common words and generate word clouds
- 😎 Analyze sentiment distribution across real and fake articles
- ⚡ Compare different models (Random Forest, Logistic Regression, SVM)

---

## 🛠️ Tech Stack

| Component               | Tool/Library                     |
|--------------------------|----------------------------------|
| Text Preprocessing       | `nltk`, `re`, `WordNetLemmatizer` |
| Word Embeddings          | `gensim` Word2Vec                |
| Classification Models    | `Random Forest`, `Logistic Regression`, `SVM` |
| Visualization            | `matplotlib`, `seaborn`, `wordcloud` |
| Sentiment Analysis       | `nltk` Vader                     |
| Environment              | Jupyter / Colab                  |

---

## 🧪 How It Works

1. 📂 Load the news dataset (`data.csv`) and validation set (`validation_data.csv`)
2. 🧹 Clean and preprocess articles for cleaner model input
3. 🧠 Train a **Word2Vec** model on the training data
4. 🔠 Represent articles by averaging their word embeddings
5. 🌳 Train a **Random Forest** classifier using document vectors
6. 📈 Evaluate using accuracy, precision, recall, F1-score
7. 🔮 Predict fake/real labels for new validation articles
8. 📊 Analyze common words and visualize data using word clouds
9. 🏆 Compare multiple models: Random Forest, Logistic Regression, and SVM

---

## 💻 Notebook Contents

- `Fake_News_Detection_Word2Vec.ipynb`
  - [x] Data loading and cleaning
  - [x] Word2Vec training and embedding generation
  - [x] Random Forest training and evaluation
  - [x] Validation set prediction
  - [x] Sentiment analysis
  - [x] Word frequency and word cloud visualization
  - [x] Model benchmarking (Random Forest, Logistic Regression, SVM)

---

## 🧠 Sample Results

| Model                  | Accuracy  |
|-------------------------|-----------|
| Random Forest           | ~95%      |
| Logistic Regression     | ~94%      |
| SVM                     | ~93%      |

- 🎯 **Word2Vec + Random Forest** achieved the best performance.
- 📰 **Real News** tends to have slightly more neutral sentiment compared to **Fake News**.

---

## ⚠️ Known Issues / Limitations

- ❌ Word2Vec averaging loses word order and context
- 🔁 Sentiment analysis using Vader is basic; could be enhanced with a fine-tuned LLM
- 📏 Model may struggle with very short or ambiguous headlines

---

## 📈 Improvements & Next Steps

- 🧠 Try fine-tuned transformer models (e.g., DistilBERT, RoBERTa) for better embeddings
- 🏷️ Include metadata like news source, publication date
- 🔍 Perform Named Entity Recognition (NER) for claim verification
- 🚀 Build a full interactive web app using **Streamlit** (bonus)

---

## 🚀 Try It Yourself

Install the required libraries first:
```bash
pip install nltk scikit-learn gensim matplotlib seaborn wordcloud
```

Download NLTK resources:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

Then simply run the notebook step by step!

---

✅ **READY!**  
This README is fully aligned to your favorite format. 🔥  

---

**Quick extras if you want:**
- I can give you a `.gitignore` template (to ignore models, vectorizers, etc.)
- Or suggest a clean project folder layout (`/models`, `/data`, `/notebooks`, `/scripts`, etc.)

Just say the word! 🚀  
And when you send the next `.ipynb`, I'm ready! 🎯

## Datasets can be found here
https://drive.google.com/drive/folders/1040eGEOM_LXIM14P2-VFZEk2pjDBfDh5?usp=sharing
