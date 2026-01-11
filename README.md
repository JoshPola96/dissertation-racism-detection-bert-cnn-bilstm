# **MSc Dissertation: Bias-Minimized Racism Detection**

**Curriculum Work / Academic Project**

---

## 🔹 Overview

This project presents a **multi-modal neural network** designed to detect racism in social media text while aiming to **minimize bias**. The model integrates:

* **Text Classification:** BERT embeddings → CNN → biLSTM for sequential and contextual understanding.
* **Numerical & Categorical Features:** Dense Neural Networks for structured data.

> ⚠️ Disclaimer: This work uses offensive content strictly for academic research. It does not promote or endorse any form of aggression or hate.

---

## 🔹 Key Features

* Multi-modal deep learning for **text + numerical features**.
* Bias-minimization strategies incorporated in model design.
* Preprocessing handles raw Twitter data, including language detection, emoji, hashtag, and URL cleaning.
* End-to-end pipeline from **data cleaning → EDA → model training → evaluation**.
* Sample predictions, metrics (confusion matrix, ROC-AUC), and visualization provided.

---

## 🔹 Technical Stack

* **Programming & ML:** Python, TensorFlow, Keras, Scikit-learn, Numpy, Pandas, Matplotlib
* **NLP Tools:** BERT (TF Hub), Ekphrasis, VaderSentiment, Tweet-preprocessor
* **Data Handling & Visualization:** Pandas, WordCloud, Matplotlib, Seaborn
* **Modeling:** CNN, biLSTM, Dense Layers, Combined Multi-Modal Model
* **Evaluation:** Confusion matrix, classification report, ROC-AUC

---

## 🔹 Dataset & Preprocessing

* Sources: Kaggle (tweets datasets), Waseem & Hovy annotations, Hydrator Twitter API exports
* **Cleaning steps:**

  * Remove URLs, mentions, hashtags, reserved words, numbers
  * Detect and select only English tweets
  * Word segmentation and stopword removal
  * Standardize user locations and perform sentiment labeling

---

## 🔹 Model Architecture

1. **Text Model:** BERT → CNN layers → MaxPooling → Dropout → biLSTM → Sigmoid Output
2. **Numerical Model:** Dense Neural Network → Dropout → Sigmoid Output
3. **Combined Model:** Concatenate text & numerical outputs → Dense → Dropout → Sigmoid Output

**Training & Evaluation:**

* Stratified train/test split (20% test)
* Class weights for label imbalance
* EarlyStopping callbacks for convergence
* Performance metrics: binary accuracy, ROC-AUC, classification report

---

## 🔹 EDA & Visualization

* WordClouds for **racist vs non-racist tweets**
* Top unigrams, bigrams, trigrams per label
* Sentiment distribution across labels
* Location-wise distribution analysis

---

## 🔹 Contribution & Contact

Developed and maintained by:
**Joshua Peter Polaprayil**
📧 [josh19peter96@gmail.com](mailto:josh19peter96@gmail.com)

---

## 🔹 References & Acknowledgements

* Kaggle datasets: [Racism Tweets](https://kaggle.com/raghadabdullah/racism-tweets), [Classified Tweets](https://kaggle.com/munkialbright/classified-tweets)
* Waseem & Hovy hate speech annotations
* TensorFlow Hub BERT models

---

## 🔹 License / Disclaimer

* Academic & research purpose only
* Offensive content used solely to support bias detection research
* Not for commercial use or dissemination of harmful content
