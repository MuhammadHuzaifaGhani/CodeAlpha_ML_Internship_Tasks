# 🎙️ Emotion Recognition from Speech

## 📌 Objective  
The goal of this project is to recognize **human emotions** (e.g., Happy, Angry, Sad) from speech audio using **Deep Learning** and **Speech Signal Processing** techniques.  

---

## 🚀 Approach  
1. **Feature Extraction**  
   - Used **MFCCs (Mel-Frequency Cepstral Coefficients)** to extract key features from audio signals.  

2. **Modeling**  
   - Implemented **CNN, RNN, and LSTM** architectures to classify speech into different emotion categories.  

3. **Datasets**  
   - [RAVDESS](https://zenodo.org/record/1188976)  
   - [TESS](https://tspace.library.utoronto.ca/handle/1807/24487)  
   - [EMO-DB](http://emodb.bilderbar.info/start.html)  

---

## 📂 Project Structure  

Emotion-Recognition/
│── data/ # Speech datasets (RAVDESS, TESS, EMO-DB)
│── notebooks/ # Jupyter Notebooks for experiments
│── src/ # Source code (feature extraction, training, evaluation)
│── models/ # Saved trained models
│── results/ # Evaluation metrics & plots
│── README.md # Project documentation


---

## 🛠️ Tech Stack  
- **Programming Language:** Python  
- **Libraries & Tools:**  
  - `librosa` – Audio feature extraction  
  - `NumPy`, `Pandas` – Data handling  
  - `Matplotlib`, `Seaborn` – Visualization  
  - `TensorFlow` / `Keras` – Deep Learning models  
  - `Scikit-learn` – Preprocessing & evaluation  

---

## 📊 Results  
- Achieved accurate classification of multiple emotion classes.  
- LSTM and CNN models performed best on sequential and spectral features.  

---

## 💡 Learnings  
- Hands-on experience with **speech signal processing**.  
- Understanding how **deep learning models** interpret emotions from audio.  
- Importance of dataset balance and preprocessing in audio ML tasks.  

---

## 🙌 Acknowledgment  
This project was completed as part of my internship with **CodeAlpha**.  

---

## 📌 Future Work  
- Expand to multi-lingual datasets.  
- Deploy as a **real-time emotion recognition app**.  
- Integrate with **chatbots and virtual assistants** for empathetic responses.  

---

## 🔖 Author  
👤 **Muhammad Huzaifa Ghani**  
