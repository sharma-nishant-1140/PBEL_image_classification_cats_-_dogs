# 🐶 Dogs vs 🐱 Cats - Image Classifier (CNN + Flask)

A deep learning web app that classifies uploaded images as either a **dog** or a **cat** using a Convolutional Neural Network (CNN) and Flask.

---

## 📦 Dataset

- Source: [Kaggle - Dogs vs Cats](https://www.kaggle.com/datasets/salader/dogs-vs-cats)
- ~25,000 labeled images (dogs and cats)
- Images resized to **128x128** and normalized

---

## 🧠 Model Summary

- **CNN Architecture**:
  - Conv2D → ReLU → MaxPooling
  - Conv2D → ReLU → MaxPooling
  - Flatten → Dense → Dropout → Output (Sigmoid)
- **Optimizer**: Adam
- **Loss**: Binary Crossentropy
- **Accuracy**: ~90% on validation set

---

## 💻 Tech Stack

- **Frontend**: HTML + CSS
- **Backend**: Python (Flask)
- **Deep Learning**: TensorFlow / Keras

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python app.py

Then open http://127.0.0.1:5000 in your browser.
