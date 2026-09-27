# 🩺 Digital Blood Pressure Monitor FAQ Chatbot

A simple NLP-based FAQ chatbot that answers common questions about digital blood pressure monitors.

The chatbot uses **NLTK for text preprocessing**, **TF-IDF for text representation**, and **cosine similarity** to find the FAQ question most similar to the user's question.

It also includes a simple **HTML/CSS web interface** connected to the Python chatbot using **Flask**.

---

## 📌 Features

- 25 manually created FAQs about digital blood pressure monitors
- Text preprocessing using NLTK
- Lowercase conversion
- Punctuation removal
- Tokenization
- Stop-word removal
- TF-IDF vectorization
- Cosine similarity matching
- Best FAQ answer selection
- Threshold for handling unrelated questions
- Flask-based backend
- HTML/CSS chatbot interface
- Enter key support for sending questions
- Blue medical-themed user interface

---

## 🛠️ Technologies Used

- **Python**
- **NLTK**
- **Scikit-learn**
- **Flask**
- **HTML**
- **CSS**
- **JavaScript**

---

## 📂 Project Structure

```text
FAQ_chatbot/
│
├── app.py
├── chatbot.py
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
