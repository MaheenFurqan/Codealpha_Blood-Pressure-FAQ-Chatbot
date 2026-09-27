import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# FAQ dataset
faqs = [
    {
        "question": "What is a digital blood pressure monitor?",
        "answer": "A digital blood pressure monitor is an electronic device used to measure blood pressure and pulse rate."
    },
    {
        "question": "What does a blood pressure monitor measure?",
        "answer": "It measures blood pressure, including systolic and diastolic pressure, and usually displays the pulse rate."
    },
    {
        "question": "How does a digital blood pressure monitor work?",
        "answer": "The monitor uses an inflatable cuff to temporarily apply pressure to the arm and sensors to detect blood flow and calculate the reading."
    },
    {
        "question": "How do I use a digital blood pressure monitor?",
        "answer": "Sit comfortably, place the cuff correctly on your arm, keep your arm supported, and start the measurement while remaining still and quiet."
    },
    {
        "question": "How should I sit when measuring blood pressure?",
        "answer": "Sit with your back supported, feet flat on the floor, and legs uncrossed. Keep your arm supported at about heart level."
    },
    {
        "question": "Where should I place the blood pressure cuff?",
        "answer": "Place the cuff around your bare upper arm according to the position indicated in the monitor's instructions."
    },
    {
        "question": "Can I measure blood pressure while standing?",
        "answer": "It is generally recommended to sit comfortably and remain still while taking a standard blood pressure measurement."
    },
    {
        "question": "Can I measure blood pressure over clothing?",
        "answer": "For more reliable measurements, place the cuff directly on the bare upper arm rather than over clothing."
    },
    {
        "question": "What does SYS mean on the monitor?",
        "answer": "SYS stands for systolic blood pressure. It is the upper number shown in a blood pressure reading."
    },
    {
        "question": "What does DIA mean on the monitor?",
        "answer": "DIA stands for diastolic blood pressure. It is the lower number shown in a blood pressure reading."
    },
    {
        "question": "What does Pulse mean on the monitor?",
        "answer": "Pulse indicates the number of heartbeats detected per minute during the measurement."
    },
    {
        "question": "Why does the monitor show different readings each time?",
        "answer": "Blood pressure can naturally vary. Movement, talking, body position, stress, and taking measurements too close together can also affect readings."
    },
    {
        "question": "How can I get an accurate blood pressure reading?",
        "answer": "Follow the monitor's instructions, sit quietly before measuring, use the correct cuff position, keep your arm supported, and remain still and quiet during the measurement."
    },
    {
        "question": "Should I rest before measuring blood pressure?",
        "answer": "Yes. Sitting quietly for several minutes before taking a measurement can help provide a more consistent reading."
    },
    {
        "question": "Can talking affect the blood pressure reading?",
        "answer": "Yes. Talking or moving during measurement can affect the reading, so it is better to remain quiet and still."
    },
    {
        "question": "Why is my blood pressure monitor showing an error?",
        "answer": "An error can occur because of incorrect cuff placement, movement, talking, or another device-related issue. Check the cuff and follow the manufacturer's instructions."
    },
    {
        "question": "Why is the cuff not inflating?",
        "answer": "Check that the cuff is connected properly, the tubing is secure, and the batteries have sufficient power. If the problem continues, check the device instructions."
    },
    {
        "question": "Why is the monitor not turning on?",
        "answer": "Check whether the batteries are installed correctly and have sufficient power. If the monitor still does not turn on, consult its instructions or manufacturer."
    },
    {
        "question": "How do I clean my blood pressure monitor?",
        "answer": "Clean the monitor and cuff according to the manufacturer's instructions. Avoid getting liquid inside the electronic monitor."
    },
    {
        "question": "How should I store my blood pressure monitor?",
        "answer": "Store the monitor and cuff in a clean, dry place and protect them from excessive heat, moisture, and damage."
    },
    {
        "question": "How often should I check my blood pressure monitor's accuracy?",
        "answer": "Check the device's accuracy according to the manufacturer's recommended schedule and instructions."
    },
    {
        "question": "What should I do if the cuff feels too tight?",
        "answer": "Stop the measurement if the cuff feels excessively tight and check that it is positioned and fitted correctly according to the manufacturer's instructions."
    },
    {
        "question": "Why is the blood pressure reading unusually high or low?",
        "answer": "Readings can be affected by factors such as movement, talking, body position, stress, or incorrect cuff placement. Repeat the measurement using the proper technique."
    },
    {
        "question": "Can I take multiple blood pressure measurements in a row?",
        "answer": "Follow the monitor's instructions when taking repeated measurements and allow an appropriate rest period between measurements."
    },
    {
        "question": "How do I replace the batteries in the monitor?",
        "answer": "Open the battery compartment, remove the old batteries, and insert new batteries according to the polarity markings and the manufacturer's instructions."
    }
]


# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


# Preprocess all FAQ questions
processed_questions = []

for faq in faqs:
    processed_question = preprocess_text(faq["question"])
    processed_questions.append(processed_question)


# Convert FAQ questions into TF-IDF vectors
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(processed_questions)


# Find the best response
def get_response(user_question):
    processed_user_question = preprocess_text(user_question)

    user_vector = vectorizer.transform([processed_user_question])

    similarities = cosine_similarity(user_vector, faq_vectors)

    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    if best_score >= 0.3:
        return faqs[best_match_index]["answer"]
    else:
        return "Sorry, I don't understand your question."


def get_response(user_question):
    processed_user_question = preprocess_text(user_question)

    user_vector = vectorizer.transform([processed_user_question])

    similarities = cosine_similarity(user_vector, faq_vectors)

    best_match_index = similarities.argmax()
    best_score = similarities[0][best_match_index]

    if best_score >= 0.3:
        return faqs[best_match_index]["answer"]
    else:
        return "Sorry, I don't understand your question."
