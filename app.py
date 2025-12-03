from flask import Flask, request, jsonify, render_template
import joblib
import pickle
import string
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import io
import base64

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

# Load the trained sentiment analysis model and vectorizer
classifier = joblib.load('model.pkl')
tfidf_vectorizer = joblib.load('vectorizer.pkl')

# Load the updated emotion detection model and vectorizer
emotion_model = joblib.load('emotion_model.pkl')
emotion_vectorizer = joblib.load('emotion_vectorizer.pkl')

print("Emotion detection model and vectorizer loaded.")  # Debug print

# Emotion mapping
EMOTION_MAP = {
               
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    input_text = request.json.get("text", "")
    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    print("Raw input text:", input_text)  # Debug print

    # Sentiment analysis
    input_features = tfidf_vectorizer.transform([input_text])
    y_proba = classifier.predict_proba(input_features)
    positive_score = float(y_proba[0][1])  # Convert np.float64 to float
    sentiment = "Neutral"
    if positive_score < 0.4:
        sentiment = "Negative"
    elif positive_score > 0.6:
        sentiment = "Positive"

    # Emotion detection
    emotion_features = emotion_vectorizer.transform([input_text])
    emotion_prediction = emotion_model.predict(emotion_features)
    emotion_probabilities = emotion_model.predict_proba(emotion_features).flatten()

    # Create barplot for emotions
    fig, ax = plt.subplots()
    emotions = [EMOTION_MAP[i] for i in range(len(emotion_probabilities))]
    ax.bar(emotions, emotion_probabilities, color='skyblue')
    ax.set_title("Emotion Probabilities")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Emotions")
    plt.tight_layout()

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()

    # Map numeric label to emotion name
    dominant_emotion_index = int(emotion_prediction[0])  # Ensure JSON serializable
    dominant_emotion = EMOTION_MAP.get(dominant_emotion_index, "Unknown")  # Map to emotion name

    # Debug: print out what we are sending to the frontend
    print("Returning response:", {
        "text": input_text,
        "sentiment": {"score": positive_score, "sentiment": sentiment},
        "dominant_emotion": {"emotion": dominant_emotion},
        "emotion_plot": "data:image/png;base64," + base64_image
    })

    return jsonify({
        "text": input_text,
        "sentiment": {"score": positive_score, "sentiment": sentiment},
        "dominant_emotion": {"emotion": dominant_emotion},
        "emotion_plot": "data:image/png;base64," + base64_image
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
