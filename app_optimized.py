from flask import Flask, request, jsonify, render_template
import joblib
import os
import nltk
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

# Download NLTK data immediately
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# Load models at startup
try:
    classifier = joblib.load('model.pkl')
    tfidf_vectorizer = joblib.load('vectorizer.pkl')
    emotion_model = joblib.load('emotion_model.pkl')
    emotion_vectorizer = joblib.load('emotion_vectorizer.pkl')
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    classifier = tfidf_vectorizer = emotion_model = emotion_vectorizer = None

EMOTION_MAP = {
    0: "anger", 1: "fear", 2: "joy", 
    3: "love", 4: "sadness", 5: "surprise"
}

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not all([classifier, tfidf_vectorizer, emotion_model, emotion_vectorizer]):
        return jsonify({"error": "Models not loaded"}), 500
        
    input_text = request.json.get("text", "")
    if not input_text:
        return jsonify({"error": "No text provided"}), 400

    # Sentiment analysis
    input_features = tfidf_vectorizer.transform([input_text])
    y_proba = classifier.predict_proba(input_features)
    positive_score = float(y_proba[0][1])
    
    sentiment = "Neutral"
    if positive_score < 0.4:
        sentiment = "Negative"
    elif positive_score > 0.6:
        sentiment = "Positive"

    # Emotion detection
    emotion_features = emotion_vectorizer.transform([input_text])
    emotion_prediction = emotion_model.predict(emotion_features)
    emotion_probabilities = emotion_model.predict_proba(emotion_features).flatten()

    # Create emotion plot
    plt.figure(figsize=(8, 4))
    emotions = [EMOTION_MAP[i] for i in range(len(emotion_probabilities))]
    plt.bar(emotions, emotion_probabilities, color='skyblue')
    plt.title("Emotion Probabilities")
    plt.ylabel("Probability")
    plt.xlabel("Emotions")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=80)
    buf.seek(0)
    base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    buf.close()
    plt.close()  # Free memory

    dominant_emotion = EMOTION_MAP.get(int(emotion_prediction[0]), "Unknown")

    return jsonify({
        "text": input_text,
        "sentiment": {"score": positive_score, "sentiment": sentiment},
        "dominant_emotion": {"emotion": dominant_emotion},
        "emotion_plot": "data:image/png;base64," + base64_image
    })

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
