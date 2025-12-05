import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Load models
classifier = joblib.load('model.pkl')
tfidf_vectorizer = joblib.load('vectorizer.pkl')
emotion_model = joblib.load('emotion_model.pkl')
emotion_vectorizer = joblib.load('emotion_vectorizer.pkl')

EMOTION_MAP = {0: "anger", 1: "fear", 2: "joy", 3: "love", 4: "sadness", 5: "surprise"}

def analyze_text(text):
    if not text.strip():
        return "Please enter some text", None, "No emotion detected"
    
    # Sentiment analysis
    input_features = tfidf_vectorizer.transform([text])
    y_proba = classifier.predict_proba(input_features)
    positive_score = float(y_proba[0][1])
    
    sentiment = "Neutral"
    if positive_score < 0.4:
        sentiment = "Negative"
    elif positive_score > 0.6:
        sentiment = "Positive"
    
    # Emotion detection
    emotion_features = emotion_vectorizer.transform([text])
    emotion_prediction = emotion_model.predict(emotion_features)
    emotion_probabilities = emotion_model.predict_proba(emotion_features).flatten()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    emotions = list(EMOTION_MAP.values())
    bars = ax.bar(emotions, emotion_probabilities, color='skyblue')
    ax.set_title("Emotion Probabilities", fontsize=16)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_xlabel("Emotions", fontsize=12)
    
    # Add value labels on bars
    for bar, prob in zip(bars, emotion_probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    dominant_emotion = EMOTION_MAP[int(emotion_prediction[0])]
    
    sentiment_result = f"Sentiment: {sentiment} (Score: {positive_score:.3f})"
    
    return sentiment_result, fig, f"Dominant Emotion: {dominant_emotion}"

# Create Gradio interface
with gr.Blocks(title="Emotion & Sentiment Analyzer") as demo:
    gr.Markdown("# 🎭 Emotion Detection & Sentiment Analysis")
    gr.Markdown("Enter text to analyze emotions and sentiment")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Enter your text here:",
                placeholder="Type something to analyze...",
                lines=3
            )
            analyze_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column():
            sentiment_output = gr.Textbox(label="Sentiment Analysis")
            emotion_output = gr.Textbox(label="Emotion Detection")
    
    plot_output = gr.Plot(label="Emotion Probabilities")
    
    analyze_btn.click(
        analyze_text,
        inputs=[text_input],
        outputs=[sentiment_output, plot_output, emotion_output]
    )

if __name__ == "__main__":
    demo.launch()
