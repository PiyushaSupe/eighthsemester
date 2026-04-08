# Run using:
# python -m streamlit run app.py

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
import random

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="DistilBERT Demo", layout="wide")

# ----------------------------
# TITLE
# ----------------------------
st.title("🤖 DistilBERT Sentiment Analyzer")
st.markdown("Simulated Transformer Model (Demo UI)")

# ----------------------------
# INPUT TEXT
# ----------------------------
text = st.text_area("Enter Text:", "This movie was absolutely fantastic!")

# ----------------------------
# FAKE MODEL FUNCTION
# ----------------------------
def fake_prediction(text):
    time.sleep(1.5)  # simulate processing delay
    
    sentiment = random.choice(["Positive", "Negative"])
    confidence = round(random.uniform(0.7, 0.99), 2)
    
    return sentiment, confidence

# ----------------------------
# BUTTON ACTION
# ----------------------------
if st.button("Analyze"):
    
    st.subheader("🔍 Processing...")
    
    sentiment, confidence = fake_prediction(text)
    
    # ----------------------------
    # RESULT
    # ----------------------------
    st.success(f"Prediction: {sentiment}")
    st.write(f"Confidence: {confidence*100:.2f}%")
    
    # ----------------------------
    # PROBABILITY CHART
    # ----------------------------
    st.subheader("📊 Prediction Probabilities")
    
    probs = [confidence, 1 - confidence]
    labels = ["Positive", "Negative"]
    
    fig1, ax1 = plt.subplots()
    ax1.bar(labels, probs)
    ax1.set_title("Class Probabilities")
    ax1.set_ylabel("Probability")
    
    st.pyplot(fig1)
    
    # ----------------------------
    # ATTENTION MAP (NO SEABORN)
    # ----------------------------
    st.subheader("🧠 Attention Map (Simulated)")
    
    words = text.split()
    n = len(words)
    
    attention = np.random.rand(n, n)
    
    fig2, ax2 = plt.subplots()
    ax2.imshow(attention)
    
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    
    ax2.set_xticklabels(words, rotation=45)
    ax2.set_yticklabels(words)
    
    ax2.set_title("Attention Heatmap")
    
    plt.colorbar(ax2.imshow(attention), ax=ax2)
    
    st.pyplot(fig2)
    
    # ----------------------------
    # TOKEN IMPORTANCE
    # ----------------------------
    st.subheader("✨ Word Importance Scores")
    
    importance = np.random.rand(n)
    
    fig3, ax3 = plt.subplots()
    ax3.bar(words, importance)
    ax3.set_title("Token Importance")
    ax3.set_ylabel("Score")
    
    plt.xticks(rotation=45)
    
    st.pyplot(fig3)
    
    # ----------------------------
    # ATTENTION DISTRIBUTION
    # ----------------------------
    st.subheader("📈 Attention Distribution")
    
    flat_attention = attention.flatten()
    
    fig4, ax4 = plt.subplots()
    ax4.hist(flat_attention, bins=20)
    ax4.set_title("Attention Score Distribution")
    ax4.set_xlabel("Score")
    ax4.set_ylabel("Frequency")
    
    st.pyplot(fig4)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown("Built for NLP Mini Project 🎓")