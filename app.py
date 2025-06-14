# app.py (Corrected)

import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

# --- Place st.set_page_config() as the VERY FIRST Streamlit command ---
st.set_page_config(page_title="SMS Spam Classifier", page_icon="✉️")

# --- 1. Load the Saved Model and Vectorizer ---
@st.cache_resource # Cache the loading of heavy resources
def load_resources():
    """
    Loads the trained Random Forest model, TF-IDF vectorizer,
    and the list of numerical features.
    """
    try:
        model = joblib.load('tuned_random_forest_spam_classifier.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        numerical_features_list = joblib.load('numerical_features_list.pkl')
        
        # Ensure NLTK resources are downloaded for the app environment
        # These will run only if not already downloaded
        try:
            nltk.data.find('corpora/stopwords')
        except nltk.downloader.DownloadError:
            nltk.download('stopwords')
        try:
            nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
            nltk.download('punkt')
            
        return model, tfidf_vectorizer, numerical_features_list

    except FileNotFoundError:
        # Use st.error after set_page_config, but ensure it doesn't break the first command rule
        st.error("Error: Model files not found. Please ensure 'tuned_random_forest_spam_classifier.pkl', 'tfidf_vectorizer.pkl', and 'numerical_features_list.pkl' are in the same directory as this script.")
        st.stop() # Stop the app if files are missing
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        st.stop()

# Load resources once when the app starts
model, tfidf_vectorizer, numerical_features_list = load_resources()

# Initialize NLTK components for text cleaning (globally for efficiency)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# --- 2. Define Text Preprocessing Functions (Identical to Training) ---
def clean_text(text):
    """
    Applies the same text cleaning steps as during training.
    """
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = nltk.word_tokenize(text)
    cleaned_words = []
    for word in words:
        if word.isalnum() and word not in stop_words:
            cleaned_words.append(stemmer.stem(word))
    return ' '.join(cleaned_words)

def extract_numerical_features(message):
    """
    Extracts the same numerical features as during training from a single message.
    """
    num_characters = len(message)
    num_words = len(nltk.word_tokenize(message))
    num_sentences = len(nltk.sent_tokenize(message))
    num_uppercase_chars = sum(1 for char in message if char.isupper())
    num_digits = sum(1 for char in message if char.isdigit())
    num_punctuation = sum(1 for char in message if char in string.punctuation)
    
    return np.array([num_characters, num_words, num_sentences,
                     num_uppercase_chars, num_digits, num_punctuation])

# --- 3. Streamlit App Interface (Main Content) ---
st.title("✉️ SMS Spam Classifier")
st.markdown("Enter an SMS message below to classify it as **Spam** or **Not Spam (Ham)**.")

# Text input area for the user
user_input = st.text_area("Enter SMS message here:", height=150, help="Type or paste the SMS message you want to classify.")

# Button to trigger classification
if st.button("Classify SMS"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner("Classifying..."):
            # Step 1: Clean the text
            cleaned_input = clean_text(user_input)

            # Step 2: Transform with TF-IDF vectorizer
            # Use transform, not fit_transform, as the vectorizer is already fitted
            text_features = tfidf_vectorizer.transform([cleaned_input]).toarray()
            
            # Step 3: Extract numerical features
            numerical_features = extract_numerical_features(user_input).reshape(1, -1) # Reshape for hstack
            
            # Step 4: Combine all features (TF-IDF + numerical)
            # Ensure the order and number of features match the training data
            combined_features = np.hstack((text_features, numerical_features))

            # Step 5: Make prediction
            prediction = model.predict(combined_features)[0]
            prediction_proba = model.predict_proba(combined_features)

            st.write("---")
            if prediction == 1:
                st.error(f"**Prediction: SPAM!**")
                st.markdown(f"**Confidence (Spam):** {prediction_proba[0][1]*100:.2f}%")
            else:
                st.success(f"**Prediction: NOT SPAM (HAM)**")
                st.markdown(f"**Confidence (Ham):** {prediction_proba[0][0]*100:.2f}%")
            st.write("---")

st.markdown("""
<style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.2em;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 3px 3px 8px rgba(0,0,0,0.3);
    }
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("About This App")
st.sidebar.info(
    "This application uses a Machine Learning model (Random Forest Classifier) "
    "trained on the SMS Spam Collection Dataset. It leverages both text-based "
    "(TF-IDF) and engineered numerical features (like message length, digit count) "
    "to classify messages."
)
st.sidebar.markdown(
    "**Project Repository:** [GitHub - SMS Spam Filtration](https://github.com/utkarshkulkarni45/SMS-Spam-Filtration)"
)
