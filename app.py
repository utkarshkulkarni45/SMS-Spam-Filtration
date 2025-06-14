# app.py (Final Corrected Version)

import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

# --- Place st.set_page_config() as the VERY FIRST Streamlit command ---
# This must be the first Streamlit command executed.
st.set_page_config(page_title="SMS Spam Classifier", page_icon="✉️")

# --- 1. Load Saved Model, Vectorizer, and Download NLTK Data ---
@st.cache_resource # Cache the loading of heavy resources and NLTK downloads
def load_all_resources():
    """
    Loads the trained Random Forest model, TF-IDF vectorizer,
    list of numerical features, and ensures NLTK data is downloaded.
    """
    try:
        model = joblib.load('tuned_random_forest_spam_classifier.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        numerical_features_list = joblib.load('numerical_features_list.pkl')
        
        # --- NLTK Data Download (inside cache_resource for efficiency and robustness) ---
        # nltk.download() is designed to check if data is already present before downloading.
        # This simplifies error handling and ensures data availability on deployment.
        nltk.download('stopwords')
        nltk.download('punkt')
        # --- END NLTK Data Download ---
            
        return model, tfidf_vectorizer, numerical_features_list

    except FileNotFoundError:
        st.error("Error: Model files not found. Please ensure 'tuned_random_forest_spam_classifier.pkl', 'tfidf_vectorizer.pkl', and 'numerical_features_list.pkl' are in the same directory as this script.")
        st.stop() # Stop the app if crucial files are missing
    except Exception as e:
        # Catch any other general exceptions during resource loading
        st.error(f"An unexpected error occurred while loading resources: {e}")
        st.stop()

# Call the function to load all resources and download NLTK data once when the app starts
model, tfidf_vectorizer, numerical_features_list = load_all_resources()

# Initialize NLTK components for text cleaning (globally for efficiency)
# These lines must be AFTER the load_all_resources() call, as they depend on NLTK data.
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
        # Ensure word is alphanumeric and not a stop word after tokenization
        if word.isalnum() and word not in stop_words:
            cleaned_words.append(stemmer.stem(word))
    return ' '.join(cleaned_words)

def extract_numerical_features(message):
    """
    Extracts the same numerical features as during training from a single message.
    """
    # Note: len(message) counts characters, not words.
    num_characters = len(message)
    num_words = len(nltk.word_tokenize(message))
    num_sentences = len(nltk.sent_tokenize(message))
    num_uppercase_chars = sum(1 for char in message if char.isupper())
    num_digits = sum(1 for char in message if char.isdigit())
    num_punctuation = sum(1 for char in message if char in string.punctuation)
    
    # Return as a NumPy array, reshaped for horizontal stacking later
    return np.array([num_characters, num_words, num_sentences,
                     num_uppercase_chars, num_digits, num_punctuation]).reshape(1, -1)

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
            # Step 1: Clean the text input
            cleaned_input = clean_text(user_input)

            # Step 2: Transform cleaned text into TF-IDF features
            # Use .transform(), not .fit_transform(), as the vectorizer is already fitted.
            text_features = tfidf_vectorizer.transform([cleaned_input]).toarray()
            
            # Step 3: Extract numerical features from the original user input
            numerical_features = extract_numerical_features(user_input)
            
            # Step 4: Combine all features (TF-IDF + numerical)
            # np.hstack combines arrays horizontally. Ensure feature order matches training.
            combined_features = np.hstack((text_features, numerical_features))

            # Step 5: Make prediction using the loaded model
            prediction = model.predict(combined_features)[0] # [0] to get the single prediction value
            prediction_proba = model.predict_proba(combined_features) # Get probability scores

            st.write("---")
            if prediction == 1:
                st.error(f"**Prediction: SPAM!**")
                # Display confidence for the spam class (index 1)
                st.markdown(f"**Confidence (Spam):** {prediction_proba[0][1]*100:.2f}%")
            else:
                st.success(f"**Prediction: NOT SPAM (HAM)**")
                # Display confidence for the ham class (index 0)
                st.markdown(f"**Confidence (Ham):** {prediction_proba[0][0]*100:.2f}%")
            st.write("---")

# --- Custom CSS for better aesthetics ---
st.markdown("""
<style>
    /* Style for the Classify SMS button */
    .stButton>button {
        background-color: #4CAF50; /* Green background */
        color: white; /* White text */
        font-size: 1.2em; /* Larger font size */
        padding: 10px 24px; /* Padding inside the button */
        border-radius: 8px; /* Rounded corners */
        border: none; /* No border */
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2); /* Subtle shadow */
        transition: all 0.3s ease; /* Smooth transition on hover */
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker green on hover */
        box-shadow: 3px 3px 8px rgba(0,0,0,0.3); /* Larger shadow on hover */
    }
    /* Style for the text input area */
    .stTextArea textarea {
        border-radius: 8px; /* Rounded corners */
        border: 1px solid #ddd; /* Light grey border */
        padding: 10px; /* Padding inside the textarea */
        resize: vertical; /* Allow vertical resizing */
    }
    /* Overall app container styling for better layout */
    .stApp {
        background-color: #f0f2f6; /* Light background for the app */
    }
    .stBlock {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar for additional information ---
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
