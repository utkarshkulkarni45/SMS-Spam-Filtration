# app.py (NLTK Punkt Fix)

import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import os # Import os module

# --- Place st.set_page_config() as the VERY FIRST Streamlit command ---
# This must be the first Streamlit command executed in the script.
st.set_page_config(page_title="SMS Spam Classifier", page_icon="✉️", layout="centered", initial_sidebar_state="expanded")

# --- Set NLTK Data Path (Crucial for deployment environments) ---
# Define a directory to store NLTK data. Using a relative path within the app's
# root ensures it's persistent and accessible in the Streamlit Cloud environment.
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), 'nltk_data')
if not os.path.exists(NLTK_DATA_DIR):
    os.makedirs(NLTK_DATA_DIR)
nltk.data.path.append(NLTK_DATA_DIR)

# --- Global variables for NLTK components, initialized later ---
global_stop_words = None
global_stemmer = None

# --- 1. Load Saved Model, Vectorizer, and Download NLTK Data ---
@st.cache_resource # Cache the loading of heavy resources and NLTK downloads
def load_all_resources():
    """
    Loads the trained Random Forest model, TF-IDF vectorizer,
    list of numerical features, and ensures NLTK data is downloaded.
    This function also initializes the global NLTK components.
    """
    try:
        # Load the pre-trained machine learning model and vectorizer
        model = joblib.load('tuned_random_forest_spam_classifier.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        numerical_features_list = joblib.load('numerical_features_list.pkl')
        
        # --- NLTK Data Download (inside cache_resource for efficiency and robustness) ---
        # Specify download_dir to ensure NLTK data goes into our controlled directory
        nltk.download('stopwords', download_dir=NLTK_DATA_DIR, quiet=True)
        nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=True)
        # --- END NLTK Data Download ---

        # Initialize NLTK components *after* their required data has been downloaded.
        global global_stop_words
        global global_stemmer
        global_stop_words = set(stopwords.words('english'))
        global_stemmer = PorterStemmer()
            
        return model, tfidf_vectorizer, numerical_features_list

    except FileNotFoundError:
        st.error("Error: Model files not found. Please ensure 'tuned_random_forest_spam_classifier.pkl', 'tfidf_vectorizer.pkl', and 'numerical_features_list.pkl' are in the same directory as this script.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading resources: {e}")
        st.stop()

# Call the function to load all resources and download NLTK data once when the app starts.
model, tfidf_vectorizer, numerical_features_list = load_all_resources()

# --- 2. Define Text Preprocessing Functions (Identical to Training Phase) ---
def clean_text(text):
    """
    Applies the same text cleaning steps as during training.
    Uses the globally initialized NLTK components.
    """
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = nltk.word_tokenize(text)
    cleaned_words = []
    for word in words:
        if word.isalnum() and word not in global_stop_words:
            cleaned_words.append(global_stemmer.stem(word))
    return ' '.join(cleaned_words)

def extract_numerical_features(message):
    """
    Extracts numerical features from a given message.
    """
    num_characters = len(message)
    num_words = len(nltk.word_tokenize(message))
    num_sentences = len(nltk.sent_tokenize(message))
    num_uppercase_chars = sum(1 for char in message if char.isupper())
    num_digits = sum(1 for char in message if char.isdigit())
    num_punctuation = sum(1 for char in message if char in string.punctuation)
    
    return np.array([num_characters, num_words, num_sentences,
                     num_uppercase_chars, num_digits, num_punctuation]).reshape(1, -1)

# --- 3. Streamlit App Interface (Main Content Layout) ---
st.title("✉️ SMS Spam Classifier")
st.markdown("Enter an SMS message below to classify it as **Spam** or **Not Spam (Ham)**.")

user_input = st.text_area("Enter SMS message here:", height=150, help="Type or paste the SMS message you want to classify.")

if st.button("Classify SMS"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        with st.spinner("Classifying..."):
            cleaned_input = clean_text(user_input)
            text_features = tfidf_vectorizer.transform([cleaned_input]).toarray()
            numerical_features = extract_numerical_features(user_input)
            combined_features = np.hstack((text_features, numerical_features))

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

# --- Custom CSS for a consistent DARK THEME ---
st.markdown("""
<style>
    /* Global styling for the entire app */
    .stApp {
        background-color: #1a1e22; /* Very dark background for the whole app */
        color: white !important; /* Ensure all text is white by default */
    }

    /* Override default Streamlit text colors to ensure visibility */
    * {
        color: white !important; /* Force all text to be white */
    }

    /* Sidebar specific styling */
    .stSidebar {
        background-color: #24292e; /* Slightly lighter dark for sidebar */
    }

    /* Main content block styling (for st.container, st.columns, etc.) */
    .stBlock {
        background-color: #2e343e; /* A distinct dark shade for content blocks */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2); /* Subtle shadow */
    }

    /* Headers (h1, h2, etc.) */
    h1, h2, h3, h4, h5, h6 {
        color: white !important; /* Ensure all headers are white */
    }

    /* Markdown and paragraph text */
    .stMarkdown, p, li {
        color: white !important; /* Ensure readability for all text elements */
    }

    /* Text input area and its label */
    .stTextArea label, 
    label.css-1jc7a0p.e16fv1bt2 { /* Targeting Streamlit's specific label classes */
        color: white !important; /* Label text color */
    }
    .stTextArea textarea {
        background-color: #3d444e; /* Darker background for the input field itself */
        color: white !important; /* Text entered by user */
        border: 1px solid #666666; /* Lighter border for definition */
        border-radius: 8px;
        padding: 10px;
        resize: vertical;
    }

    /* Button styling */
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white !important;
        font-size: 1.2em;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker green on hover */
        box-shadow: 3px 3px 8px rgba(0,0,0,0.3);
    }

    /* Streamlit's alert/notification messages (e.g., st.success, st.error, st.warning) */
    .stAlert, .stNotification {
        background-color: #f0f0f0 !important; /* Light background for visibility */
        color: black !important; /* Black text for contrast */
        border-radius: 8px;
        padding: 10px;
    }
    .stAlert p, .stNotification p { /* Ensure text within alerts is black */
        color: black !important;
    }

</style>
""", unsafe_allow_html=True)

# --- Sidebar Content ---
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
