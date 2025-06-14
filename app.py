# app.py (Optimized Dark Theme)

import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np

# --- Place st.set_page_config() as the VERY FIRST Streamlit command ---
# This must be the first Streamlit command executed in the script.
st.set_page_config(page_title="SMS Spam Classifier", page_icon="✉️", layout="centered", initial_sidebar_state="expanded")

# --- 1. Load Saved Model, Vectorizer, and Download NLTK Data ---
@st.cache_resource # Cache the loading of heavy resources and NLTK downloads
def load_all_resources():
    """
    Loads the trained Random Forest model, TF-IDF vectorizer,
    list of numerical features, and ensures NLTK data is downloaded.
    This function also initializes the global NLTK components for consistent use.
    """
    try:
        # Load the pre-trained machine learning model and vectorizer
        model = joblib.load('tuned_random_forest_spam_classifier.pkl')
        tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        numerical_features_list = joblib.load('numerical_features_list.pkl')
        
        # --- NLTK Data Download ---
        # NLTK data (stopwords and punkt) is required for text preprocessing.
        # nltk.download() checks if data is already present before attempting a download,
        # making this robust for both local runs and cloud deployments.
        nltk.download('stopwords', quiet=True) # quiet=True to suppress download messages in logs
        nltk.download('punkt', quiet=True)     # quiet=True to suppress download messages in logs
        # --- END NLTK Data Download ---

        # Initialize NLTK components *after* their required data has been downloaded.
        # These are set as global variables to be accessible by preprocessing functions.
        global global_stop_words
        global global_stemmer
        global_stop_words = set(stopwords.words('english'))
        global_stemmer = PorterStemmer()
            
        return model, tfidf_vectorizer, numerical_features_list

    except FileNotFoundError:
        # Provide a clear error message if essential model files are missing.
        st.error("Error: Model files not found. Please ensure 'tuned_random_forest_spam_classifier.pkl', 'tfidf_vectorizer.pkl', and 'numerical_features_list.pkl' are in the same directory as this script.")
        st.stop() # Halt the application gracefully.
    except Exception as e:
        # Catch any other unexpected errors during the loading process.
        st.error(f"An unexpected error occurred while loading resources: {e}")
        st.stop() # Halt the application.

# Call the function to load all resources and download NLTK data once when the app starts.
# This ensures all necessary components are ready before the UI is rendered.
model, tfidf_vectorizer, numerical_features_list = load_all_resources()

# --- 2. Define Text Preprocessing Functions (Identical to Training Phase) ---
# These functions must exactly mirror the preprocessing steps used during model training
# to ensure consistent feature extraction for new input messages.

def clean_text(text):
    """
    Applies the same text cleaning steps as during training:
    lowercase conversion, punctuation removal, tokenization, stop word removal, and stemming.
    """
    text = text.lower() # Convert text to lowercase
    text = ''.join([char for char in text if char not in string.punctuation]) # Remove punctuation
    words = nltk.word_tokenize(text) # Tokenize text into words
    cleaned_words = []
    for word in words:
        # Filter out non-alphanumeric words and stop words, then apply stemming
        if word.isalnum() and word not in global_stop_words:
            cleaned_words.append(global_stemmer.stem(word))
    return ' '.join(cleaned_words) # Join processed words back into a string

def extract_numerical_features(message):
    """
    Extracts numerical features from a given message, replicating the feature
    engineering done during the training phase.
    """
    num_characters = len(message) # Total number of characters
    num_words = len(nltk.word_tokenize(message)) # Number of words
    num_sentences = len(nltk.sent_tokenize(message)) # Number of sentences
    num_uppercase_chars = sum(1 for char in message if char.isupper()) # Count uppercase characters
    num_digits = sum(1 for char in message if char.isdigit()) # Count digits
    num_punctuation = sum(1 for char in message if char in string.punctuation) # Count punctuation marks
    
    # Return as a NumPy array, reshaped to (1, -1) for horizontal stacking with TF-IDF features.
    return np.array([num_characters, num_words, num_sentences,
                     num_uppercase_chars, num_digits, num_punctuation]).reshape(1, -1)

# --- 3. Streamlit App Interface (Main Content Layout) ---
st.title("✉️ SMS Spam Classifier")
st.markdown("Enter an SMS message below to classify it as **Spam** or **Not Spam (Ham)**.")

# Text input area where the user can type or paste an SMS message.
user_input = st.text_area("Enter SMS message here:", height=150, help="Type or paste the SMS message you want to classify.")

# Button to trigger the classification process.
if st.button("Classify SMS"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.") # Prompt user if input is empty
    else:
        with st.spinner("Classifying..."): # Show a spinner while processing
            # --- Preprocessing the user input ---
            # Step 1: Clean the raw text input.
            cleaned_input = clean_text(user_input)

            # Step 2: Transform the cleaned text into TF-IDF features.
            # Use the loaded tfidf_vectorizer's .transform() method.
            text_features = tfidf_vectorizer.transform([cleaned_input]).toarray()
            
            # Step 3: Extract numerical features from the original user input.
            numerical_features = extract_numerical_features(user_input)
            
            # Step 4: Combine all features into a single array.
            # This array must have the same feature order as the training data.
            combined_features = np.hstack((text_features, numerical_features))

            # --- Model Prediction ---
            # Step 5: Make a prediction using the loaded Random Forest model.
            prediction = model.predict(combined_features)[0] # Get the single prediction (0 for ham, 1 for spam)
            prediction_proba = model.predict_proba(combined_features) # Get probability scores for both classes

            st.write("---") # Visual separator

            # --- Display Classification Result ---
            if prediction == 1:
                st.error(f"**Prediction: SPAM!**") # Display a prominent error message for spam
                st.markdown(f"**Confidence (Spam):** {prediction_proba[0][1]*100:.2f}%") # Show confidence for spam class
            else:
                st.success(f"**Prediction: NOT SPAM (HAM)**") # Display a prominent success message for ham
                st.markdown(f"**Confidence (Ham):** {prediction_proba[0][0]*100:.2f}%") # Show confidence for ham class
            st.write("---") # Visual separator

# --- Custom CSS for a consistent DARK THEME ---
# This CSS aims to ensure all text and elements are perfectly visible on a dark background.
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
