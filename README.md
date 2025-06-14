# SMS Spam Classifier

## Overview
This project develops a machine learning model to accurately classify SMS messages as either "ham" (legitimate) or "spam". It demonstrates a complete end-to-end machine learning pipeline, focusing on Natural Language Processing (NLP) techniques, comprehensive feature engineering, and robust model evaluation.

## Problem Statement
SMS spam is a pervasive issue, leading to annoyance, potential security risks (phishing), and wasted time. Building an automated system to filter these unwanted messages is crucial for improving user experience and digital security.

## Dataset
The project utilizes the **SMS Spam Collection Dataset** from the UCI Machine Learning Repository (available on Kaggle). This dataset consists of 5,572 SMS messages, each labeled as 'ham' or 'spam'.

## Methodology

### Data Loading and Initial Exploration
- Loaded the `spam.csv` dataset using `pandas`.
- Performed initial cleaning by dropping unnecessary columns and renaming 'v1' to 'label' and 'v2' to 'message'.
- Converted 'ham'/'spam' labels to numerical `0`/`1` respectively.
- Analyzed the class distribution (ham vs. spam) and message length characteristics.

### Feature Engineering
This project emphasizes comprehensive feature engineering, combining both text-based and numerical features to enrich the model's understanding:

1.  **Text Preprocessing:**
    -   Converted all text to **lowercase**.
    -   **Removed punctuation**.
    -   **Tokenized** messages into individual words.
    -   **Removed common English stopwords** (e.g., "the", "is", "a").
    -   Applied **Porter Stemming** to reduce words to their root form (e.g., "running" -> "run").
2.  **Text Feature Extraction (TF-IDF):**
    -   Transformed the cleaned text messages into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization**. This technique weighs words based on their relevance across the entire message corpus.
3.  **Numerical Feature Engineering:**
    -   Engineered several **additional numerical features** from the raw message text to capture non-textual patterns, which are often indicative of spam:
        -   `message_length`: Total number of characters in the message.
        -   `num_words`: Count of words in the message.
        -   `num_sentences`: Count of sentences in the message.
        -   `num_uppercase_chars`: Count of uppercase characters (spam often uses excessive capitalization).
        -   `num_digits`: Count of digits (spam often includes phone numbers or codes).
        -   `num_punctuation`: Count of punctuation marks (spam can use excessive punctuation).
    -   These numerical features were **concatenated with the TF-IDF features** to form a comprehensive feature matrix for model training.

### Model Training
The combined feature set was split into training and testing sets (80/20 split) with stratification to maintain class balance. Several classification algorithms were trained and evaluated:

-   **Multinomial Naive Bayes**
-   **Logistic Regression**
-   **Random Forest Classifier**

### Hyperparameter Tuning
**GridSearchCV** was employed to fine-tune the hyperparameters of the **Random Forest Classifier** to optimize its performance. Key parameters like `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, and `min_samples_leaf` were explored.

## Results

| Model                     | Accuracy | Precision (Spam) | Recall (Spam) | F1-Score (Spam) |
| :------------------------ | :------- | :--------------- | :------------ | :-------------- |
| Multinomial Naive Bayes   | 0.9525   | 0.7892           | 0.8792        | 0.8317          |
| Logistic Regression       | 0.9785   | 0.9699           | 0.8658        | 0.9149          |
| **Random Forest Classifier (Untuned)** | **0.9848** | **1.0000** | **0.8859** | **0.9395** |
| **Random Forest Classifier (Tuned)** | **0.9848** | **1.0000** | **0.8859** | **0.9395** |

**Key Findings:**
-   The **Random Forest Classifier** emerged as the top-performing model, achieving **perfect precision (1.0000) for the spam class** on the test set. This is crucial as it means the model does not misclassify legitimate "ham" messages as "spam", minimizing user inconvenience.
-   Despite hyperparameter tuning, the Random Forest model's performance on the test set remained consistent, indicating that its initial parameters were already highly effective for this dataset. This confirms the robustness of the chosen model and feature set.
-   The engineered numerical features likely contributed significantly to the superior performance of Logistic Regression and Random Forest compared to Multinomial Naive Bayes, demonstrating the value of combining different feature types.

## Conclusion
This project successfully developed a highly accurate and reliable SMS spam classifier using a combination of NLP techniques and creative feature engineering. The Random Forest Classifier achieved exceptional performance, particularly in avoiding false positives, making it a strong candidate for real-world application.

## Future Work
-   Explore more advanced NLP techniques like Word Embeddings (Word2Vec, GloVe) or contextual embeddings (BERT) for text representation.
-   Experiment with deep learning models (e.g., LSTMs, Transformers) for potentially higher accuracy, though often at the cost of interpretability and complexity.
-   Investigate other ensemble methods or stacking different models.
-   **Deploy the model as a web service (e.g., using Flask or Streamlit) for interactive predictions.**

## How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/utkarshkulkarni45/SMS-Spam-Filtration.git](https://github.com/utkarshkulkarni45/SMS-Spam-Filtration.git)
    cd SMS-Spam-Filtration
    ```
2.  **Install Python 3.10.x:**
    It is highly recommended to use **Python 3.10.x** (e.g., [Python 3.10.12](https://www.python.org/downloads/release/python-31012/)) for optimal compatibility with `scikit-learn==1.2.2`. Ensure you select "Add Python to PATH" during installation.
3.  **Create and Activate a Virtual Environment:**
    ```bash
    # Create a virtual environment using Python 3.10
    py -3.10 -m venv spam_app_env

    # Activate the virtual environment
    # On Windows Command Prompt:
    spam_app_env\Scripts\activate
    # On Git Bash / macOS / Linux:
    # source spam_app_env/bin/activate
    ```
    *(Note: `spam_app_env` folder is ignored by Git and will not be pushed to the repository.)*
4.  **Install Dependencies:**
    With the virtual environment activated, install all required packages:
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    Your application should open in your default web browser (usually at `http://localhost:8501`).
---
