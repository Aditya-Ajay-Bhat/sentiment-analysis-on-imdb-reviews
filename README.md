# IMDB Movie Review Sentiment Analysis using Classical Machine Learning Models

## 📜 Description

This project performs sentiment analysis on the IMDB movie review dataset. The goal is to classify whether a given movie review has a "positive" or "negative" sentiment. The notebook walks through a complete machine learning pipeline: from extensive text preprocessing and feature engineering to training, evaluating, and comparing four different classical machine learning models:
* Support Vector Machine (SVM)
* Logistic Regression
* XGBoost
* Random Forest

The project also includes steps for model optimization and saving the trained models for future use.

## 💾 Dataset

The dataset used is the **IMDB Dataset.csv**, which contains 50,000 movie reviews. Each review is labeled with either a "positive" or "negative" sentiment.

-   **review**: The text content of the movie review.
-   **sentiment**: The corresponding sentiment label ('positive' or 'negative').

## ⚙️ Workflow Pipeline

The notebook is structured as follows:

### 1. Setup & Data Loading
-   Mounts Google Drive to access the dataset.
-   Loads the `IMDB Dataset.csv` into a pandas DataFrame.

### 2. Text Preprocessing
A series of steps are applied to clean and standardize the raw review text to make it suitable for feature extraction:
-   **Label Encoding**: The categorical `sentiment` labels ('positive', 'negative') are mapped to numerical values (1, 0).
-   **Noise Removal**: URLs and HTML tags are removed from the reviews using regular expressions.
-   **Normalization**: All text is converted to lowercase.
-   **Punctuation & Special Character Removal**: All punctuation and non-alphanumeric characters are removed.
-   **Tokenization**: The cleaned text is split into a list of individual words (tokens) using NLTK's `word_tokenize`.
-   **Stopword Removal**: Common English words that do not contribute much to the sentiment (e.g., "the", "a", "is") are removed using NLTK's stopwords list.
-   **Lemmatization**: Words are reduced to their base or dictionary form (lemma) to handle variations. For example, "watching" and "watched" both become "watch". NLTK's `WordNetLemmatizer` is used for this step.

### 3. Feature Engineering: TF-IDF Weighted Word2Vec
To convert the processed text into numerical vectors that machine learning models can understand, a hybrid approach combining Word2Vec and TF-IDF is used:
-   **Word2Vec Training**: A `Word2Vec` model is trained on the lemmatized tokens to generate a 100-dimensional vector embedding for each unique word in the vocabulary. These embeddings capture the semantic meaning and context of words.
-   **TF-IDF Weighting**: A `TfidfVectorizer` is used to calculate the TF-IDF score for each word, which represents its importance in a document relative to the entire corpus.
-   **Vector Combination**: For each review, the final feature vector is created by taking the mean of the Word2Vec vectors of its words, where each word vector is weighted by its TF-IDF score. This gives more importance to words that are significant for a particular review.

### 4. Model Training and Evaluation
-   **Data Splitting**: The dataset is split into training (80%) and testing (20%) sets.
-   **Model Implementation**: Four classification models are trained on the feature vectors:
    -   Support Vector Machine (`SVC`)
    -   Logistic Regression (`LogisticRegression`)
    -   XGBoost (`XGBClassifier`)
    -   Random Forest (`RandomForestClassifier`)
-   **Evaluation**: The models are evaluated on the test set using the following metrics:
    -   Accuracy
    -   Precision
    -   Recall
    -   Confusion Matrix
    -   ROC AUC Score & Curve

### 5. Model Optimization
-   **Logistic Regression**: The `max_iter` parameter is increased to 1000 to ensure the model converges properly.
-   **XGBoost**: `RandomizedSearchCV` is used to perform hyperparameter tuning and find an optimal set of parameters for the XGBoost model, improving its performance.

### 6. Model Saving
-   The final trained models (including the optimized ones) are saved to `.pkl` files using `joblib` for easy reuse and deployment.

## 📊 Results

The SVM model demonstrated the best overall performance, particularly in its ability to distinguish between positive and negative classes, as shown by the highest ROC AUC score. Optimizing the XGBoost model with `RandomizedSearchCV` resulted in a significant performance boost, making it highly competitive with the SVM model.

Here is a summary of the final model performance:

| Model                             | Accuracy | Precision | Recall | ROC AUC |
| :-------------------------------- | :------- | :-------- | :----- | :------ |
| SVM                               | 0.8709   | 0.8642    | 0.8825 | 0.9444  |
| Logistic Regression (Initial)     | 0.8617   | 0.8580    | 0.8694 | 0.9351  |
| Logistic Regression (Retrained)   | 0.8614   | 0.8581    | 0.8686 | 0.9351  |
| XGBoost (Initial)                 | 0.8530   | 0.8505    | 0.8593 | 0.9308  |
| **XGBoost (Optimized)** | **0.8703** | **0.8649** | **0.8801** | **0.9418** |
| Random Forest                     | 0.8371   | 0.8312    | 0.8492 | 0.9154  |

### ROC AUC Curves
![ROC Curves for all models](https://i.imgur.com/k93yG7N.png)

## 🚀 How to Run

1.  **Prerequisites**: Ensure you have Python and Jupyter Notebook/Google Colab environment set up.
2.  **Dataset**: Download the `IMDB Dataset.csv` and place it in your Google Drive under a folder named `My Drive`.
3.  **Libraries**: Install the required libraries. The notebook uses `pandas`, `numpy`, `scikit-learn`, `nltk`, `gensim`, `xgboost`, and `matplotlib`. You can install them using pip:
    ```bash
    pip install pandas numpy scikit-learn nltk gensim xgboost matplotlib joblib
    ```
4.  **NLTK Data**: The notebook will automatically download the necessary NLTK data packages (`punkt`, `stopwords`, `wordnet`).
5.  **Execution**: Open the `classical_Ml_Sentiment_analysis.ipynb` notebook in Google Colab or Jupyter and run the cells in sequential order.
