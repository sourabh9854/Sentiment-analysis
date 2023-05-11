# Sentiment-analysis
This project performs sentiment analysis on customer reviews using machine learning techniques. It aims to classify customer sentiments as positive, negative, or neutral based on their reviews.

## Overview

The sentiment analysis process involves the following steps:

1. **Data Cleaning**: The dataset is preprocessed to handle missing values, remove special characters, and convert text to lowercase.

2. **Text Preprocessing**: The text is tokenized, stopwords are removed, and words are lemmatized to prepare the data for modeling.

3. **Feature Engineering**: The textual data is transformed into numerical features using TF-IDF vectorization. Additional numerical features, such as ratings, are also included.

4. **Resampling**: To handle imbalanced classes, resampling techniques like oversampling and undersampling are applied to create a more balanced training dataset.

5. **Model Training**: Several machine learning models, including Logistic Regression, XGBoost, and Random Forest, are trained on the resampled data.

6. **Model Evaluation**: The trained models are evaluated using precision, recall, F1-score, and accuracy metrics to measure their performance.


