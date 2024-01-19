
# Twitter Sentiment Analysis

## Introduction
This sentiment analysis project, conducted by a team of data scientists, aims to harness the power of Natural Language Processing (NLP) to analyze public sentiment towards technology giants Apple and Google on Twitter. By leveraging a dataset of over 9,000 manually annotated tweets, the team has developed and evaluated models to classify sentiments as positive, negative, or neutral. The insights derived from this analysis can be valuable for businesses, marketing strategists, and researchers in the consumer electronics domain.

## Problem Statement
In a dynamic digital landscape dominated by social media, understanding public sentiment towards technology companies is crucial. The challenge at hand is to develop an NLP model capable of accurately discerning sentiments within a vast array of tweets related to Apple and Google products. The model must address the intricacies of social media language, varying sentiment intensity, and the challenge of handling a three-class classification system (positive, negative, and neutral). The ultimate goal is to provide a tool that offers nuanced insights into public sentiment, aiding businesses in making informed decisions.

## Features

- Data Understanding
- Data Cleaning and Preparation
- Exploratory Data Analysis and Visualization
- Data Preprocessing
- Modelling
    -Base Model

    -Multiclass Model

- Conclusion
- Next Steps

## Objectives

1. Identify unique words associated with positive, negative, and neutral sentiments in the dataset.
2. Initiate with a binary classifier for positive and negative sentiments, gradually extending to handle neutral sentiments (Proof of Concept).
3. Develop a sophisticated NLP model capable of accurately classifying tweet sentiments as positive, negative, or neutral, with an accuracy target of 80% and above.

## Challenges

- Diverse language patterns on social media, including slang and abbreviations.
- Nuanced classification of the broad neutral sentiment category.

## Solutions

- Robust text preprocessing pipeline with tokenization, lemmatization, and handling of slang.
- Granular approach to neutral sentiment classification, exploring sentiment intensity analysis.


## Dependencies for the Project
To reproduce and extend the project, the following dependencies are crucial:

1. **Python Environment:** The project is implemented in Python. Ensure you have Python installed on your system.

2. **Libraries and Packages:**
   - NumPy
   - Pandas
   - Scikit-learn
   - NLTK (Natural Language Toolkit)
   - Imbalanced-learn (for handling class imbalance)
   - Matplotlib and Seaborn (for data visualization)

3. **Jupyter Notebooks:** The project involves the use of Jupyter Notebooks for data exploration, preprocessing, and model development. Ensure you have a Jupyter Notebook environment set up.

4. **Access to Twitter API (Optional):** If you plan to fetch new tweet data for real-time analysis, you may need access to the Twitter API. Note that obtaining access may take some time.

## Project Structure

1. **Data Understanding**

- Dataset from CrowdFlower via data.world with 9,093 manually annotated tweets.
- Columns: 'tweet_text,' 'emotion_in_tweet_is_directed_at,' 'is_there_an_emotion_directed_at_a_brand_or_product.'
- Labels for sentiments: positive, negative, or neutral.

2. **Data Cleaning and Preparation**

- Removal of duplicate entries and handling of null values.
- Text cleaning with regular expressions.
- Introduction of functions for tweet text cleaning and hashtag extraction.

3. **Exploratory Data Analysis and Visualization**

- Distribution analysis of tweet_text, emotion_in_tweet_is_directed_at, and is_there_an_emotion_directed_at_a_brand_or_product.
- Visualization of emotion distribution for specific brands, brand-emotion relationships, and common hashtags.

4. **Data Preprocessing**

- Text data refinement through stop word removal, English word filtering, lemmatization, and removal of one-word sentences.
- Label encoding for categorical columns.

5. **Modelling**

**Base Model**

- Binary classifier using Bernoulli Naive Bayes.
- TF-IDF vectorization with a maximum of 500 features.
- Accuracy: 86.72%.

**Multiclass Model**

- Logistic Regression and Random Forest models for multiclass classification.
- RandomOverSampler for addressing class imbalance.
- Accuracy for Random Forest model: 86.21%.

5. **Conclusion**

- The Random Forest classification model demonstrates strong overall performance, with potential areas for improvement in predicting negative sentiments. Additional data collection, hyperparameter tuning, and continuous monitoring are suggested next steps.

6.  **Next Steps**

- Consider additional data collection for negative sentiments.
- Explore hyperparameter tuning and feature engineering.
- Address class imbalance with techniques like resampling or adjusting class weights.
- Continuously monitor and refine the model for real-world effectiveness.

## Usage

1. Clone the repository.
2. Install dependencies using `requirements.txt`.
3. Run the provided notebooks for data analysis and model training.

## API Integration (Pending)

- To fetch new tweet data and run the model, a request to Twitter API (X) is required. Follow their instructions for API access and update the relevant sections in the code.

## Acknowledgments

- Dataset from CrowdFlower via data.world.
- Open-source libraries: scikit-learn, imbalanced-learn, matplotlib, seaborn.



## Authors
- [Maryimmaculate Kariuki](https://github.com/mumbikariuki)
- [William Omballa](https://github.com/Omballa)
- [Valerie Jerono](https://github.com/VAL-Jerono)
- [Ruth Nanjala](https://github.com/RuthNanjala)
