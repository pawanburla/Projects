Sentiment Analysis Project

This is a sentiment analysis project that uses machine learning techniques to predict the sentiment of French tweets (or any text input). The model is trained using the Naive Bayes algorithm and vectorized using CountVectorizer.

Project Overview

This project performs sentiment analysis on text data and classifies it as either "Positive" or "Negative." The model is trained using a labeled dataset of French tweets, and it uses natural language processing (NLP) techniques to preprocess the data and prepare it for machine learning. Once trained, the model is saved to a file and can be used to make predictions on new text inputs.

Files in the Project

french_tweets.csv: The dataset containing French tweets along with their corresponding sentiment labels.
train_model.py: The Python script that trains the sentiment analysis model and saves it to a file (sentiment_model.pkl) for future use.
predict_sentiment.py: The Python script used to predict the sentiment of a new input sentence.
sentiment_model.pkl: The trained sentiment analysis model (Naive Bayes).
vectorizer.pkl: The vectorizer used for converting text into a numerical format for model prediction.
Figure_1.png: A figure showcasing the results or visualizations (e.g., distribution of sentiment).
Setup Instructions

Prerequisites
Before you begin, ensure you have the following installed on your machine:

Python 3.x
pip (Python package manager)
Virtual environment (optional but recommended)
Step 1: Clone the repository
Clone this repository to your local machine using the following command:

git clone https://github.com/your-username/SentimentAnalysis.git
Step 2: Create a virtual environment (optional but recommended)
Navigate to the project directory:

cd SentimentAnalysis
Create a virtual environment:

python3 -m venv .venv
Activate the virtual environment:

For macOS/Linux:

source .venv/bin/activate
For Windows:

.venv\Scripts\activate
Step 3: Install dependencies
Install the required Python libraries by running the following command:

pip install -r requirements.txt
If requirements.txt is not available, you can manually install dependencies:

pip install pandas scikit-learn nltk joblib
Step 4: Download NLTK Resources
Before running the script, make sure to download the required NLTK resources by running the following Python code:

import nltk
nltk.download('stopwords')
Step 5: Train the Model (Optional)
If the model and vectorizer files (sentiment_model.pkl and vectorizer.pkl) are not already available, you will need to train the model yourself. To train the model, run:

python train_model.py
This will generate the sentiment_model.pkl and vectorizer.pkl files.

Step 6: Predict Sentiment
To predict the sentiment of a text input, run the predict_sentiment.py script:

python predict_sentiment.py
The script will ask you to enter a sentence, and it will then output whether the sentiment of the text is Positive or Negative.

Example:

Enter a sentence to predict sentiment: I love this product!
Sentiment: Positive
Model and Data

The model is trained on a dataset of French tweets with labels indicating positive or negative sentiment. You can modify the dataset or retrain the model with your own data if needed.

Troubleshooting

NLTK Data Issues: If you encounter issues downloading NLTK data, ensure that the download path is correct or try manually downloading the required datasets.
File Not Found Errors: If the sentiment_model.pkl or vectorizer.pkl files are missing, make sure to run the train_model.py script first to generate them.
Conclusion

This sentiment analysis project provides a foundation for analyzing the sentiment of text data. You can enhance the project by exploring other machine learning models, tuning hyperparameters, or expanding the dataset.

Feel free to fork the repository, contribute, or reach out if you have any questions!

