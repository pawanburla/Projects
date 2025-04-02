import joblib
import nltk
from sklearn.feature_extraction.text import CountVectorizer

# Ensure that you have downloaded the necessary resources for nltk
nltk.download('stopwords')

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


# Function to preprocess input text
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()

    # Remove non-alphabetic characters
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])

    # Tokenize the text
    tokens = text.split()

    # Remove stopwords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Optionally apply stemming or lemmatization (using PorterStemmer as an example)
    stemmer = nltk.PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Join the tokens back into a cleaned string
    return ' '.join(tokens)


# Ask the user for an input sentence
input_text = input("Enter a sentence to predict sentiment: ")

# Preprocess the input text
cleaned_input = preprocess_text(input_text)

# Transform the input text using the same vectorizer used during training
input_vectorized = vectorizer.transform([cleaned_input])

# Get the model's prediction
prediction = model.predict(input_vectorized)

# Output the prediction result
if prediction == 1:
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")
