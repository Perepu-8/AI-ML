import os
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

MODEL_FILENAME = 'model.pkl'
VECTORIZER_FILENAME = 'vectorizer.pkl'
ps = PorterStemmer()


def ensure_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


def transform_text(text):
    text = str(text).lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.isalnum()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [ps.stem(token) for token in tokens]
    return ' '.join(tokens)


def load_pickle(filename):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def main():
    ensure_nltk_resources()

    sms_text = input('Enter SMS text: ').strip()
    if not sms_text:
        print('No message provided. Exiting.')
        return

    try:
        vectorizer = load_pickle(VECTORIZER_FILENAME)
        model = load_pickle(MODEL_FILENAME)
    except FileNotFoundError:
        print('Error: vectorizer.pkl or model.pkl not found in this folder.')
        return

    cleaned = transform_text(sms_text)
    features = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(features)[0]
    print('Prediction:', 'spam' if prediction == 1 else 'ham')


if __name__ == '__main__':
    main()
