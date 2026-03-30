# SMS Spam Detector

## Project Name

SMS Spam Detector

## Project Overview

This project is a Python-based machine learning pipeline that detects whether an SMS message is spam or ham (safe). It uses a labeled SMS dataset to train a text classifier and saves the trained model for later prediction.

## Features

- Train and evaluate multiple machine learning classifiers on SMS text data
- Preprocess input text by cleaning, tokenizing, removing stopwords, and stemming
- Convert text to numeric features using TF-IDF
- Save a trained model and vectorizer for reuse
- Classify new SMS messages from the terminal in a lightweight runtime script

## Technologies and Tools Used

- Python
- pandas
- NumPy
- NLTK
- scikit-learn
- xgboost
- matplotlib
- seaborn
- wordcloud
- pickle

## Installation

1. Open a terminal in the project folder.
2. Install dependencies:

```bash
pip install numpy pandas nltk scikit-learn xgboost matplotlib seaborn wordcloud
```

3. Make sure the following files exist in the project folder:

- `sms_spam_detector_model.py`
- `main.py`
- `spam.csv`
- `vectorizer.pkl`
- `model.pkl`

## Running the Project

### 1. Train the model

If you want to rebuild the model and vectorizer from the dataset, run:

```bash
python sms_spam_detector_model.py
```

This script will:

- load `spam.csv`
- clean and preprocess SMS text
- vectorize text using `TfidfVectorizer`
- train several classifiers
- print accuracy and precision for each model
- save `vectorizer.pkl` and `model.pkl`

### 2. Classify a new message

To classify a new SMS message using the saved model, run:

```bash
python main.py
```

Then enter your message when prompted.

Example:

```text
Enter SMS text: Win a FREE holiday now! Reply YES to claim your prize.
Prediction: spam
```

## Project Structure

- `sms_spam_detector_model.py` - training pipeline, model evaluation, and export
- `main.py` - terminal interface that loads saved files and predicts new SMS messages
- `spam.csv` - SMS dataset labeled with `spam` and `ham`
- `vectorizer.pkl` - saved TF-IDF vectorizer for text preprocessing
- `model.pkl` - saved trained classifier used for prediction
- `README.MD` - project documentation

## Testing Instructions

### Manual testing

1. Run `main.py`:

```bash
python main.py
```

2. Enter a new SMS message.
3. Verify the output is either `spam` or `ham`.

### Example test messages

- Ham example:
  - `Hey, are we still meeting at 7 tonight? Let me know.`
- Spam example:
  - `Win £1000 cash now! Text WIN to 80085 to claim.`

### Rebuild and verify

If you want to retrain the model and test the saved files:

```bash
python sms_spam_detector_model.py
python main.py
```

## Notes

- `main.py` depends on `vectorizer.pkl` and `model.pkl`; if these files are missing, it will not work.
- The saved model may not classify every new message correctly, especially if the message style differs from the training dataset.
- For better accuracy, add more labeled SMS examples and retrain the model.
