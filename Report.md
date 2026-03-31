SMS Spam Detection Report
1. Problem Chosen
I chose to solve the problem of classifying SMS messages as either spam or ham (safe). The project builds a lightweight machine learning pipeline that takes raw SMS text, processes it, and predicts whether the message is unwanted spam.

2. Why It Matters
Spam messages remain a major source of fraud, phishing, and unwanted interruption on mobile devices.
Automatic filtering improves user experience by reducing noise and protecting users from malicious content.
Text classification is a foundational natural language processing task with real-world impact in email, messaging, and customer support systems.
3. Approach
The solution is implemented in two main parts:

sms_spam_detector_model.py: trains the model and saves artifacts
main.py: loads saved artifacts for real-time prediction
3.1 Data Preparation
Load dataset from spam.csv
Remove irrelevant columns and rename the target/text fields
Encode labels: ham=0, spam=1
Remove duplicate rows to improve model quality
3.2 Text Preprocessing
Normalize text to lowercase
Tokenize sentences into words
Keep only alphanumeric tokens
Remove English stopwords
Apply Porter stemming to reduce word forms
This preprocessing pipeline is implemented in both training and runtime code, ensuring consistent input handling.

3.3 Feature Engineering
Use TfidfVectorizer to convert cleaned SMS text into numeric features
Limit the vocabulary to a practical size (max_features=3000 in training) to reduce sparsity
3.4 Model Training and Evaluation
Split the dataset into training and testing sets
Train multiple classifiers:
Support Vector Classifier
K-Nearest Neighbors
Naive Bayes
Decision Tree
Logistic Regression
Random Forest
AdaBoost
Bagging
Extra Trees
Gradient Boosting
XGBoost
Evaluate each model using accuracy and precision
Optionally train a voting ensemble to combine best models
3.5 Export and Deployment
Save the trained TF-IDF vectorizer as vectorizer.pkl
Save the chosen classifier model as model.pkl
Use main.py to:
load these artifacts
preprocess a new SMS message
predict spam vs ham
display the result to the user
4. Key Decisions
Use a well-known, labeled SMS dataset from spam.csv for supervised training.
Choose TF-IDF as the feature extractor because it balances term frequency and document importance.
Use NLTK for tokenization, stopword removal, and stemming to standardize text input.
Train many candidate models rather than relying on a single classifier, enabling comparison and selection.
Persist the vectorizer and model with pickle so prediction is fast and does not require retraining.
5. Challenges Faced
Text classification is sensitive to preprocessing choices, so I needed a consistent pipeline for both training and runtime.
SMS text includes slang, punctuation, and irregular capitalization; cleaning and normalization were essential.
NLTK resource availability can vary, so the runtime script must ensure required corpora are downloaded before usage.
The model may not generalize perfectly to all unseen SMS styles, especially if the new text diverges from the training distribution.
Working with potentially unbalanced labels requires care when interpreting accuracy; precision is a better measure for spam detection.
6. What I Learned
Building an end-to-end machine learning pipeline involves more than just training a model: it includes data cleaning, feature extraction, evaluation, and deployment.
Consistency between training preprocessing and runtime preprocessing is critical for reliable predictions.
Saving model artifacts separately from code makes the classifier reusable and lightweight.
Evaluating multiple algorithms provides insight into which model behaves best for the problem.
Even a simple classifier can add value when applied correctly to a real-world text classification task.
7. Summary
This project demonstrates a complete SMS spam detection workflow:

define the problem,
prepare and clean data,
transform text into TF-IDF features,
train and compare classifiers,
save reusable artifacts,
deploy a prediction script for new messages.
The result is a practical classification system that turns raw SMS text into an actionable spam/ham decision while highlighting the importance of preprocessing, model selection, and persistence.
