SMS Spam Detection Report
<br>
1. Problem Chosen:
<br>
I chose to solve the problem of classifying SMS messages as either spam or ham (safe). The project builds a lightweight machine learning pipeline that takes raw SMS text, processes it, and predicts whether the message is unwanted spam.
<br>
2. Why It Matters:
<br>
Spam messages remain a major source of fraud, phishing, and unwanted interruption on mobile devices.
Automatic filtering improves user experience by reducing noise and protecting users from malicious content.
Text classification is a foundational natural language processing task with real-world impact in email, messaging, and customer support systems.
<br>
3. Approach:
<br>
The solution is implemented in two main parts:
sms_spam_detector_model.py: trains the model and saves artifacts
main.py: loads saved artifacts for real-time prediction
<br>
3.1 Data Preparation
<br>
Load dataset from spam.csv
<br>
Remove irrelevant columns and rename the target/text fields
<br>
Encode labels: ham=0, spam=1
<br>
Remove duplicate rows to improve model quality
<br>
3.2 Text Preprocessing
<br>
Normalize text to lowercase
<br>
Tokenize sentences into words
<br>
Keep only alphanumeric tokens
<br>
Remove English stopwords
<br>
Apply Porter stemming to reduce word forms
<br>
This preprocessing pipeline is implemented in both training and runtime code, ensuring consistent input handling.
<br>
3.3 Feature Engineering:
<br>
Use TfidfVectorizer to convert cleaned SMS text into numeric features
Limit the vocabulary to a practical size (max_features=3000 in training) to reduce sparsity
<br>
3.4 Model Training and Evaluation:
<br>
Split the dataset into training and testing sets
<br>
Train multiple classifiers:
<br>
Support Vector Classifier,
K-Nearest Neighbors,
Naive Bayes,
Decision Tree,
Logistic Regression,
Random Forest,
AdaBoost,
Bagging,
Extra Trees,
Gradient Boosting,
XGBoost
<br>
Evaluate each model using accuracy and precision
Optionally train a voting ensemble to combine best models
<br>
3.5 Export and Deployment
<br>
Save the trained TF-IDF vectorizer as vectorizer.pkl
<br>
Save the chosen classifier model as model.pkl
<br>
Use main.py to:
<br>
load these artifacts
<br>
preprocess a new SMS message
<br>
predict spam vs ham
<br>
display the result to the user
<br>
4. Key Decisions
<br>
Use a well-known, labeled SMS dataset from spam.csv for supervised training.
<br>
Choose TF-IDF as the feature extractor because it balances term frequency and document importance.
<br>
Use NLTK for tokenization, stopword removal, and stemming to standardize text input.
<br>
Train many candidate models rather than relying on a single classifier, enabling comparison and selection.
<br>
Persist the vectorizer and model with pickle so prediction is fast and does not require retraining.
<br>
5. Challenges Faced:
<br>
Text classification is sensitive to preprocessing choices, so I needed a consistent pipeline for both training and runtime.
SMS text includes slang, punctuation, and irregular capitalization; cleaning and normalization were essential.
NLTK resource availability can vary, so the runtime script must ensure required corpora are downloaded before usage.
The model may not generalize perfectly to all unseen SMS styles, especially if the new text diverges from the training distribution.
Working with potentially unbalanced labels requires care when interpreting accuracy; precision is a better measure for spam detection.
<br>
6. What I Learned:
<br>
Building an end-to-end machine learning pipeline involves more than just training a model: it includes data cleaning, feature extraction, evaluation, and deployment.
Consistency between training preprocessing and runtime preprocessing is critical for reliable predictions.
Saving model artifacts separately from code makes the classifier reusable and lightweight.
Evaluating multiple algorithms provides insight into which model behaves best for the problem.
Even a simple classifier can add value when applied correctly to a real-world text classification task.
<br>
7. Summary:
<br>
This project demonstrates a complete SMS spam detection workflow:

define the problem,
prepare and clean data,
transform text into TF-IDF features,
train and compare classifiers,
save reusable artifacts,
deploy a prediction script for new messages.
The result is a practical classification system that turns raw SMS text into an actionable spam/ham decision while highlighting the importance of preprocessing, model selection, and persistence.
