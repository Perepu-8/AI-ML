import numpy as np
import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from collections import Counter

# Scikit-learn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

# 1. SETUP & DOWNLOADS
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = str(text).lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

# 2. DATA LOADING & CLEANING
df = pd.read_csv('spam.csv', encoding='latin-1')

# Drop unnecessary columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, errors='ignore')

# Rename columns
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode target
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target']) # ham=0, spam=1

# Remove duplicates
df = df.drop_duplicates(keep='first')

# 3. EDA
df['num_characters'] = df['text'].apply(len)
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))

print("Preprocessing text... Please wait.")
df['transformed_text'] = df['text'].apply(transform_text)

# 4. VECTORIZATION
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# 5. MODEL DEFINITIONS
svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs = {
    'SVC': svc, 'KN': knc, 'NB': mnb, 'DT': dtc, 'LR': lrc, 
    'RF': rfc, 'AdaBoost': abc, 'BgC': bc, 'ETC': etc, 'GBDT': gbdt, 'xgb': xgb
}

def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy, precision

# 6. TRAINING LOOP
accuracy_scores = []
precision_scores = []

for name, clf in clfs.items():
    curr_acc, curr_prec = train_classifier(clf, X_train, y_train, X_test, y_test)
    print(f"Algorithm: {name} | Accuracy: {curr_acc:.4f} | Precision: {curr_prec:.4f}")
    accuracy_scores.append(curr_acc)
    precision_scores.append(curr_prec)

# 7. PERFORMANCE DATAFRAME (Outside the loop to avoid length errors)
performance_df = pd.DataFrame({
    'Algorithm': list(clfs.keys()),
    'Accuracy': accuracy_scores,
    'Precision': precision_scores
}).sort_values('Precision', ascending=False)

print("\n--- Final Performance Table ---")
print(performance_df)

# Plotting Results
plt.figure(figsize=(10, 6))
sns.barplot(x='Algorithm', y='Accuracy', data=performance_df)
plt.title('Algorithm Accuracy Comparison')
plt.xticks(rotation='vertical')
plt.show()

# 8. VOTING CLASSIFIER (Best performing models combined)
voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)], voting='soft')
voting.fit(X_train, y_train)
y_pred_vote = voting.predict(X_test)
print(f"\nVoting Accuracy: {accuracy_score(y_test, y_pred_vote)}")
print(f"Voting Precision: {precision_score(y_test, y_pred_vote)}")

# 9. EXPORTING MODELS
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))

print("\nFiles 'vectorizer.pkl' and 'model.pkl' saved successfully!")