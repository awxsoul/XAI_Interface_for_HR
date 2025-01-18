import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
import kagglehub
from sklearn.compose import ColumnTransformer
import pickle
import matplotlib.pyplot as plt

##########################
#data from kaggle
path = kagglehub.dataset_download("ayushtankha/70k-job-applicants-data-human-resource")
df = pd.read_csv(path+'/stackoverflow_full.csv')

#refining data
df.dropna(inplace=True)
df.isna().sum()
X = df.drop(['Employed','Unnamed: 0','Accessibility','Employment','Gender','MentalHealth','Country','PreviousSalary','ComputerSkills'],axis=1)
y = df['Employed']

data=X

print("Preprocess Completed")
#########################
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Custom tokenizer for semicolon-separated languages
def custom_tokenizer(text):
    return text.split(';')

# Define categorical and text columns
categorical_cols = ['Age', 'EdLevel', 'MainBranch']

# Create TfidfVectorizer with the custom tokenizer and lowercase=False
vectorizer = TfidfVectorizer(lowercase=False, tokenizer=custom_tokenizer)
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)  # sparse=False for compatibility

# Create the ColumnTransformer
transformer = ColumnTransformer([
    ('vectorizer', vectorizer, 'HaveWorkedWith'),  # Apply vectorizer to 'HaveWorkedWith' column
    ('encoder', one_hot_encoder, categorical_cols)  # Apply one-hot encoder to categorical columns
])

print("Transformer Created")
#########################
X = transformer.fit_transform(X)
feature_names = transformer.get_feature_names_out()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = roc_auc_score(y_test,y_pred)

print("Model Trained")

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)