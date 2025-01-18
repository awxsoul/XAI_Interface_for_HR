
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
import lime
import lime.lime_tabular
import shap
shap.initjs()

from flask import Flask, render_template
from flask import Flask, render_template, request
##########################
#data from kaggle
path = kagglehub.dataset_download("ayushtankha/70k-job-applicants-data-human-resource")
df = pd.read_csv(path+'/stackoverflow_full.csv')

#refining data
df.dropna(inplace=True)
df.isna().sum()
X = df.drop(['Employed','Unnamed: 0','Accessibility','Employment','Gender','MentalHealth','Country','PreviousSalary','ComputerSkills'],axis=1)
y = df['Employed']

data_initial=X

print("Preprocess Completed")
#########################

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

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    print("Model dumped")

y_pred = model.predict(X_test)

accuracy = roc_auc_score(y_test,y_pred)
#########################

# Create a LimeTabularExplainer object
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.toarray(),  # Convert sparse matrix to dense array
    feature_names=feature_names,
    class_names=['Not Employed', 'Employed'],
    mode='classification'
)

################################


#############################3


app = Flask(__name__)


@app.route('/',methods=['GET', 'POST'])
def index():
    size=100
    cand_data=data_initial.iloc[0]
    skills=cand_data["HaveWorkedWith"].split(";")
    if request.method == 'POST':
        button_value = request.form['cand_value']
        info=button_value
        num=int(info.split()[-1])
        instance = X_test[num-1].toarray()[0]
        exp = explainer.explain_instance(instance, model.predict_proba, num_features=10)
        print("Explanation for : ",button_value)
        html_explanation = exp.as_html()
        cand_data=data_initial.iloc[num-1]
        skills=cand_data["HaveWorkedWith"].split(";")

        return render_template('index.html',status="Data Loaded",size=size,info=html_explanation,cand_data=cand_data,skills=skills)
    else:
        return render_template('index.html',status="Data Loaded",size=size,info="html_explanation",cand_data=cand_data,skills=skills)
    

if __name__ == '__main__':
  app.run(debug=True)