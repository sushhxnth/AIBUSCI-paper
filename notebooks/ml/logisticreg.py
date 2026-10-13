import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model  import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer

data = pd.read_csv(r"exoplanets.csv\exoplanets.csv")
data = data.dropna(axis=1, how='all')
data = data.drop(columns=['kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition', 'koi_tce_delivname'])#all strings 
x = data.drop('koi_disposition', axis=1)
y = data['koi_disposition']
le = LabelEncoder()
y = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42, stratify=y)
se = SimpleImputer(strategy='mean')
x_train = se.fit_transform(x_train)
x_test =  se.transform(x_test)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("accuracy is:", accuracy_score(y_test, y_pred))
print("\n Classification report is:", classification_report(y_test, y_pred))
print(("confusion matrix is:", confusion_matrix(y_test, y_pred)))

