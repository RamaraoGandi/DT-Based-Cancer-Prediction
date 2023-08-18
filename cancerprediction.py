import pandas as pd
import numpy as np

dataset = pd.read_csv("/content/DT-Based-Cancer-Prediction/data/cell_samples.csv")
dataset.loc[dataset['BareNuc']=='?'].shape
dataset.drop(dataset.loc[dataset['BareNuc']=='?'].index,inplace=True)
dataset['BareNuc'] = dataset['BareNuc'].astype('int64')

X = dataset[["Clump","UnifSize","UnifShape","MargAdh","SingEpiSize","BareNuc","BlandChrom","NormNucl","Mit"]]
y = dataset["Class"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="entropy") #Information gain as criteria
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

from sklearn.metrics import accuracy_score
print(f"Accuacy of the model is {accuracy_score(y_test,y_pred)*100}") #--> this test accuracy