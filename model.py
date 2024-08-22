import pandas as pd 
import joblib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  GridSearchCV
import json
import streamlit as st

df = pd.read_csv("ODI_Match_info.csv")
df = df.dropna()

label_encoder_venue = LabelEncoder()
label_encoder_team1 = LabelEncoder()
label_encoder_team2 = LabelEncoder()
label_encoder_winner = LabelEncoder()
df['venue_encoded'] = label_encoder_venue.fit_transform(df['venue'])
df['team1_encoded'] = label_encoder_team1.fit_transform(df['team1'])
df['team2_encoded'] = label_encoder_team2.fit_transform(df['team2'])
df['winner_encoded'] = label_encoder_winner.fit_transform(df['winner'])
X = df[['team1_encoded', 'team2_encoded', 'venue_encoded']]
y = df['winner_encoded']
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_train_dt = dt.predict(x_train)
y_test_dt= dt.predict(x_test)
precision_dt = r2_score(y_test, y_test_dt)
train_accuracy_dt = r2_score(y_train, y_train_dt)
test_accuracy_dt = r2_score(y_test, y_test_dt)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(x_test)

y_train_pred = best_rf.predict(x_train)
y_test_pred = best_rf.predict(x_test)
train_accuracy_rf = accuracy_score(y_train, y_train_pred)
test_accuracy_rf = accuracy_score(y_test, y_test_pred)
precision_rf = r2_score(y_test, y_test_pred)


metrics = {
    'Training Accuracy dt': train_accuracy_dt,
    'Testing Accuracy dt': test_accuracy_dt,
    'Precision dt': precision_dt,
    'Training Accuracy rf': train_accuracy_rf,
    'Testing Accuracy rf': test_accuracy_rf,
    'Precision rf': precision_rf,
    'Best param' : best_params
}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)
