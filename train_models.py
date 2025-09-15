import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("final.csv")

# Fill NaN values with the mean of each column
df.fillna(df.mean(), inplace=True)

X = df.drop("diabetes", axis=1)
y = df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with open("model_probabilities.txt", "w") as f:
    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_preds)
    f.write(f"XGBoost Accuracy: {xgb_accuracy}\n")
    xgb_probs = xgb.predict_proba(X_test)
    f.write("XGBoost Probabilities:\n")
    f.write(str(xgb_probs) + "\n")

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_preds)
    f.write(f"\nRandom Forest Accuracy: {rf_accuracy}\n")
    rf_probs = rf.predict_proba(X_test)
    f.write("\nRandom Forest Probabilities:\n")
    f.write(str(rf_probs) + "\n")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_preds)
    f.write(f"\nLogistic Regression Accuracy: {lr_accuracy}\n")
    lr_probs = lr.predict_proba(X_test)
    f.write("\nLogistic Regression Probabilities:\n")
    f.write(str(lr_probs) + "\n")
