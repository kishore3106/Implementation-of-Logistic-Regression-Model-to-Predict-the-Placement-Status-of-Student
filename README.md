# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1: Data Collection and Preprocessing
Load the placement dataset and remove unnecessary columns. Check for missing and duplicate values, and convert all categorical variables into numerical form using Label Encoding.

Step 2: Feature Selection and Data Splitting
Separate the dataset into independent variables (features) and the dependent variable (placement status). Split the data into training and testing sets.

Step 3: Model Training
Apply the Logistic Regression algorithm on the training data to build the prediction model.

Step 4: Prediction and Evaluation
Use the trained model to predict placement status on test data and evaluate the performance using accuracy score, confusion matrix, and classification report.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: KISHORE B
RegisterNumber: 212224100032
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:/Users/91908/Downloads/Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])
datal

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)
y_pred

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)


classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
print("Prediction of LR:")
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])



```

## Output:
HEAD
<img width="1083" height="238" alt="Screenshot 2026-02-04 093045" src="https://github.com/user-attachments/assets/22494173-bd80-4a3c-a0d3-84b4dd0c4c91" />
COPY
<img width="1093" height="268" alt="Screenshot 2026-02-04 093058" src="https://github.com/user-attachments/assets/df34bda6-61b2-4f43-bafe-7f0ab349a61d" />
FIT TRANSFORM
<img width="1083" height="627" alt="Screenshot 2026-02-04 093113" src="https://github.com/user-attachments/assets/836d86c2-994b-4da2-ac4e-469bde910fc5" />
LOGISTIC REGRESSION

<img width="601" height="91" alt="Screenshot 2026-02-04 093809" src="https://github.com/user-attachments/assets/b9c408cb-e023-48e9-9a59-2c371948c22b" />
  
ACCURACY SCORE
<img width="1092" height="76" alt="Screenshot 2026-02-04 093131" src="https://github.com/user-attachments/assets/23fe1f81-867e-45a4-839f-8b19ec58b233" />
CONFUSION MATRIX
<img width="1095" height="106" alt="Screenshot 2026-02-04 093142" src="https://github.com/user-attachments/assets/66d70bd6-f5c3-4a71-bd67-4f34ee9873a1" />
CLASSIFICATION REPORT & PREDICTION
<img width="585" height="405" alt="Screenshot 2026-02-04 094809" src="https://github.com/user-attachments/assets/a77b470d-f551-4897-bddd-4723d18c75c4" />








## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
