#pip install pandas scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Step 1: Create dataset
data = {
    "hours_studied": [1,2,3,4,5,6,7,8],
    "sleep_hours": [8,7,6,6,5,5,4,4],
    "attendance":[1,0,1,0,1,0,1,0],
    "pass": [0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Step 2: Features & Target
X = df[["hours_studied", "sleep_hours","attendance"]]
y = df["pass"]

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 4: Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Predict
print("Model expects features:", X.columns.tolist())
prediction = model.predict([[5,6,1]])#X ki values as a test case

print("Prediction (Pass=1, Fail=0):", prediction)