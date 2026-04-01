import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


data = {
    "size": [500, 800, 1200, 1500, 2000, 2200, 2500],
    "rooms": [1, 2, 3, 3, 4, 4, 5],
    "location_score": [3, 5, 7, 8, 9, 6, 10],
    # "price": [10000, 15000, 22000, 26000, 32000, 30000, 40000]
    "price": [10000, 15000, 25000, 30000, 50000, 70000, 100000]
}

df = pd.DataFrame(data)

# print(df)


X = df[["size", "rooms", "location_score"]]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)#randomness for same testing set


tree = DecisionTreeRegressor()
tree.fit(X_train, y_train)
prediction = tree.predict(X_test)


model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("X_test:\n", X_test)


print("Linear Regression Predicted price:", predictions)
print("Tree Prediction:", prediction)

print("Error:", mean_absolute_error(y_test, predictions))
print("Error:", mean_absolute_error(y_test, prediction))