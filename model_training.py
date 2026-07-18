import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load processed dataset
df = pd.read_csv("processed_house_prices.csv")

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "house_price_model.pkl")

print("Model Training Completed Successfully!")