import pandas as pd
import matplotlib.pyplot as plt

# Load cleaned dataset
df = pd.read_csv("cleaned_house_prices.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print(df.info())

# Statistical summary
print(df.describe())

# Histogram for all numeric columns
df.hist(figsize=(12, 8))
plt.show()