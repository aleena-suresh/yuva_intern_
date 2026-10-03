import pandas as pd
import matplotlib.pyplot as plt

print(" python project started successfully!")
print(" Pandas version:", pd.__version__)
from ucimlrepo import fetch_ucirepo
adult = fetch_ucirepo(id=2)
print(adult.data.features.head())

# check the size of the dataset 
print(" Dataset shape:",adult.data.features.shape)

#check the column names
print("\nColumn names:")
print(adult.data.features.columns)

#check the datatypes 
print("\nData types:")
print(adult.data.features.dtypes)

#check missing values
print("\nMissing values:")
print(adult.data.features.isnull().sum())

# Handle missing values using the most frequent value
for column in ['workclass', 'occupation', 'native-country']:
    adult.data.features[column] = adult.data.features[column].fillna(
        adult.data.features[column].mode()[0]
    )

print("\nMissing values after cleaning:")
print(adult.data.features.isnull().sum())

# Check for '?' entries
print("\nQuestion mark entries:")
print((adult.data.features == '?').sum())

# Replace '?' with missing values
adult.data.features = adult.data.features.replace('?', pd.NA)

# Fill the new missing values with the most frequent value
for column in ['workclass', 'occupation', 'native-country']:
    adult.data.features[column] = adult.data.features[column].fillna(
        adult.data.features[column].mode()[0]
    )

# Check again
print("\nMissing values after cleaning '?' entries:")
print(adult.data.features.isnull().sum())

# Check numerical outliers using IQR method

numeric_columns = [
    'age',
    'fnlwgt',
    'education-num',
    'capital-gain',
    'capital-loss',
    'hours-per-week'
]

print("\nOutlier counts:")

for column in numeric_columns:
    Q1 = adult.data.features[column].quantile(0.25)
    Q3 = adult.data.features[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = (
        (adult.data.features[column] < lower_limit) |
        (adult.data.features[column] > upper_limit)
    ).sum()

    print(column, ":", outliers)

    # Check for duplicate rows
duplicates = adult.data.features.duplicated().sum()

print("\nDuplicate rows:", duplicates)

adult.data.features = adult.data.features.drop_duplicates()

print("Shape after removing duplicates:",
      adult.data.features.shape)

# Convert categorical columns into numerical columns

categorical_columns = adult.data.features.select_dtypes(
    include=['object']
).columns

adult.data.features = pd.get_dummies(
    adult.data.features,
    columns=categorical_columns,
    dtype=int
)

print("Shape after encoding:",
      adult.data.features.shape)

print("\nFirst 5 rows after encoding:")
print(adult.data.features.head())

adult.data.features.to_csv(
    "cleaned_adult_dataset.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

plt.figure(figsize=(8,5))
plt.hist(adult.data.features['age'], bins=20)
plt.xlabel("Age")
plt.ylabel("Number of People")
plt.title("Age Distribution")
plt.show()
plt.figure(figsize=(8,5))
plt.scatter(
    adult.data.features['age'],
    adult.data.features['hours-per-week'],
    alpha=0.3
)

plt.xlabel("Age")
plt.ylabel("Hours per Week")
plt.title("Age vs Hours per Week")
plt.show()
plt.figure(figsize=(8,5))
plt.hist(adult.data.features['capital-gain'], bins=30)

plt.xlabel("Capital Gain")
plt.ylabel("Number of People")
plt.title("Capital Gain Distribution")
plt.show()

plt.figure(figsize=(8,5))

plt.hist(
    adult.data.features['hours-per-week'],
    bins=30
)

plt.xlabel("Hours per Week")
plt.ylabel("Number of People")
plt.title("Hours per Week Distribution")

plt.show()

categorical_columns = adult.data.features.select_dtypes(
    include=['object']
).columns

adult.data.features = pd.get_dummies(
    adult.data.features,
    columns=categorical_columns,
    dtype=int
)

print("\nShape after encoding:")
print(adult.data.features.shape)

print("\nFirst 5 rows after encoding:")
print(adult.data.features.head())

adult.data.features.to_csv(
    "cleaned_adult_dataset.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")


print("\n================================")
print("WEEK 1 TASK COMPLETED!")
print("================================")

print("Final dataset shape:",
      adult.data.features.shape)

print("\nFiles created:")
print("cleaned_adult_dataset.csv")

print("\nAll preprocessing steps completed successfully!")