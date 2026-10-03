import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("cleaned_adult_dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Basic information
print("\nFirst 5 rows:")
print(df.head())

print("\nBasic statistics:")
print(df.describe())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum().sum())

# 1. Age Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["age"], bins=30)
plt.xlabel("Age")
plt.ylabel("Number of People")
plt.title("Age Distribution")
plt.tight_layout()
plt.show()

# 2. Hours per Week Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["hours-per-week"], bins=30)
plt.xlabel("Hours per Week")
plt.ylabel("Number of People")
plt.title("Working Hours per Week Distribution")
plt.tight_layout()
plt.show()

# 3. Age vs Hours per Week
plt.figure(figsize=(8, 5))
plt.scatter(
    df["age"],
    df["hours-per-week"],
    alpha=0.3
)
plt.xlabel("Age")
plt.ylabel("Hours per Week")
plt.title("Age vs Hours per Week")
plt.tight_layout()
plt.show()

# 4. Capital Gain Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["capital-gain"], bins=30)
plt.xlabel("Capital Gain")
plt.ylabel("Number of People")
plt.title("Capital Gain Distribution")
plt.tight_layout()
plt.show()

# 5. Boxplot
numeric_columns = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numeric_columns])
plt.title("Boxplot of Numerical Variables")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 6. Correlation Heatmap
correlation = df[numeric_columns].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 7. Education Level Analysis
education_columns = [
    column for column in df.columns
    if column.startswith("education_")
]

education_counts = df[education_columns].sum().sort_values(
    ascending=False
)

print("\nEducation level counts:")
print(education_counts.head(10))

plt.figure(figsize=(10, 6))
education_counts.head(10).plot(kind="bar")
plt.xlabel("Education Level")
plt.ylabel("Number of People")
plt.title("Top Education Levels")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Summary
print("\nAge Summary:")
print("Minimum age:", df["age"].min())
print("Maximum age:", df["age"].max())
print("Average age:", round(df["age"].mean(), 2))

print("\nWorking Hours Summary:")
print("Minimum hours:", df["hours-per-week"].min())
print("Maximum hours:", df["hours-per-week"].max())
print("Average hours:", round(df["hours-per-week"].mean(), 2))

print("\nCapital Gain Summary:")
print("Minimum:", df["capital-gain"].min())
print("Maximum:", df["capital-gain"].max())
print("Average:", round(df["capital-gain"].mean(), 2))

print("\nTask 2 EDA completed successfully!")