import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    silhouette_score
)
from sklearn.decomposition import PCA

print("Loading dataset...")

adult = fetch_ucirepo(id=2)

df = adult.data.features.copy()
target = adult.data.targets.copy()

target.columns = ["income"]
df["income"] = target["income"].values

print("Dataset shape:", df.shape)
print("Dataset loaded successfully.")

df = df.replace("?", np.nan)

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:
    df[column] = df[column].fillna(
        df[column].mode()[0]
    )

print("\nMissing values handled.")

numeric_columns = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

X_cluster = df[numeric_columns].copy()

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X_cluster
)

print("Data preprocessing completed.")

inertia = []

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(
        kmeans.inertia_
    )

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 9),
    inertia,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)
plt.tight_layout()
plt.show()

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(
    X_scaled
)

sample_size = min(
    3000,
    len(X_scaled)
)

sample_indices = np.random.RandomState(
    42
).choice(
    len(X_scaled),
    sample_size,
    replace=False
)

silhouette = silhouette_score(
    X_scaled[sample_indices],
    df["Cluster"].iloc[sample_indices]
)

print(
    "\nSilhouette Score:",
    round(silhouette, 4)
)

cluster_summary = df.groupby(
    "Cluster"
)[numeric_columns].mean()

print("\nCluster Summary:")
print(cluster_summary)

cluster_summary.to_csv(
    "week6_cluster_summary.csv"
)

pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=df["Cluster"],
    palette="viridis"
)

plt.title("PCA Cluster Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

sns.countplot(
    x=df["Cluster"]
)

plt.title("Cluster Distribution")
plt.xlabel("Cluster")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

print("\nStarting supervised learning...")

df["Income_Label"] = df["income"].astype(
    str
).apply(
    lambda x: 1 if ">50K" in x else 0
)

features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

X = df[features]
y = df["Income_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

scaler_model = StandardScaler()

X_train_scaled = scaler_model.fit_transform(
    X_train
)

X_test_scaled = scaler_model.transform(
    X_test
)

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)

y_pred = model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\nSupervised Learning Results")
print("---------------------------")

print(
    "Accuracy:",
    round(accuracy, 4)
)

print(
    "Precision:",
    round(precision, 4)
)

print(
    "Recall:",
    round(recall, 4)
)

print(
    "F1 Score:",
    round(f1, 4)
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title(
    "Income Prediction Confusion Matrix"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.show()

model_results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

model_results.to_csv(
    "week6_model_results.csv",
    index=False
)

df.to_csv(
    "week6_capstone_dataset.csv",
    index=False
)

print("\nOutput files saved successfully!")

print("\n==========================================")
print("Week 6 Capstone Project completed successfully!")
print("==========================================")