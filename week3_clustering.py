import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# --------------------------------------------------
# 1. Load the cleaned dataset
# --------------------------------------------------

df = pd.read_csv("cleaned_adult_dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 2. Select numerical features for clustering
# --------------------------------------------------

features = [
    "age",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

X = df[features].copy()

print("\nSelected features:")
print(X.head())


# --------------------------------------------------
# 3. Check missing values
# --------------------------------------------------

print("\nMissing values:")
print(X.isnull().sum())


# --------------------------------------------------
# 4. Standardize the data
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nData standardization completed.")


# --------------------------------------------------
# 5. Find a suitable number of clusters
# --------------------------------------------------

inertia = []
silhouette_scores = []

K_values = range(2, 8)

for k in K_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertia.append(model.inertia_)

    score = silhouette_score(X_scaled, labels)

    silhouette_scores.append(score)


# --------------------------------------------------
# 6. Elbow Method
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(K_values, inertia, marker="o")

plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Selecting Number of Clusters")

plt.grid(True)
plt.show()


# --------------------------------------------------
# 7. Silhouette Score
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    K_values,
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different Cluster Numbers")

plt.grid(True)
plt.show()


# --------------------------------------------------
# 8. Apply K-Means Clustering
# --------------------------------------------------

# We use 4 clusters based on the evaluation plots

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


print("\nK-Means clustering completed!")


# --------------------------------------------------
# 9. Cluster counts
# --------------------------------------------------

print("\nNumber of records in each cluster:")

print(df["Cluster"].value_counts().sort_index())


# --------------------------------------------------
# 10. Cluster characteristics
# --------------------------------------------------

cluster_summary = df.groupby("Cluster")[features].mean()

print("\nCluster characteristics:")
print(cluster_summary)


# --------------------------------------------------
# 11. Visualize clusters
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="age",
    y="hours-per-week",
    hue="Cluster",
    palette="Set1",
    s=50
)

plt.title("K-Means Clusters: Age vs Working Hours")
plt.xlabel("Age")
plt.ylabel("Hours per Week")

plt.legend(title="Cluster")

plt.show()


# --------------------------------------------------
# 12. Cluster visualization using Education
# --------------------------------------------------

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="education-num",
    y="hours-per-week",
    hue="Cluster",
    palette="Set2",
    s=50
)

plt.title("K-Means Clusters: Education vs Working Hours")
plt.xlabel("Education Number")
plt.ylabel("Hours per Week")

plt.legend(title="Cluster")

plt.show()


# --------------------------------------------------
# 13. Save clustered dataset
# --------------------------------------------------

df.to_csv(
    "adult_clustered_dataset.csv",
    index=False
)

print("\nClustered dataset saved successfully!")




final_score = silhouette_score(
    X_scaled,
    df["Cluster"]
)

print("\nFinal Silhouette Score:", round(final_score, 4))

print("\nTask 3 clustering completed successfully!")