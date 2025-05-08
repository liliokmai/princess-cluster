# streamlit_app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.decomposition import PCA

#Load model and scaler
kmeans = joblib.load("kmeans_princess_model.pkl")
scaler = joblib.load("feature_scaler.pkl")

#Load dataset
df = pd.read_csv("disney_princess_popularity_dataset_300_rows.csv")

#Select online influence features
features = ["InstagramFanPages", "GoogleSearchIndex2024", "TikTokHashtagViewsMillions"]
X_scaled = scaler.transform(df[features])
df["Cluster"] = kmeans.predict(X_scaled)

#PCA for Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]

# Streamlit Interface
st.set_page_config(page_title="Disney Princess Clusters", layout="wide")

st.title("Disney Princess Popularity Clustering Based on Online influance")

# Cluster Summary
st.subheader("Cluster Summary Table")
st.dataframe(df[["PrincessName", "InstagramFanPages", "GoogleSearchIndex2024", "TikTokHashtagViewsMillions", "Cluster"]])

# Cluster Visualization
st.subheader("Cluster Visualization (via PCA)")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster", palette="Set2", s=100, ax=ax)
for i, row in df.iterrows():
    ax.text(row["PC1"] + 0.1, row["PC2"], row["PrincessName"], fontsize=7)
plt.title("Clusters of Disney Princesses by Online Popularity")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
st.pyplot(fig)
