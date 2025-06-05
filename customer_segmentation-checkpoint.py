{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "15487001-0f1f-484d-9cdd-1cd0f5248c86",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Customer Segmentation using K-Means Clustering\n",
    "\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.decomposition import PCA\n",
    "\n",
    "# Load dataset\n",
    "url = \"https://raw.githubusercontent.com/shubham0204/Dataset_Repository/main/mall_customers.csv\"\n",
    "df = pd.read_csv(url)\n",
    "\n",
    "# Preprocess data\n",
    "df.drop('CustomerID', axis=1, inplace=True)\n",
    "df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})\n",
    "features = ['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']\n",
    "X = df[features]\n",
    "\n",
    "# Standardize data\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "# Elbow Method to find optimal k\n",
    "wcss = []\n",
    "for i in range(1, 11):\n",
    "    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)\n",
    "    kmeans.fit(X_scaled)\n",
    "    wcss.append(kmeans.inertia_)\n",
    "\n",
    "plt.plot(range(1, 11), wcss, marker='o')\n",
    "plt.title('Elbow Method')\n",
    "plt.xlabel('Number of Clusters')\n",
    "plt.ylabel('WCSS')\n",
    "plt.show()\n",
    "\n",
    "# Apply KMeans\n",
    "k = 5\n",
    "kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)\n",
    "clusters = kmeans.fit_predict(X_scaled)\n",
    "df['Cluster'] = clusters\n",
    "\n",
    "# PCA for 2D Visualization\n",
    "pca = PCA(n_components=2)\n",
    "reduced = pca.fit_transform(X_scaled)\n",
    "df['PCA1'] = reduced[:, 0]\n",
    "df['PCA2'] = reduced[:, 1]\n",
    "\n",
    "plt.figure(figsize=(8, 6))\n",
    "sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')\n",
    "plt.title('Customer Segments (PCA Visualization)')\n",
    "plt.show()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
