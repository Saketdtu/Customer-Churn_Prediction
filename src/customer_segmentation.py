import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os


class CustomerSegmentation:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.pca = None
        self.cluster_centers = None

    def find_optimal_clusters(self, data, max_clusters=10):
        """
        Find the optimal number of clusters using elbow method and silhouette score

        Parameters:
        data (pandas.DataFrame): Input data
        max_clusters (int): Maximum number of clusters to test

        Returns:
        tuple: (inertias, silhouette_scores)
        """
        inertias = []
        silhouette_scores = []

        print("Finding optimal number of clusters...")

        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(data)

            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))

            print(
                f"Clusters: {k}, Inertia: {kmeans.inertia_:.2f}, Silhouette Score: {silhouette_score(data, kmeans.labels_):.4f}")

        # Plot elbow curve and silhouette scores
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(2, max_clusters + 1), inertias, 'bo-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method')

        plt.subplot(1, 2, 2)
        plt.plot(range(2, max_clusters + 1), silhouette_scores, 'ro-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score')

        plt.tight_layout()

        # Save the plot
        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/cluster_analysis.png')
        plt.show()

        return inertias, silhouette_scores

    def fit_predict(self, data, n_clusters=None):
        """
        Fit K-means and predict clusters

        Parameters:
        data (pandas.DataFrame): Input data
        n_clusters (int): Number of clusters (uses self.n_clusters if None)

        Returns:
        numpy.ndarray: Cluster labels
        """
        if n_clusters is None:
            n_clusters = self.n_clusters

        print(f"Fitting K-means with {n_clusters} clusters...")

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = self.kmeans.fit_predict(data)

        self.cluster_centers = self.kmeans.cluster_centers_

        print("Clustering completed")
        return clusters

    def perform_pca(self, data, n_components=2):
        """
        Perform PCA for dimensionality reduction

        Parameters:
        data (pandas.DataFrame): Input data
        n_components (int): Number of components

        Returns:
        tuple: (pca_data, explained_variance_ratio)
        """
        print(f"Performing PCA with {n_components} components...")

        self.pca = PCA(n_components=n_components)
        pca_data = self.pca.fit_transform(data)

        explained_variance_ratio = self.pca.explained_variance_ratio_
        print(f"Explained variance ratio: {explained_variance_ratio}")

        return pca_data, explained_variance_ratio

    def visualize_clusters(self, data, clusters, pca_data=None):
        """
        Visualize clusters

        Parameters:
        data (pandas.DataFrame): Original data
        clusters (numpy.ndarray): Cluster labels
        pca_data (numpy.ndarray): PCA-transformed data (optional)
        """
        # Create a copy of the data with cluster labels
        clustered_data = data.copy()
        clustered_data['Cluster'] = clusters

        # If PCA data is provided, use it for visualization
        if pca_data is not None:
            # Create a DataFrame with PCA components
            pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = clusters

            # Plot PCA components
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=50, alpha=0.7)
            plt.title('Customer Segments (PCA)')

            # Plot cluster centers
            if self.cluster_centers is not None and self.pca is not None:
                centers_pca = self.pca.transform(self.cluster_centers)
                plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, marker='X', label='Centroids')
                plt.legend()

            plt.tight_layout()
            plt.savefig('visualizations/customer_segments_pca.png')
            plt.show()

        # Visualize clusters using key features
        # Select a few important features for visualization
        numeric_columns = data.select_dtypes(include=[np.number]).columns

        if len(numeric_columns) >= 2:
            # Create a pairplot for selected features
            selected_features = numeric_columns[:min(4, len(numeric_columns))].tolist()
            selected_features.append('Cluster')

            plt.figure(figsize=(12, 10))
            sns.pairplot(clustered_data[selected_features], hue='Cluster', palette='viridis')
            plt.suptitle('Customer Segments - Feature Relationships', y=1.02)
            plt.tight_layout()
            plt.savefig('visualizations/customer_segments_pairplot.png')
            plt.show()

    def analyze_clusters(self, data, clusters):
        """
        Analyze cluster characteristics

        Parameters:
        data (pandas.DataFrame): Original data
        clusters (numpy.ndarray): Cluster labels

        Returns:
        pandas.DataFrame: Cluster analysis
        """
        # Create a copy of the data with cluster labels
        clustered_data = data.copy()
        clustered_data['Cluster'] = clusters

        # Calculate cluster statistics
        cluster_analysis = clustered_data.groupby('Cluster').agg(['mean', 'std'])

        # Calculate cluster sizes
        cluster_sizes = clustered_data['Cluster'].value_counts().sort_index()

        print("\nCluster Analysis:")
        print("=" * 50)

        for cluster in sorted(clustered_data['Cluster'].unique()):
            print(f"\nCluster {cluster}:")
            print(
                f"Size: {cluster_sizes[cluster]} customers ({cluster_sizes[cluster] / len(clustered_data) * 100:.1f}%)")

            # Get the cluster data
            cluster_data = clustered_data[clustered_data['Cluster'] == cluster]

            # Calculate key metrics
            numeric_columns = cluster_data.select_dtypes(include=[np.number]).columns

            print("\nKey characteristics:")
            for col in numeric_columns[:5]:  # Show first 5 numeric columns
                mean_val = cluster_data[col].mean()
                overall_mean = clustered_data[col].mean()

                if mean_val > overall_mean * 1.1:
                    comparison = "Higher than average"
                elif mean_val < overall_mean * 0.9:
                    comparison = "Lower than average"
                else:
                    comparison = "About average"

                print(f"- {col}: {mean_val:.2f} ({comparison})")

        return cluster_analysis

    def save_model(self, filepath):
        """
        Save the trained model

        Parameters:
        filepath (str): Path to save the model
        """
        import joblib

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'kmeans': self.kmeans,
            'pca': self.pca,
            'n_clusters': self.n_clusters,
            'cluster_centers': self.cluster_centers
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a trained model

        Parameters:
        filepath (str): Path to the saved model
        """
        import joblib

        try:
            model_data = joblib.load(filepath)
            self.kmeans = model_data['kmeans']
            self.pca = model_data['pca']
            self.n_clusters = model_data['n_clusters']
            self.cluster_centers = model_data['cluster_centers']
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")

    def predict_cluster(self, data):
        """
        Predict cluster for new data

        Parameters:
        data (pandas.DataFrame): New data

        Returns:
        numpy.ndarray: Predicted cluster labels
        """
        if self.kmeans is None:
            print("Model not trained. Please train the model first.")
            return None

        return self.kmeans.predict(data)