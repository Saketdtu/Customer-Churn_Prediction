import pandas as pd
import numpy as np
import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_preprocessing import DataPreprocessor
from src.customer_segmentation import CustomerSegmentation
from src.churn_prediction import ChurnPredictor
from src.business_insights import BusinessInsights


def main():
    """
    Main function to run the customer churn prediction project
    """
    print("=" * 60)
    print("CUSTOMER CHURN PREDICTION PROJECT")
    print("=" * 60)

    # Initialize data preprocessor
    preprocessor = DataPreprocessor()

    # Step 1: Load and explore data
    print("\nStep 1: Loading and exploring data...")

    # For this example, we'll create synthetic data
    # In a real project, you would load your own data
    print("Creating synthetic data for demonstration...")

    # Create synthetic data
    np.random.seed(42)
    n_samples = 10000

    synthetic_data = {
        'CustomerID': range(1, n_samples + 1),
        'CreditScore': np.random.randint(350, 850, n_samples),
        'Age': np.random.randint(18, 80, n_samples),
        'Tenure': np.random.randint(0, 10, n_samples),
        'Balance': np.random.uniform(0, 200000, n_samples),
        'NumOfProducts': np.random.randint(1, 4, n_samples),
        'HasCrCard': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'IsActiveMember': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'EstimatedSalary': np.random.uniform(20000, 200000, n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }

    df = pd.DataFrame(synthetic_data)

    # Save raw data
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/customer_data.csv', index=False)
    print("Synthetic data saved to data/raw/customer_data.csv")

    # Explore data
    exploration = preprocessor.explore_data(df)

    # Step 2: Clean and preprocess data
    print("\nStep 2: Cleaning and preprocessing data...")

    cleaned_df = preprocessor.clean_data(df)
    encoded_df = preprocessor.encode_categorical(cleaned_df)

    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    encoded_df.to_csv('data/processed/processed_data.csv', index=False)
    print("Processed data saved to data/processed/processed_data.csv")

    # Step 3: Customer segmentation
    print("\nStep 3: Customer segmentation...")

    # Prepare data for clustering (exclude target variable)
    clustering_data = encoded_df.drop(columns=['Churn'])

    # Initialize customer segmentation
    segmenter = CustomerSegmentation()

    # Find optimal number of clusters
    inertias, silhouette_scores = segmenter.find_optimal_clusters(clustering_data, max_clusters=8)

    # Fit clustering model
    clusters = segmenter.fit_predict(clustering_data, n_clusters=5)

    # Perform PCA for visualization
    pca_data, explained_variance = segmenter.perform_pca(clustering_data, n_components=2)

    # Visualize clusters
    segmenter.visualize_clusters(clustering_data, clusters, pca_data)

    # Analyze clusters
    cluster_analysis = segmenter.analyze_clusters(clustering_data, clusters)

    # Save clustering model
    segmenter.save_model('models/cluster_model.pkl')

    # Step 4: Churn prediction
    print("\nStep 4: Churn prediction...")

    # Split data for churn prediction
    X_train, X_test, y_train, y_test = preprocessor.split_data(encoded_df, 'Churn')

    # Initialize churn predictor
    churn_predictor = ChurnPredictor()

    # Train models
    model_results = churn_predictor.train_models(X_train, y_train)

    # Evaluate best model
    evaluation_metrics = churn_predictor.evaluate_model(X_test, y_test)

    # Get feature importance
    feature_importance = churn_predictor.get_feature_importance(X_train)

    # Optimize model
    optimization_results = churn_predictor.optimize_model(X_train, y_train)

    # Save the model
    churn_predictor.save_model('models/churn_model.pkl')

    # Step 5: Business insights
    print("\nStep 5: Generating business insights...")

    # Make predictions on the entire dataset
    churn_predictions, churn_probabilities = churn_predictor.predict_churn(encoded_df.drop(columns=['Churn']))

    # Initialize business insights
    insights = BusinessInsights(encoded_df, clusters, churn_predictions)

    # Generate cluster insights
    cluster_insights = insights.generate_cluster_insights()

    # Create retention strategies
    strategies = insights.create_retention_strategies()

    # Calculate financial impact
    financial_impact = insights.calculate_financial_impact(strategies)

    # Visualize insights
    insights.visualize_insights(cluster_insights, strategies)

    # Generate recommendation report
    report = insights.generate_recommendation_report(cluster_insights, strategies, financial_impact)

    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nGenerated files:")
    print("- Data: data/raw/customer_data.csv, data/processed/processed_data.csv")
    print("- Models: models/cluster_model.pkl, models/churn_model.pkl")
    print("- Visualizations: visualizations/ (multiple PNG files)")
    print("- Report: docs/business_recommendations.md")

    print("\nKey findings:")
    print(f"- Identified {len(cluster_insights)} customer segments")
    print(f"- Best churn prediction model: {churn_predictor.best_model_name}")
    print(f"- Model accuracy: {evaluation_metrics['accuracy']:.2%}")
    print(f"- Overall churn rate: {encoded_df['Churn'].mean():.1%}")
    print(
        f"- Potential value saved through retention: ₹{sum([fi['value_saved'] for fi in financial_impact.values()]):,.0f}")


if __name__ == "__main__":
    main()