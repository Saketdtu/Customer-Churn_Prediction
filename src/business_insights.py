import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import os


class BusinessInsights:
    def __init__(self, data, clusters, churn_predictions):
        """
        Initialize BusinessInsights with data, clusters, and churn predictions

        Parameters:
        data (pandas.DataFrame): Original data
        clusters (numpy.ndarray): Cluster labels
        churn_predictions (numpy.ndarray): Churn predictions
        """
        self.data = data.copy()
        self.data['Cluster'] = clusters
        self.data['Churn_Prediction'] = churn_predictions

    def generate_cluster_insights(self):
        """
        Generate business insights for each customer segment

        Returns:
        dict: Cluster insights
        """
        insights = {}

        print("\nGenerating cluster insights...")

        for cluster in sorted(self.data['Cluster'].unique()):
            cluster_data = self.data[self.data['Cluster'] == cluster]

            # Calculate churn rate for cluster
            churn_rate = cluster_data['Churn_Prediction'].mean()

            # Analyze key characteristics
            numeric_columns = cluster_data.select_dtypes(include=[np.number]).columns

            characteristics = {}
            for col in ['Balance', 'CreditScore', 'Age', 'EstimatedSalary']:
                if col in numeric_columns:
                    characteristics[col] = {
                        'mean': cluster_data[col].mean(),
                        'median': cluster_data[col].median(),
                        'std': cluster_data[col].std()
                    }

            # Generate recommendations
            if churn_rate > 0.3:
                recommendation = "High churn risk cluster - Implement targeted retention strategies"
                priority = "High"
            elif churn_rate > 0.15:
                recommendation = "Medium churn risk cluster - Monitor closely and offer incentives"
                priority = "Medium"
            else:
                recommendation = "Low churn risk cluster - Maintain current strategies"
                priority = "Low"

            insights[f'Cluster_{cluster}'] = {
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(self.data) * 100,
                'churn_rate': churn_rate,
                'characteristics': characteristics,
                'recommendation': recommendation,
                'priority': priority
            }

            print(f"\nCluster {cluster}:")
            print(f"Size: {len(cluster_data)} customers ({len(cluster_data) / len(self.data) * 100:.1f}%)")
            print(f"Churn rate: {churn_rate:.2%}")
            print(f"Recommendation: {recommendation}")

        return insights

    def create_retention_strategies(self):
        """
        Create targeted retention strategies

        Returns:
        dict: Retention strategies
        """
        strategies = {}

        print("\nCreating retention strategies...")

        # High-value customers with high churn risk
        high_value_threshold = self.data['Balance'].quantile(0.7)
        high_value_high_risk = self.data[
            (self.data['Balance'] > high_value_threshold) &
            (self.data['Churn_Prediction'] == 1)
            ]

        strategies['high_value_retention'] = {
            'target_customers': len(high_value_high_risk),
            'description': 'High-value customers at risk of churn',
            'strategy': 'Personalized offers, dedicated account manager, premium support',
            'expected_impact': 'Reduce churn by 15-20%',
            'implementation_cost': 'Medium',
            'roi': 'High'
        }

        # Young customers with low engagement
        young_low_engagement = self.data[
            (self.data['Age'] < 30) &
            (self.data['NumOfProducts'] < 2)  # Assuming this column exists
            ]

        strategies['young_engagement'] = {
            'target_customers': len(young_low_engagement),
            'description': 'Young customers with low product engagement',
            'strategy': 'Mobile-first approach, gamification, social features',
            'expected_impact': 'Increase product adoption by 25%',
            'implementation_cost': 'Medium',
            'roi': 'Medium'
        }

        # Long-term customers showing signs of churn
        # Assuming we have a 'Tenure' column
        if 'Tenure' in self.data.columns:
            long_term_threshold = self.data['Tenure'].quantile(0.7)
            long_term_churn_risk = self.data[
                (self.data['Tenure'] > long_term_threshold) &
                (self.data['Churn_Prediction'] == 1)
                ]

            strategies['long_term_retention'] = {
                'target_customers': len(long_term_churn_risk),
                'description': 'Long-term customers at risk of churn',
                'strategy': 'Loyalty rewards, exclusive benefits, personalized communication',
                'expected_impact': 'Reduce churn by 10-15%',
                'implementation_cost': 'Low',
                'roi': 'High'
            }

        # Low activity customers
        # Assuming we have an 'ActivityScore' column
        if 'ActivityScore' in self.data.columns:
            low_activity_threshold = self.data['ActivityScore'].quantile(0.3)
            low_activity_customers = self.data[
                (self.data['ActivityScore'] < low_activity_threshold) &
                (self.data['Churn_Prediction'] == 0)
                ]

            strategies['low_activity'] = {
                'target_customers': len(low_activity_customers),
                'description': 'Low activity customers',
                'strategy': 'Re-engagement campaigns, personalized content, special offers',
                'expected_impact': 'Increase activity by 30%',
                'implementation_cost': 'Low',
                'roi': 'Medium'
            }

        # Print strategies
        for strategy_name, strategy_details in strategies.items():
            print(f"\n{strategy_name.replace('_', ' ').title()}:")
            print(f"Target customers: {strategy_details['target_customers']}")
            print(f"Strategy: {strategy_details['strategy']}")
            print(f"Expected impact: {strategy_details['expected_impact']}")

        return strategies

    def calculate_financial_impact(self, strategies):
        """
        Calculate the financial impact of retention strategies

        Parameters:
        strategies (dict): Retention strategies

        Returns:
            dict: Financial impact analysis
        """
        financial_impact = {}

        # Calculate average customer value
        avg_customer_value = self.data['Balance'].mean() * 0.1  # Assume 10% of balance is annual value

        print("\nCalculating financial impact...")

        for strategy_name, strategy_details in strategies.items():
            # Estimate reduction in churn rate
            if "Reduce churn by" in strategy_details['expected_impact']:
                churn_reduction = float(
                    strategy_details['expected_impact'].split('Reduce churn by ')[1].split('%')[0]) / 100
            elif "Increase" in strategy_details['expected_impact']:
                # For strategies that increase engagement, assume a smaller churn reduction
                churn_reduction = 0.05  # 5% reduction

            # Calculate financial impact
            customers_affected = strategy_details['target_customers']
            current_churn_rate = self.data['Churn_Prediction'].mean()

            customers_saved = customers_affected * current_churn_rate * churn_reduction
            value_saved = customers_saved * avg_customer_value

            financial_impact[strategy_name] = {
                'customers_affected': customers_affected,
                'churn_reduction': churn_reduction,
                'customers_saved': customers_saved,
                'value_saved': value_saved,
                'implementation_cost': strategy_details['implementation_cost'],
                'roi': strategy_details['roi']
            }

            print(f"\n{strategy_name.replace('_', ' ').title()}:")
            print(f"Customers affected: {customers_affected}")
            print(f"Estimated customers saved: {customers_saved:.0f}")
            print(f"Value saved: ₹{value_saved:,.2f}")

        return financial_impact

    def visualize_insights(self, cluster_insights, strategies):
        """
        Create visualizations for business insights

        Parameters:
        cluster_insights (dict): Cluster insights
        strategies (dict): Retention strategies
        """
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Business Insights Dashboard', fontsize=16)

        # Cluster sizes and churn rates
        ax1 = axes[0, 0]
        clusters = list(cluster_insights.keys())
        sizes = [cluster_insights[c]['size'] for c in clusters]
        churn_rates = [cluster_insights[c]['churn_rate'] for c in clusters]

        x = np.arange(len(clusters))
        width = 0.35

        ax1.bar(x - width / 2, sizes, width, label='Cluster Size', alpha=0.7)
        ax1_twin = ax1.twinx()
        ax1_twin.bar(x + width / 2, churn_rates, width, label='Churn Rate', color='orange', alpha=0.7)

        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Size', color='blue')
        ax1_twin.set_ylabel('Churn Rate', color='orange')
        ax1.set_title('Cluster Sizes and Churn Rates')
        ax1.set_xticks(x)
        ax1.set_xticklabels(clusters)

        # Strategy impact
        ax2 = axes[0, 1]
        strategy_names = list(strategies.keys())
        target_customers = [strategies[s]['target_customers'] for s in strategy_names]

        ax2.barh(strategy_names, target_customers, color='green', alpha=0.7)
        ax2.set_xlabel('Number of Customers')
        ax2.set_title('Retention Strategy Target Customers')

        # Customer distribution by cluster and churn
        ax3 = axes[1, 0]
        cluster_churn = pd.crosstab(self.data['Cluster'], self.data['Churn_Prediction'])
        cluster_churn.plot(kind='bar', stacked=True, ax=ax3)
        ax3.set_title('Customer Distribution by Cluster and Churn')
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Number of Customers')
        ax3.legend(['Not Churn', 'Churn'])

        # Feature importance for churn
        ax4 = axes[1, 1]
        # Calculate correlation with churn
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        correlations = self.data[numeric_columns].corrwith(self.data['Churn_Prediction']).sort_values(ascending=False)

        # Plot top 10 features
        top_features = correlations.head(10)
        ax4.barh(top_features.index, top_features.values, color='purple', alpha=0.7)
        ax4.set_title('Top Features Correlated with Churn')
        ax4.set_xlabel('Correlation with Churn')

        plt.tight_layout()

        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/business_insights.png', dpi=300, bbox_inches='tight')
        plt.show()

    def generate_recommendation_report(self, cluster_insights, strategies, financial_impact):
        """
        Generate a comprehensive recommendation report

        Parameters:
        cluster_insights (dict): Cluster insights
        strategies (dict): Retention strategies
        financial_impact (dict): Financial impact analysis

        Returns:
        str: Recommendation report
        """
        report = "# Customer Churn Analysis and Retention Strategy Report\n\n"

        report += "## Executive Summary\n\n"
        report += f"This analysis identifies {len(cluster_insights)} distinct customer segments "
        report += f"with varying churn risks. The overall churn rate is {self.data['Churn_Prediction'].mean():.1%}. "
        report += "Targeted retention strategies could save approximately "
        report += f"₹{sum([fi['value_saved'] for fi in financial_impact.values()]):,.0f} in customer value.\n\n"

        report += "## Customer Segments\n\n"
        for cluster, insights in cluster_insights.items():
            report += f"### {cluster}\n\n"
            report += f"- **Size**: {insights['size']} customers ({insights['percentage']:.1f}%)\n"
            report += f"- **Churn Rate**: {insights['churn_rate']:.1%}\n"
            report += f"- **Priority**: {insights['priority']}\n"
            report += f"- **Recommendation**: {insights['recommendation']}\n\n"

        report += "## Retention Strategies\n\n"
        for strategy, details in strategies.items():
            report += f"### {strategy.replace('_', ' ').title()}\n\n"
            report += f"- **Target Customers**: {details['target_customers']}\n"
            report += f"- **Strategy**: {details['strategy']}\n"
            report += f"- **Expected Impact**: {details['expected_impact']}\n"
            report += f"- **Implementation Cost**: {details['implementation_cost']}\n"
            report += f"- **ROI**: {details['roi']}\n\n"

        report += "## Financial Impact\n\n"
        for strategy, impact in financial_impact.items():
            report += f"### {strategy.replace('_', ' ').title()}\n\n"
            report += f"- **Customers Affected**: {impact['customers_affected']}\n"
            report += f"- **Estimated Customers Saved**: {impact['customers_saved']:.0f}\n"
            report += f"- **Value Saved**: ₹{impact['value_saved']:,.2f}\n\n"

        report += "## Implementation Timeline\n\n"
        report += "1. **Month 1**: Implement high-value retention strategy\n"
        report += "2. **Month 2**: Launch young customer engagement program\n"
        report += "3. **Month 3**: Develop long-term customer loyalty initiatives\n"
        report += "4. **Month 4**: Create low-activity customer re-engagement campaigns\n"
        report += "5. **Month 5-6**: Monitor results and optimize strategies\n\n"

        report += "## Success Metrics\n\n"
        report += "- Reduce overall churn rate by 15% within 6 months\n"
        report += "- Increase customer retention rate by 10% within 6 months\n"
        report += "- Improve customer satisfaction scores by 20% within 6 months\n"
        report += "- Achieve positive ROI on all retention strategies\n\n"

        # Save the report
        os.makedirs('docs', exist_ok=True)
        with open('docs/business_recommendations.md', 'w') as f:
            f.write(report)

        print("Recommendation report saved to docs/business_recommendations.md")
        return report