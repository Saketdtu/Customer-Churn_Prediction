import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
import os


class ChurnPredictor:
    def __init__(self):
        self.models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None

    def train_models(self, X_train, y_train):
        """
        Train multiple models and compare performance

        Parameters:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target

        Returns:
        dict: Model performance results
        """
        results = {}

        print("Training models...")

        for name, model in self.models.items():
            print(f"\nTraining {name}...")

            # Scale features for models that need it
            if name in ['LogisticRegression', 'SVM']:
                X_train_scaled = self.scaler.fit_transform(X_train)
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)

            # Cross-validation
            if name in ['LogisticRegression', 'SVM']:
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

            results[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Select best model
        best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name

        print(f"\nBest model: {best_model_name} with CV accuracy: {results[best_model_name]['cv_mean']:.4f}")

        return results

    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the best model on test set

        Parameters:
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test target

        Returns:
        dict: Evaluation metrics
        """
        if self.best_model is None:
            print("No model trained. Please train models first.")
            return None

        print(f"\nEvaluating {self.best_model_name}...")

        # Scale features if needed
        if self.best_model_name in ['LogisticRegression', 'SVM']:
            X_test_scaled = self.scaler.transform(X_test)
            y_pred = self.best_model.predict(X_test_scaled)
            y_pred_proba = self.best_model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_pred = self.best_model.predict(X_test)
            y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }

        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Not Churn', 'Churn'],
                    yticklabels=['Not Churn', 'Churn'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')

        os.makedirs('visualizations', exist_ok=True)
        plt.savefig('visualizations/confusion_matrix.png')
        plt.show()

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Churn', 'Churn']))

        return metrics

    def get_feature_importance(self, X):
        """
        Get feature importance from the best model

        Parameters:
        X (pandas.DataFrame): Feature data

        Returns:
        pandas.DataFrame: Feature importance
        """
        if self.best_model is None:
            print("No model trained. Please train models first.")
            return None

        # Check if model has feature_importances_ attribute
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_

            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': importance
            }).sort_values('importance', ascending=False)

            self.feature_importance = feature_importance

            # Plot feature importance
            plt.figure(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
            plt.title('Feature Importance')
            plt.tight_layout()

            os.makedirs('visualizations', exist_ok=True)
            plt.savefig('visualizations/feature_importance.png')
            plt.show()

            return feature_importance
        else:
            print(f"{self.best_model_name} does not have feature_importances_ attribute")
            return None

    def optimize_model(self, X_train, y_train):
        """
        Optimize the best model using grid search

        Parameters:
        X_train (pandas.DataFrame): Training features
        y_train (pandas.Series): Training target

        Returns:
        dict: Best parameters and score
        """
        if self.best_model is None:
            print("No model trained. Please train models first.")
            return None

        print(f"\nOptimizing {self.best_model_name}...")

        # Define parameter grids for different models
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            },
            'SVM': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        }

        # Scale features if needed
        if self.best_model_name in ['LogisticRegression', 'SVM']:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X = X_train_scaled
        else:
            X = X_train

        # Perform grid search
        param_grid = param_grids[self.best_model_name]
        grid_search = GridSearchCV(
            self.best_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search.fit(X, y_train)

        # Update the best model
        self.best_model = grid_search.best_estimator_

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }

    def predict_churn(self, X):
        """
        Predict churn for new data

        Parameters:
        X (pandas.DataFrame): New data

        Returns:
        tuple: (predictions, probabilities)
        """
        if self.best_model is None:
            print("No model trained. Please train models first.")
            return None, None

        # Scale features if needed
        if self.best_model_name in ['LogisticRegression', 'SVM']:
            X_scaled = self.scaler.transform(X)
            predictions = self.best_model.predict(X_scaled)
            probabilities = self.best_model.predict_proba(X_scaled)[:, 1]
        else:
            predictions = self.best_model.predict(X)
            probabilities = self.best_model.predict_proba(X)[:, 1]

        return predictions, probabilities

    def save_model(self, filepath):
        """
        Save the trained model

        Parameters:
        filepath (str): Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance
        }

        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a trained model

        Parameters:
        filepath (str): Path to the saved model
        """
        try:
            model_data = joblib.load(filepath)
            self.best_model = model_data['best_model']
            self.best_model_name = model_data['best_model_name']
            self.scaler = model_data['scaler']
            self.feature_importance = model_data['feature_importance']
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")