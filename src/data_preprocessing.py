import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os


class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_data(self, filepath):
        """
        Load data from CSV file

        Parameters:
        filepath (str): Path to the CSV file

        Returns:
        pandas.DataFrame: Loaded data
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def explore_data(self, df):
        """
        Perform initial data exploration

        Parameters:
        df (pandas.DataFrame): Input data

        Returns:
        dict: Data exploration summary
        """
        exploration = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'data_types': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'description': df.describe().to_dict()
        }

        print("Data shape:", exploration['shape'])
        print("\nData types:")
        for col, dtype in exploration['data_types'].items():
            print(f"{col}: {dtype}")

        print("\nMissing values:")
        for col, missing in exploration['missing_values'].items():
            if missing > 0:
                print(f"{col}: {missing}")

        return exploration

    def clean_data(self, df):
        """
        Clean the data by handling missing values and outliers

        Parameters:
        df (pandas.DataFrame): Input data

        Returns:
        pandas.DataFrame: Cleaned data
        """
        # Make a copy to avoid modifying the original data
        cleaned_df = df.copy()

        # Handle missing values
        # For numeric columns, fill with median
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if cleaned_df[col].isnull().sum() > 0:
                median_value = cleaned_df[col].median()
                cleaned_df[col].fillna(median_value, inplace=True)
                print(f"Filled missing values in {col} with median: {median_value}")

        # For categorical columns, fill with mode
        categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if cleaned_df[col].isnull().sum() > 0:
                mode_value = cleaned_df[col].mode()[0]
                cleaned_df[col].fillna(mode_value, inplace=True)
                print(f"Filled missing values in {col} with mode: {mode_value}")

        # Remove outliers using IQR method for numeric columns
        print("\nRemoving outliers...")
        for col in numeric_columns:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Count outliers before removal
            outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"Found {outliers} outliers in {col}")

                # Remove outliers
                cleaned_df = cleaned_df[
                    (cleaned_df[col] >= lower_bound) &
                    (cleaned_df[col] <= upper_bound)
                    ]

        print(f"Data shape after cleaning: {cleaned_df.shape}")
        return cleaned_df

    def encode_categorical(self, df):
        """
        Encode categorical variables

        Parameters:
        df (pandas.DataFrame): Input data

        Returns:
        pandas.DataFrame: Data with encoded categorical variables
        """
        encoded_df = df.copy()
        categorical_columns = encoded_df.select_dtypes(include=['object']).columns

        print("Encoding categorical variables...")
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()

            encoded_df[col] = self.label_encoders[col].fit_transform(encoded_df[col])
            print(f"Encoded {col}")

        return encoded_df

    def scale_features(self, df):
        """
        Scale numerical features

        Parameters:
        df (pandas.DataFrame): Input data

        Returns:
        pandas.DataFrame: Data with scaled features
        """
        scaled_df = df.copy()
        numeric_columns = scaled_df.select_dtypes(include=[np.number]).columns

        print("Scaling numerical features...")
        scaled_df[numeric_columns] = self.scaler.fit_transform(scaled_df[numeric_columns])
        print("Features scaled successfully")

        return scaled_df

    def split_data(self, df, target_column, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets

        Parameters:
        df (pandas.DataFrame): Input data
        target_column (str): Name of the target column
        test_size (float): Proportion of data for testing
        random_state (int): Random seed for reproducibility

        Returns:
        tuple: (X_train, X_test, y_train, y_test)
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        print(f"Data split into training ({X_train.shape[0]} samples) and testing ({X_test.shape[0]} samples)")

        return X_train, X_test, y_train, y_test

    def save_processed_data(self, df, filepath):
        """
        Save processed data to CSV

        Parameters:
        df (pandas.DataFrame): Data to save
        filepath (str): Path to save the data
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

    def load_processed_data(self, filepath):
        """
        Load processed data from CSV

        Parameters:
        filepath (str): Path to the CSV file

        Returns:
        pandas.DataFrame: Loaded data
        """
        try:
            df = pd.read_csv(filepath)
            print(f"Processed data loaded from {filepath}")
            return df
        except Exception as e:
            print(f"Error loading processed data: {e}")
            return None