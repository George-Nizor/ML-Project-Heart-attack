import pandas as pd
from sklearn.model_selection import train_test_split

class DataGenerator:
    def __init__(self, file_path, target_column=None, drop_columns=None):
        """
        Initializes the DataGenerator.
        
        :param file_path: Path to the CSV file.
        :param target_column: The name of the target column to separate.
        :param drop_columns: List of columns to drop from the data.
        """
        self.file_path = file_path
        self.target_column = target_column
        self.drop_columns = drop_columns or []
        self.data = None
        self.X = None
        self.y = None

    def load_data(self):
        """Loads data from CSV, dropping missing values and duplicates."""
        self.data = pd.read_csv(self.file_path)
        self.data.dropna(inplace=True)
        self.data.drop_duplicates(inplace=True)
        self.data.reset_index(drop=True, inplace=True)
        return self.data

    def preprocess(self):
        """Performs cleaning and feature engineering on the data."""
        # Drop specified columns
        if self.drop_columns:
            self.data = self.data.drop(columns=self.drop_columns)
        # For example: Convert categorical variables using one-hot encoding if needed
        if self.target_column is None:
            self.X = pd.get_dummies(self.data, drop_first=True)
        else:
            self.y = self.data[self.target_column]
            self.X = self.data.drop(columns=[self.target_column])
            self.X = pd.get_dummies(self.X, drop_first=True)
        return self.X, self.y

    def split_data(self, test_size=0.2, random_state=42):
        """Splits the data into training and testing sets."""
        if self.X is None or (self.target_column and self.y is None):
            raise ValueError("Data has not been preprocessed. Call preprocess() first.")
        
        if self.target_column:
            return train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
        else:
            return train_test_split(self.X, test_size=test_size, random_state=random_state)

# Example usage:
if __name__ == "__main__":
    # Initialize with parameters
    generator = DataGenerator(
        file_path='data/heart_attack_prediction_indonesia.csv', 
        target_column='heart_attack', 
        drop_columns=['age']  # Add any columns you want to drop
    )
    
    # Load, preprocess, and split the data
    generator.load_data()
    generator.preprocess()
    X_train, X_test, y_train, y_test = generator.split_data()
    
    print("Data loaded and split successfully!")