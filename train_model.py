import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

def preprocess_data(df):
    # Create a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Convert protocol names to numerical values
    if 'protocol' in df_processed.columns:
        protocol_mapping = {
            'TCP': 6,
            'UDP': 17,
            'ICMP': 1,
            'HTTP': 80,
            'HTTPS': 443,
            'DNS': 53,
            'FTP': 21,
            'SSH': 22,
            'SMTP': 25
        }
        df_processed['protocol'] = df_processed['protocol'].map(protocol_mapping)
        # Fill any remaining NaN values with 0 (unknown protocol)
        df_processed['protocol'] = df_processed['protocol'].fillna(0)
    
    # Convert any remaining categorical columns to numerical
    for column in df_processed.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_processed[column] = le.fit_transform(df_processed[column].astype(str))
    
    return df_processed

def train_anomaly_detector(csv_file_path):
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(csv_file_path)
    
    # Select the features
    features = [
        'src_port', 'dst_port', 'protocol', 'flow_duration', 'pkt_count',
        'pkt_size_avg', 'bytes_sent', 'bytes_received', 'conn_count_last_10s',
        'same_dst_count', 'srv_serror_rate', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'entropy', 'honeypot_flag'
    ]
    
    # Preprocess the data
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    # Prepare the data
    X = df_processed[features]
    
    # Initialize and fit the scaler
    print("Fitting scaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train the Isolation Forest model
    print("Training model...")
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.1,  # Adjust based on your expected anomaly rate
        random_state=42
    )
    model.fit(X_scaled)
    
    # Save the model and scaler
    print("Saving model and scaler...")
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("Training completed successfully!")
    return model, scaler

if __name__ == "__main__":
    # Replace 'your_dataset.csv' with your actual dataset file name
    train_anomaly_detector('cybersecurity_anomaly_dataset_2000.csv')