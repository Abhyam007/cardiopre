import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data = data.fillna(data.mean(numeric_only=True))
    data = data.drop_duplicates()

    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    z_scores = stats.zscore(data[numeric_cols])
    abs_z_scores = abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    data = data[filtered_entries]

    cat_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
    data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    features = data.drop('cardio', axis=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    data_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    data_scaled['cardio'] = data['cardio'].values

    return data_scaled, scaler

def train_and_save(file_path):
    data, scaler = load_and_preprocess_data(file_path)

    X = data.drop('cardio', axis=1)
    y = data['cardio'].astype(int)

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    joblib.dump(model, 'rf_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved.")

if __name__ == "__main__":
    csv_path = r"C:\Users\abhya\Downloads\cardiorisk\up1.csv"  # Update with your dataset path
    train_and_save(csv_path)
