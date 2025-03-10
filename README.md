# Enhanced-Backend-with-AI

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime

class RouteDeviationPredictor:
    MODEL_PATH = "route_deviation_model.pkl"
    SCALER_PATH = "scaler.pkl"

    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()

    def load_model(self):
        """Load or train the model if it doesn't exist."""
        if os.path.exists(self.MODEL_PATH) and os.path.exists(self.SCALER_PATH):
            with open(self.MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.SCALER_PATH, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.train_model()

    def train_model(self):
        """Train the model with sample data."""
        # Sample historical data (in practice, this would come from your database)
        data = pd.read_csv("data/training_data.csv")  # Format: lat, lon, speed, hour, deviation (0/1)
        X = data[['latitude', 'longitude', 'speed', 'hour']]
        y = data['deviation']

        # Split and scale data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train Random Forest
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # Save model and scaler
        with open(self.MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.SCALER_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)

    def predict_deviation(self, latitude, longitude, speed):
        """Predict if the vehicle will deviate from its route."""
        hour = datetime.utcnow().hour
        features = np.array([[latitude, longitude, speed, hour]])
        features_scaled = self.scaler.transform(features)
        probability = self.model.predict_proba(features_scaled)[0][1]  # Probability of deviation
        return probability > 0.7  # Threshold for alert

# Initialize predictor
predictor = RouteDeviationPredictor()
