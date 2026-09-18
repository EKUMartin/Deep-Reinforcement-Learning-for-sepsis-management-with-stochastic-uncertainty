 # HMM 
import numpy as np
import joblib
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

class SepsisHMM:
    def __init__(self, n_components=4, random_state=42):
        self.n_components = n_components
        self.model = hmm.GaussianHMM(n_components=self.n_components, covariance_type="full", random_state=random_state)
        self.scaler = StandardScaler()

    def train(self, df, feature_cols):
        X_raw = df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X_raw)
        
        lengths = df.groupby('stay_id').size().values
        
        print("Training HMM")
        self.model.fit(X_scaled, lengths)
        print("HMM Training Complete.")
        return self.model

    def predict(self, df, feature_cols):
        X_scaled = self.scaler.transform(df[feature_cols].values)
        lengths = df.groupby('stay_id').size().values
        states = self.model.predict(X_scaled, lengths)
        return states

    def save_model(self, model_path='hmm_model.pkl', scaler_path='hmm_scaler.pkl'):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print("HMM Model and Scaler saved.")