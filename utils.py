import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import joblib



def encode_features(x):
    """
    Realiza ingeniería de características en el DataFrame dado.
    """
    scaler = joblib.load("scalers/scaler.pkl")

    df = pd.read_csv("data/idealista_madrid.csv")

    df = df[['baths', 'rooms', 'address', 'sqft', 'typology', 'price']]

    df['neighborhood'] = df['address'].str.split(',').str[0]
    df = df.drop(columns=['address'])

    # Crear una copia del DataFrame original
    features = x.copy()

    features["sqft"] = features["sqft"].astype(float)
    features["rooms"] = features["rooms"].astype(int)
    features["baths"] = features["baths"].astype(int)

    # 3. Space per room
    features['sqft_per_room'] = features['sqft'] / features['rooms']

    features['sqft_per_bath'] = features['sqft'] / features['baths']

    features["bath_to_room_ratio"] = features["baths"] / features["rooms"]

    features["sqft_rooms_interaction"] = features["sqft"] * features["rooms"]

    features["luxury_score"] = (features["sqft"] * features["bath_to_room_ratio"]) / features["rooms"]

    # 4. Total rooms (bathrooms + bedrooms)
    features['total_rooms'] = features['rooms'] + features['baths']

    # 5. Is luxury property (based on percentiles)

    size_threshold = np.percentile(df['sqft'], 75)
    features['is_luxury'] = ((features['sqft'] > size_threshold)).astype(int)

    # 6. Neighborhood statistics
    neighborhood_avg = df.groupby('neighborhood')['price'].mean().to_dict()
    neighborhood_median = df.groupby('neighborhood')['price'].median().to_dict()

    features['neighborhood_price_avg'] = df['neighborhood'].map(neighborhood_avg)
    features["neighborhood_price_median"] = df["neighborhood"].map(neighborhood_median)

    features["avg_sqft_neighborhood"] = df["neighborhood"].map(df.groupby("neighborhood")["sqft"].mean())
    features["avg_rooms_neighborhood"] = df["neighborhood"].map(df.groupby("neighborhood")["rooms"].mean())

    features["neighborhood_density"] = features["neighborhood_price_median"] / features["sqft"]
    # 9. Room density (rooms per 100 sqft)
    features['room_density'] = (features['rooms'] / features['sqft']) * 100

    features["log_sqft"] = np.log1p(features["sqft"])

    features["log_neighborhood_price_avg"] = np.log1p(features["neighborhood_price_avg"])

    features["log_neighborhood_price_median"] = np.log1p(features["neighborhood_price_median"])

    features["typology"] = features["typology"] == "Pisos"

    features = features.drop(columns=["neighborhood"])

    features = scale_features(features, scaler)

    return features



def scale_features(x, scaler):
    features = x.copy()

    columns_to_normalize = [
        # Original numerical features
        'rooms', 'baths', 'sqft',
        
        # Price-related features
        'neighborhood_price_avg', "neighborhood_price_median", "luxury_score", "neighborhood_density",
        
        # Ratio features
        'sqft_per_room', 'room_density', "sqft_per_bath", "sqft_rooms_interaction", "bath_to_room_ratio",

        # log values
        'log_sqft', "log_neighborhood_price_avg", "log_neighborhood_price_median",

        "avg_sqft_neighborhood", "avg_rooms_neighborhood",
        
        # Count features
        'total_rooms'
    ]

    features[columns_to_normalize] = scaler.transform(features[columns_to_normalize])

    return features
