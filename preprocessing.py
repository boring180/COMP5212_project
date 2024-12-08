from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

def outlier_removal(df, iso_forest = None):
    
    if iso_forest is None:
        X = df.values

        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=0.01)
        iso_forest.fit(X)

        # Save model
        with open('isoforest_model.pkl', 'wb') as f:
            pickle.dump(iso_forest, f)

    # Predict outliers
    X = df.values
    outliers = iso_forest.predict(X)
    print("Inliers:", X[outliers == 1].shape)
    df = df[outliers == 1]
    return df

def standardize(df, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[['year', 'operating_hours', 'efficiency', 'registration_fees', 'engine_capacity']])
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
    
    df[['year', 'operating_hours', 'efficiency', 'registration_fees', 'engine_capacity']] = scaler.transform(df[['year', 'operating_hours', 'efficiency', 'registration_fees', 'engine_capacity']])
    
        
    return df

def encoder(df, enc=None):
    
    if enc is None:
        list_of_model = df['model'].unique()
        print('list_of_model shape:', list_of_model.shape)
        list_of_gearbox_type = df['gearbox_type'].unique()
        print('list_of_gearbox_type shape:', list_of_gearbox_type.shape)
        list_of_fuel_type = df['fuel_type'].unique()
        print('list_of_fuel_type shape:', list_of_fuel_type.shape)

        enc = OneHotEncoder(handle_unknown='infrequent_if_exist', categories=[list_of_model, list_of_gearbox_type, list_of_fuel_type])
        enc.fit(df[['model', 'gearbox_type', 'fuel_type']])
        
        with open('encoder.pkl', 'wb') as f:
            pickle.dump(enc, f)
    
    encoded_feature = enc.transform(df[['model', 'gearbox_type', 'fuel_type']])
    
    df.drop(['model', 'gearbox_type', 'fuel_type'], axis=1, inplace=True)
    df = pd.concat([df, pd.DataFrame(encoded_feature.toarray())], axis=1)
    
    return df
