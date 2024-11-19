from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd

df = pd.read_csv('data/train.csv')

est = KBinsDiscretizer(n_bins=[12, 20], strategy='kmeans', subsample=None)
est.fit(df[['registration_fees', 'engine_capacity']])

list_of_manufacturers = df['manufacturer'].unique()
list_of_model = df['model'].unique()
list_of_gearbox_type = df['gearbox_type'].unique()
list_of_fuel_type = df['fuel_type'].unique()

enc = OneHotEncoder(handle_unknown='ignore', categories=[list_of_manufacturers, list_of_model, list_of_gearbox_type, list_of_fuel_type])
enc.fit(df[['manufacturer', 'model', 'gearbox_type', 'fuel_type']])



def encoder(df):
    
    discrete_feature = est.transform(df[['registration_fees', 'engine_capacity']]).toarray()
    encoded_feature = enc.transform(df[['manufacturer', 'model', 'gearbox_type', 'fuel_type']])
    
    df.drop(['manufacturer', 'model', 'gearbox_type', 'fuel_type', 'registration_fees', 'engine_capacity'], axis=1, inplace=True)
    df = pd.concat([df, pd.DataFrame(encoded_feature.toarray())], axis=1)
    df = pd.concat([df, pd.DataFrame(discrete_feature)], axis=1) 
    
    return df

from sklearn.preprocessing import StandardScaler
import pickle

def standardize(df, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
    
    df['year'] = scaler.fit_transform(df['year'].values.reshape(-1, 1))
    df['operating_hours'] = scaler.fit_transform(df['operating_hours'].values.reshape(-1, 1))
    df['efficiency'] = scaler.fit_transform(df['efficiency'].values.reshape(-1, 1))

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    return df

def test_validation_split(df):
    from sklearn.model_selection import train_test_split
    train, val = train_test_split(df, test_size=0.2)
    return train, val

