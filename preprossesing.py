from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd

df = pd.read_csv('data/train.csv')

est = KBinsDiscretizer(n_bins=[8, 10], strategy='kmeans', subsample=None)
est.fit(df[['registration_fees', 'engine_capacity']])

list_of_model = df['model'].unique()
print('list_of_model shape:', list_of_model.shape)
list_of_gearbox_type = df['gearbox_type'].unique()
print('list_of_gearbox_type shape:', list_of_gearbox_type.shape)
list_of_fuel_type = df['fuel_type'].unique()
print('list_of_fuel_type shape:', list_of_fuel_type.shape)

enc = OneHotEncoder(handle_unknown='ignore', categories=[list_of_model, list_of_gearbox_type, list_of_fuel_type])
enc.fit(df[['model', 'gearbox_type', 'fuel_type']])

def encoder(df):
    df = df.drop(['manufacturer'], axis=1)
    
    # discrete_feature = est.transform(df[['registration_fees', 'engine_capacity']]).toarray()
    # data_count = discrete_feature.sum(axis=0)
    # print('Data counts:', data_count)
    
    encoded_feature = enc.transform(df[['model', 'gearbox_type', 'fuel_type']])
    
    df.drop(['model', 'gearbox_type', 'fuel_type'], axis=1, inplace=True)
    df = pd.concat([df, pd.DataFrame(encoded_feature.toarray())], axis=1)

    # df.drop(['model', 'gearbox_type', 'fuel_type', 'registration_fees', 'engine_capacity'], axis=1, inplace=True)
    # df = pd.concat([df, pd.DataFrame(discrete_feature)], axis=1) 
    
    return df

from sklearn.preprocessing import StandardScaler
import pickle

def standardize(df1, df2=None, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(df1[['year', 'operating_hours', 'efficiency', 'registration_fees', 'engine_capacity']])
        # scaler.fit(df['year', 'operating_hours', 'efficiency'])
    
    # df['year', 'operating_hours', 'efficiency'] = scaler.transform(df['year', 'operating_hours', 'efficiency'])
    df1[['year', 'operating_hours', 'efficiency', 'registration_fees', 'engine_capacity']] = scaler.transform(df1[['year', 'operating_hours', 'efficiency', 'registration_fees', 'engine_capacity']])
    if df2 is not None:
        df2[['year', 'operating_hours', 'efficiency', 'registration_fees', 'engine_capacity']] = scaler.transform(df2[['year', 'operating_hours', 'efficiency', 'registration_fees', 'engine_capacity']])
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    return df1, df2

# df = pd.read_csv('data/train.csv')
# standardize(df)
