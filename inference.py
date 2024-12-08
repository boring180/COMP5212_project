import torch
import torch.nn as nn
import pandas as pd
import pickle
import preprocessing as pp

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
elif torch.backends.cuda.is_built():
    mps_device = torch.device("cuda")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")
    
# Define model
class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()
        
        self.layers = nn.Sequential(
        nn.Linear(8, 1024),
        nn.ReLU(), 

        nn.Linear(1024, 64),
        nn.ReLU(), 
        
        nn.Linear(64, 32),
        nn.ReLU(),
        
        nn.Linear(32, 4),
        nn.ReLU(),
        
        nn.Linear(4, 1)
        )
        
        self.model_layer = nn.Sequential(
        nn.Linear(169, 1),
        nn.ReLU()
        )
        
        self.gear_box_layer = nn.Sequential(
        nn.Linear(3, 1),
        nn.ReLU()
        )
        
        self.fuel_type_layer = nn.Sequential( 
        nn.Linear(5, 1),
        nn.ReLU()
        )
        
    def forward(self, x):
        # print(x.shape)
        model = self.model_layer(x[:, 5:174])
        gear_box = self.gear_box_layer(x[:, 174:177])
        fuel_type = self.fuel_type_layer(x[:, 177:182])
        
        x = torch.cat((model, gear_box, fuel_type, x[:, :5]), 1)
        
        return self.layers(x)
    
df = pd.read_csv('data/test.csv')
model = torch.load('mlp_model.pth', weights_only=False)
model.eval()
index = df['id']
df = df.drop(['id'], axis=1)
df = df.drop(['manufacturer'], axis=1)
df = pp.standardize(df, pickle.load(open('scaler.pkl', 'rb'))) 
df = pp.encoder(df, pickle.load(open('encoder.pkl', 'rb')))
X = torch.tensor(df.values, dtype=torch.float32)
y_pred = model(X)[:, 0]

df = pd.DataFrame(y_pred.detach().numpy(), columns=['price'])
df = pd.concat([index, df], axis=1)
df.rename(columns={'price': 'answer'}, inplace=True)
df.to_csv('submission.csv', index=False)