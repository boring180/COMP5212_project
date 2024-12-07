import torch
import torch.nn as nn
import pandas as pd

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
        nn.Linear(164, 1),
        nn.ReLU()
        )
        
        self.gear_box_layer = nn.Sequential(
        nn.Linear(3, 1),
        nn.ReLU()
        )
        
        self.fuel_type_layer = nn.Sequential( 
        nn.Linear(4, 1),
        nn.ReLU()
        )
        
    def forward(self, x):
        # print(x.shape)
        model = self.model_layer(x[:, 5:169])
        gear_box = self.gear_box_layer(x[:, 169:172])
        fuel_type = self.fuel_type_layer(x[:, 172:176])
        
        # registration_fee = self.registration_fee_layer(x[:, 156:164])
        # engine_capacity = self.engine_capacity_layer(x[:, 164:174])
        # operating_hours = x[:, 174].view(-1, 1)
        # year = x[:, 175].view(-1, 1)
        # efficiency = x[:, 176].view(-1, 1)

        
        # x = torch.cat((model, year, gear_box, operating_hours, fuel_type, registration_fee, efficiency, engine_capacity), 1)
        
        x = torch.cat((model, gear_box, fuel_type, x[:, :5]), 1)
        
        return self.layers(x)

model = mlp()
model

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
    
    
df = pd.read_csv('data/test.csv')
model = torch.load('mlp_model.pth', weights_only=False)
model.eval()
index = df['id']
X = df.drop(['id'], axis=1)
X = torch.tensor(X.values, dtype=torch.float32)
y_pred = model(X)[:, 0]

df = pd.DataFrame(y_pred.detach().numpy(), columns=['price'])
df = pd.concat([index, df], axis=1)
df.rename(columns={'price': 'answer'}, inplace=True)
df.to_csv('output.csv', index=False)