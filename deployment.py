from datetime import datetime
import json
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import paho.mqtt.client as paho

class weatherDataset(Dataset):
    """My own Weather Dataset"""

    def __init__(self, csv_file,sequence_len,device):
        self.device = device
        dataset = pd.read_csv(csv_file, header=0)
        self.time = dataset["time"]
        dataset = dataset.drop(columns="time")
        #a = dataset.reset_index(drop=True, inplace=True)     
        #print(a)   
        dataset = dataset.values
        self.sequence_length = sequence_len

        # for data in dataset:
        #     newData += ([eval(i) for i in data])
        #dataset = ((newData-newData.mean())/newData.std())
        dataset_tens = torch.tensor(dataset, dtype=torch.float)
        self.norm_data = dataset_tens[:,2:].clone().detach()
        self.solar = dataset_tens[:,0].clone().detach()
        self.wind = dataset_tens[:,1].clone().detach()
        # self.norm_data = X / X.max(0, keepdim=True)[0]
        # self.solar = weatherData_y_solar / weatherData_y_solar.max(0, keepdim=True)[0]
        # self.wind = weatherData_y_wind / weatherData_y_wind.max(0, keepdim=True)[0]

    def __len__(self):
        return len(self.norm_data)
    
    def get_date(self,row):
        return self.time[row]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.norm_data[i_start:(i + 1)]
        else:
            padding = self.norm_data[0].repeat(self.sequence_length - i - 1, 1)
            x = self.norm_data[0:(i + 1)]
            x = torch.cat((padding, x), 0)

        predictionSolar = self.solar[i]
        predictionWind = self.wind[i]
        sample = [x.to(self.device),predictionSolar.to(self.device),predictionWind.to(self.device)]
        return sample

class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_features, hidden_units, device):
        super().__init__()
        self.device = device
        self.num_sensors = num_features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 1

        self.lstm = nn.LSTM(
            input_size=self.num_sensors,
            hidden_size=self.hidden_units,
            batch_first=True,
            dropout= 0.5,
            num_layers=self.num_layers
        )
        self.fc_1 =  nn.Linear(self.hidden_units, 32) #fully connected 1
        self.fc = nn.Linear(32, 16) #fully connected last layer
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=16, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(self.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(self.device)
        
        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.relu(hn[0])  # First dim of Hn is num_layers, which is set to 1 above.
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Fin
        out = self.linear(out).flatten()
        return out

# def denorm(val,min,max):
# return (val+min)*(max-min)
def deploy_model(model:ShallowRegressionLSTM,row:int,s_l):
    SEQUENCE_LEN = s_l # 7 hours needed for 1 hour prediction
    device = torch.device("cpu")
    dataset_test = weatherDataset("Testing_Set.csv",SEQUENCE_LEN,device)
    date = dataset_test.get_date(row)
    datetime_object = datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z')
    [x,y_sol,y_win] = dataset_test[row]
    model.eval()
    prediction = model(torch.unsqueeze(x,dim=0))
    return datetime_object.strftime("%m/%d/%Y, %H:%M:%S"),prediction.detach().numpy()[0],y_sol.detach().numpy(),y_win.detach().numpy()

def paho_client():
    # CLIENT PAHO
    port = 1883
    username = 'mosquittoBroker'
    password = 'se4gd'
    client_id = f'Solar_Client'
    client = paho.Client(client_id)
    client.username_pw_set(username, password)
    if client.connect("localhost",1883,60)!=0:
        print("Could not connect to MQTT Broker!")
        sys.exit(-1)
    else:
        print("connected")
    client.publish("Agriculture/solar",1)
    return client
    #client.disconnect()

def energyProduction(i,model,amountOfSolarPanels=10):
    #Most residential solar panels on today's market are rated to produce between 250 and 400 watts each per hour (kwh).
    SOLAR_PANEL_PRODUCTION = 400*amountOfSolarPanels
    [date,prediction_sol,y_sol,_y_win] = deploy_model(model,i,48)
    if(prediction_sol<0):
        prediction_sol=0
    return [date,prediction_sol*SOLAR_PANEL_PRODUCTION]
    
def energyConsumption(date:datetime,sizeInSqm=0,people=1):
    date = datetime.strptime(date,"%m/%d/%Y, %H:%M:%S")
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8847370/
    AVERAGE_MONTHLY = 4000000/12
    if(people>1):
        AVERAGE_MONTHLY+=(AVERAGE_MONTHLY//2)*(people-1)
    def pdf(x):
        mean = np.mean(x)
        std = np.std(x)
        y_out = (std * np.sqrt(2 * np.pi)) * np.exp( - (x - mean)**2 / (2 * std**2))
        return y_out
    start = (AVERAGE_MONTHLY//2)-(AVERAGE_MONTHLY//2*(people-1))
    end = AVERAGE_MONTHLY+(AVERAGE_MONTHLY//2*(people-1))
    print(start,end)
    x = np.arange(start, end, (end-start)/5)
    x = np.roll(x,6)
    x = pdf(x)
    month = date.month
    dailyEnergy = x[month-1]/30
    hourlyEnergy_x = (dailyEnergy/24)
    start = (hourlyEnergy_x//12)
    end = ((hourlyEnergy_x*2))
    hourlyEnergy = np.arange(start,end,(end-start)/24)
    hourlyEnergy = np.roll(hourlyEnergy,3)
    hourlyEnergy = pdf(hourlyEnergy)
    return hourlyEnergy[date.hour-1]
def main():
    i=24
    simulation_len = len(pd.read_csv("Testing_Set.csv", header=0))
    solar_model = ShallowRegressionLSTM(30,32,"cpu")
    stateDict_sol = torch.load("solar_model_bigger.pt",map_location=torch.device('cpu'))
    solar_model.load_state_dict(stateDict_sol)
    solar_model.eval()
    client = paho_client()
    while(True):
        if(i>=simulation_len-1):
                i=0
        time.sleep(1)
        [dateForI,solar] = energyProduction(i,solar_model)
        consumption = energyConsumption(dateForI)
        print(f"Date:{dateForI}, Production: {solar}, Consumption:{consumption}")
        solar_publish = json.dumps({"production":solar,"timestamp":dateForI})
        consumption_publish = json.dumps({"consumption":solar,"timestamp":dateForI})
        client.publish("prediction/production",solar_publish)
        client.publish("prediction/consumption",consumption_publish)
        i=i+1
main()