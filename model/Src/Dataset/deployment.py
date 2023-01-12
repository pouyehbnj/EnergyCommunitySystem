from datetime import datetime
import torch
from dataset import ShallowRegressionLSTM,weatherDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import pyplot

def deploy_model(model:ShallowRegressionLSTM,row:int,s_l):
    """Deploy Solar Model for production """
    SEQUENCE_LEN = s_l # 7 hours needed for 1 hour prediction
    device = torch.device("cpu")
    dataset_test = weatherDataset("Data/Testing_Set.csv",SEQUENCE_LEN,device)
    date = dataset_test.get_date(row)
    datetime_object = datetime.strptime(date, '%Y-%m-%d %H:%M:%S%z')
    [x,y_sol,y_win] = dataset_test[row]
    model.eval()
    prediction = model(torch.unsqueeze(x,dim=0))
    return datetime_object,prediction.detach().numpy()[0],y_sol.detach().numpy(),y_win.detach().numpy()


solar_model = ShallowRegressionLSTM(30,32,"cpu")

stateDict_sol = torch.load("Models\solar_model_bigger.pt")


solar_model.load_state_dict(stateDict_sol)
solar_model.eval()



wind_prediction = []
wind = []
solar_prediction = []
solar = []
time = []
all_add = []
i = 24
while(i<48*14):
    generation_solar = [0.0, 5792.0]
    generation_wind_onshore =  [0.0, 17436.0]
    print(i)
    [date,prediction_sol,y_sol,_y_win] = deploy_model(solar_model,i,48)
    wind.append(_y_win)
    solar.append(y_sol)
    solar_prediction.append(prediction_sol)
    time.append(date)
    i+=1
dict = { 'date': time, 'y_sol': solar, 'prediction_sol': solar_prediction } 

df = pd.DataFrame(dict)
plt.plot(df["date"], df["prediction_sol"],label="Solar Prediction",linewidth=0.5)
plt.plot(df["date"], df["y_sol"],label="Real Solar",linewidth=0.5)
plt.legend()
plt.xticks(rotation=30, ha='right')
plt.savefig('Testing.jpg')
plt.show()