import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.nn.functional import normalize,relu
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

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
        
def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train(True)
    i = 0
    arr = []
    for X, y_sol,y_win in data_loader:
        output = model(X)
        loss = loss_function(output, y_sol)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if(i % 25 == 0 or i == 0):
            progress = i/len(data_loader)
            arr.append(loss.item())
            print(f"Train loss: {loss.item()}, progress = {progress}")
        i += 1



    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")
    return avg_loss

def test_model(data_loader, model, loss_function):
    num_batches = len(data_loader)
    total_loss = 0
    i = 0
    arr = []
    out = []
    model.eval()
    with torch.no_grad():
        for X, y_sol,y_win  in data_loader: 
            output = model(X)
            loss = loss_function(output, y_sol)
            out.append([output[0],y_sol[0]])
            total_loss += loss.item()
            if(i % 25 == 0 or i == 0):
                arr.append(loss.item())
                progress = i/len(data_loader)
                print(f"Test loss: {loss.item()}, progress = {progress}")
            i+=1
        
    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    return avg_loss

def main():
    ### Load the data
    torch.manual_seed(103)

    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device(dev) 
    #print(f"Training on device: {device}")
    BATCH_SIZE = 64
    SEQUENCE_LEN = 48
    LEARNING_RATE = 0.0001
    HIDDEN_SIZE = 32
    NUM_FEATURES = 30

    w_dataset_train = weatherDataset("Data/Training_Set.csv",SEQUENCE_LEN,device)
    w_dataset_test = weatherDataset("Data/Testing_Set.csv",SEQUENCE_LEN,device)
    # train_size = int(0.8 * len(chessDataset))
    # test_size = len(chessDataset) - train_size
    # train_set, val_set = torch.utils.data.random_split(chessDataset, [train_size, test_size])
    #print(len(train_set.dataset))

    dataloader_train = DataLoader(w_dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    dataloader_test = DataLoader(w_dataset_test, batch_size=BATCH_SIZE, shuffle=True)
    #dataloader_test = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)

    model = ShallowRegressionLSTM(num_features=NUM_FEATURES, hidden_units=HIDDEN_SIZE,device=device).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    #Main optimization loop
    running_accuracy = []
    myMSE = list()
    nums = list()
    # Loop over epochs
    i = 0

    print("Untrained test\n--------")
    #Test loss: 0.3402042749666074
    arr_0 = test_model(dataloader_test, model, loss_function)
    arr_train = []
    arr_test = []
    for ix_epoch in range(30):
        print(f"Epoch {ix_epoch}\n---------")
        #print(f"\n\n{len(arr_train)}\n\n")
        arr_train.append(train_model(dataloader_train, model, loss_function, optimizer=optimizer))
        arr_test.append(test_model(dataloader_test, model, loss_function))
    solar_prediction = []
    solar = []
    date_arr = []
    model.eval()
    for i in range(30):
        [x,y_sol,y_win] = w_dataset_test[i]
        date_i = w_dataset_test.get_date(i)
        prediction = model(torch.unsqueeze(x,dim=0))
        solar.append(y_sol.detach().cpu().numpy())
        date_arr.append(date_i)
        solar_prediction.append(prediction.detach().cpu().numpy()[0])
    dict = {'date': date_arr, 'y_sol': solar, 'prediction_sol': solar_prediction} 
    df = pd.DataFrame(dict)
    plt.plot(df["date"], df["y_sol"],label="Real Solar",linewidth=0.5)
    plt.plot(df["date"], df["prediction_sol"],label="Solar Prediction",linewidth=0.5)
    plt.legend()
    plt.xticks(rotation=30, ha='right')
    plt.show()
    torch.save(model.state_dict(), "Models/solar_model_bigger.pt")

    #arr_test = test_model(dataloader_test, model, loss_function)
    x = np.arange(len(arr_test))

    plt.title("Training/Testing graph")
    plt.plot(x, arr_train, label="Training",linewidth=0.5)
    plt.plot(x, arr_test, label="Testing",linewidth=0.5)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('Training.jpg')
    plt.show()

    plt.title("Testing graph")

    #plt.plot(x, arr_0, color="red", label="before")
    
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('Testing.jpg')
    plt.show()

#main()