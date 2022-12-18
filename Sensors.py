import datetime
import random
import sys
import time
import pandas as pd
from csv import writer
import paho.mqtt.client as paho

def Nutrition_Actuator(npk_val):
    lastFeeding = 0
    if(npk_val<=40):
        lastFeeding+=5
    elif(npk_val<=35):
        lastFeeding+=8
    elif(npk_val<=30):
        lastFeeding+=10
    return lastFeeding

def Water_Actuator(humidity):
    lastWatering = 0
    if(humidity<=45):
        lastWatering+=10
    elif(humidity<=50):
        lastWatering+=4
    elif(humidity<=30):
        lastWatering+=20
    elif(humidity<=10):
        lastWatering+=30
    
    return lastWatering


def get_solar_irr(lastVal):
    # 1000 ist unsere Grenze
    val = lastVal+1
    MIN = 500
    MAX = 5000
    energy_data = pd.read_csv("Data_set.csv")
    solar = energy_data["generation solar"]
    if(lastVal>=len(solar)-1):
        val = 0
    returnSol = solar[val]
    if(returnSol>=MAX):
        returnSol = MAX
    elif(returnSol<=MIN):
        returnSol = MIN
    returnSol = (returnSol-MIN)/(MAX-MIN)

    return [val,returnSol*100]

def get_humidity(lastVal):
    # 1000 ist unsere Grenze
    val = lastVal+1
    energy_data = pd.read_csv("Data_set.csv")
    humidity = energy_data["humidity"]
    if(lastVal>=len(humidity)-1):
        val = 0
    returnPress = humidity[val]
    returnPress = (returnPress-humidity.min())/(humidity.max()-humidity.min())

    return returnPress*100

def get_pressure(lastVal):
    # 1000 ist unsere Grenze
    val = lastVal+1
    energy_data = pd.read_csv("Data_set.csv")
    pressure = energy_data["pressure"]
    if(lastVal>=len(pressure)-1):
        val = 0
    returnPress = pressure[val]
    returnPress = (returnPress-pressure.min())/(pressure.max()-pressure.min())

    return returnPress*100

def get_temp(lastVal):
    # 1000 ist unsere Grenze
    val = lastVal+1
    energy_data = pd.read_csv("Data_set.csv")
    temp = energy_data["temp"] - 273.15
    if(lastVal>=len(temp)-1):
        val = 0
    returnPress = temp[val]
    return returnPress

def get_npks(lastFeeding):
    ppm = 60
    factor = 1
    if(lastFeeding>5):
        factor-=0.05
    if(lastFeeding>8):
        factor-=0.1
    if(lastFeeding>15):
        factor-=0.15
    if(lastFeeding>20):
        factor-=0.2
    if(lastFeeding>30):
        factor-=0.3
    if(lastFeeding>40):
        factor -= 0.175
    return ppm*factor

def get_soil_moist(val,Temp,lastWatering):
    heat = Temp/40
    hum_sensed = 0
    energy_data = pd.read_csv("Data_set.csv")
    humidity = energy_data["humidity"]
    val -= 5
    if(val<=0):
        hum_sensed = humidity[humidity.size+val]
    else:
        hum_sensed = humidity[val]
    #0-1
    returnHum = (hum_sensed-humidity.min())/(humidity.max()-humidity.min()) 

    if(lastWatering>5 and lastWatering<10):
        returnHum = returnHum*random.uniform(0.8,0.9)
    elif(lastWatering>10 and lastWatering<20):
        returnHum = returnHum*random.uniform(0.5,0.8)
    elif(lastWatering>20 and lastWatering<30):
        returnHum = returnHum*random.uniform(0.2,0.5)
    elif(lastWatering>30 and lastWatering<40):
        returnHum = returnHum*random.uniform(0.2,0.5)
    return (returnHum - 0.1*(heat))*100


def get_ph(npk_vals):
    # here between 5 and 8
    return 0.05*npk_vals+5

def get_timestamp(val,seconds=0):
    # here between 5 and 8
    energy_data = pd.read_csv("Data_set.csv")
    year = energy_data["year"]
    month = energy_data["month"]
    day = energy_data["day"]
    hour = energy_data["hour"]
    timestamp = datetime.datetime(year=year[val],month=month[val],day=day[val],hour=hour[val],minute=0,second=seconds).strftime("%m/%d/%Y, %H:%M:%S")
    return timestamp


def paho_client():
    # CLIENT PAHO
    port = 1883
    broker = f'mqtt://192.168.199.161:{port}/'
    username = 'agricultureLove'
    password = 'se4gd'
    client_id = f'Solar_Client'
    client = paho.Client(client_id)
    client.username_pw_set(username, password)
    if client.connect("192.168.152.161",1883,60)!=0:
        print("Could not connect to MQTT Broker!")
        sys.exit(-1)
    else:
        print("connected")
    client.publish("Agriculture/solar",1)
    return client
    #client.disconnect()

def main(client):
    val = random.randint(0,35000)
    i = 0
    lastFeeding = 40
    lastWatering = 10
    while(True):
        time.sleep(5)
        Solar,Pressure,Temp,Humidity,npk_val,ph,soil_moist = "None","None","None","None","None","None","None"
        [nextVal,Solar] = get_solar_irr(val)
        solarTime,pressureTime,tempTime,humidityTime,npkTime,phTime,moistTime,feedingTime,wateringTime = "None","None","None","None","None","None","None","None","None"
        solarTime = get_timestamp(val,0)
        print(Solar)
        if(Solar>0.0):
            # Converted from Pascal to Percent
            Pressure = get_pressure(val)
            pressureTime = get_timestamp(val,1)
            # in °C
            Temp = get_temp(val)
            tempTime = get_timestamp(val,2)

            # In Percent
            Humidity = get_humidity(val)
            humidityTime = get_timestamp(val,3)

            # in ppm (40-60 is optimum)
            npk_val = get_npks(lastFeeding)
            npkTime = get_timestamp(val,4)

            # In Percent
            soil_moist = get_soil_moist(val,Temp,lastWatering)
            moistTime = get_timestamp(val,5)

            # In ph Scale
            ph = get_ph(npk_val)
            phTime = get_timestamp(val,6)

            # water actuator
            lastWatering -= Water_Actuator(humidity=Humidity)
            feedingTime = get_timestamp(val, 10)
            lastFeeding -= Nutrition_Actuator(npk_val)
            wateringTime = get_timestamp(val, 12)

         


            
            client.publish("Agriculture/solar",str(Solar)+","+solarTime)
            client.publish("Agriculture/pressure",str(Pressure)+","+pressureTime)
            client.publish("Agriculture/temp",str(Temp)+","+tempTime)
            client.publish("Agriculture/humidity",str(Humidity)+","+humidityTime)
            client.publish("Agriculture/npk_val",str(npk_val)+","+npkTime)
            client.publish("Agriculture/soil_moist",str(soil_moist)+","+moistTime)
            client.publish("Agriculture/ph",str(ph)+","+phTime)
            client.publish("Agriculture/feeding",str(lastWatering)+","+feedingTime)
            client.publish("Agriculture/watering",str(lastFeeding)+","+wateringTime)
        # client.publish("Agriculture/timeStamp",time_stamp)

        lastFeeding+=1
        if(lastWatering<100):
            lastWatering+=1
        val = nextVal
        arr = [Solar,Pressure,Temp,Humidity,npk_val,ph,soil_moist,lastFeeding,lastWatering]
        i+=1

        with open('sensor_data.csv', 'a',newline='') as f_object:
            # Pass this file object to csv.writer()
            # and get a writer object
            writer_object = writer(f_object)
        
            # Pass the list as an argument into
            # the writerow()
            writer_object.writerow(arr)
        
            # Close the file object
            f_object.close()


client = paho_client()
main(client)
