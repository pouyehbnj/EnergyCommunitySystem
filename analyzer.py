import json
import paho.mqtt.client as mqtt
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
    
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("prediction/production")
    client.subscribe("prediction/consumption")

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
globalArr_time = []
globalArr_production = []

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    global globalArr_time
    global globalArr_production
    global fig
    global ax

    if(str(msg.topic)=="prediction/production"):
        payl = json.loads(str(msg.payload.decode("utf-8","ignore")))
        print("topic:"+msg.topic+"- message:"+str(msg.payload))
        x = []
        y = []
        
        if(len(globalArr_time)>=6):
            arr1 = globalArr_time[-6:]
            arr2 = globalArr_production[-6:]
            print(arr1)
            plt.clf()

            arr1.append(payl["timestamp"])
            globalArr_time.append(payl["timestamp"])
            x = arr1
            arr2.append(payl["production"])
            globalArr_production.append(payl["production"])

            y = arr2
        else:
            globalArr_time.append(payl["timestamp"])
            globalArr_production.append(payl["production"])
            x = globalArr_time
            y = globalArr_production
        
        plt.plot(x,y)
        plt.xticks(rotation=45, ha='right')
        plt.subplots_adjust(bottom=0.30)
        plt.title('Production over Time')
        plt.ylabel('Production')
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        



    # Format plot



    

   
    
client = mqtt.Client()
username = 'mosquittoBroker'
password = 'se4gd'
client.username_pw_set(username, password)
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883, 60)
plt.show(block=True) 

# block=True lets the window stay open at the end of the animation.
# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# # manual interface.
# client.loop_start()
# time.sleep(4)
# client.loop_stop()
client.loop_forever()

# test