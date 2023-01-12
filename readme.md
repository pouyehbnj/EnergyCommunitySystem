# Renewable Energy Communities

The system allows individual households to be added to an energy community where they tend to become prosumers (households which produce and consume their own energy). The system is specialised for Spain as it is using its weather data. The system allows the households to be connected to grids as creating a fully energy-independent community is not a realistic scenario. This way users can buy or sell energy from or to the energy companies. The system shows the households data about their energy production and consumption. It also provides data about users’ monthly bills or a receipt of how much money they made by selling their extra produced energy.

The services provided are splitted into the following:

- We established the VM instance that is running on the Google Cloud Platform with corresponding sql database. 
- Communication with the mqtt was established both locally and with the docker on virtual machine.
- microservice establishes a MQTT connection by subscribing to the topics related to prediction of energy production and consumption
- the microservice retrieve data from our prediction model for analysis and visualisation to the users
- Using MQTT protocol allows the microservice to be notified whenever the new prediction data is produced by the Prediction Model microservice
- After receiving data, the microservice starts plotting the most updated data continuously for the users for analysis and visualisation purposes. 
- We can also use this generic microservice further for improving the data analysis by adding real energy consumption and production data to the plots and making valuable comparisons for the users of the energy community
