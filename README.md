## End To End Diamond Price Project

### created a environment
```
conda create -p vk python==3.8

conda activate vk/
```
### Install all necessary libraries
```
pip install -r requirements.txt
```
## Diamond Price Prediction

### Introduction About the Data :

**The dataset** The goal is to predict `Time_taken (min)` of given diamond (Regression Analysis).

There are 20 independent variables (including `id`):

* `id` : unique identifier of each diamond
* `Delivery_person_ID ` : unique identifier of each Delivery person.
* `Delivery_person_Age` : Age of every Delivery person.
* `Delivery_person_Ratings`: Rating about Delivery person.
* `Restaurant_latitude`:  
* `Restaurant_longitude`:       
* `Delivery_location_latitude`: 
* `Delivery_location_longitude`:
* `Order_Date`: About Order Date.                
* `Time_Orderd`: what time order booked.              
* `Time_Order_picked`: what time order Received.       
* `Weather_conditions`: how's the weather.      
* `Road_traffic_density`: how was the traffic.      
* `Vehicle_condition`: how was the vehicle.         
* `Type_of_order`: what type of order.             
* `Type_of_vehicle`: what type of vehicle.           
* `multiple_deliveries`: show how many multiple deliveries.      
* `Festival`: show was there a festival or not.              
* `City`: show Order delivery in which city. 

Target variable:
* `Time_taken (min)`: Time_taken (min) of the given Diamond.

Dataset Source Link :
[https://drive.google.com/file/d/1tr1ozeDCuE9AvoL0D7l00hjfl-k83Yab/view?usp=share_link](https://drive.google.com/file/d/1tr1ozeDCuE9AvoL0D7l00hjfl-k83Yab/view?usp=share_link)