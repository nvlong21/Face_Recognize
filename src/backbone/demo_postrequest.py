# importing the requests library 
import requests 
  
# defining the api-endpoint  
API_ENDPOINT = "http://35.187.243.134:8084/emotion/add_raw"

 
data = {
		“id_class”: "CL485162",
		“time_video”:"0:0:30",
		“time_class”: "19:00",
		“id_teacher”: 234,
		“name_teacher”: "John Wick"
		“angry”: ,
		“happy”:,
		“surprise: ,
		“sad”: 
		}

  
# sending post request and saving response as response object 
r = requests.post(url = API_ENDPOINT, data = data)
