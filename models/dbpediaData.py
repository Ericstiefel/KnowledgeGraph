import requests

url = "http://api.live.dbpedia.org/resource/en/Yacine_Diop"
response = requests.get(url)

print("Status Code:", response.status_code)
print("Response Text:", repr(response.text))

# Only try to parse JSON if status is 200 
if response.status_code == 200 and response.text.strip():
    data = response.json()
    print(data)
else:
    print("Failed to get valid JSON from the server.")
