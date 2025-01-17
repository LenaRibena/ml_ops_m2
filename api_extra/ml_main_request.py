import requests

# Upload image.jpg to the server
file_path = "../water.jpeg"
with open(file_path, 'rb') as f:
	response = requests.post('http://127.0.0.1:8020/caption/', files={'image': f})

print(response.json())