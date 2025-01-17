import requests

# Example of a simple GET request
response = requests.get('https://api.github.com')
if response.status_code == 200:
    print('Success!')
elif response.status_code == 404:
    print('Not Found.')

# Example of a simple GET and how it's contents can be accessed
response = requests.get("https://api.github.com/repos/SkafteNicki/dtu_mlops")
print(response.json())

# Example of a simple GET request with parameters 
response = requests.get(
    'https://api.github.com/search/repositories',
    params={'q': 'requests+language:python'},
)

# Example of a simple GET request for an image
response = requests.get('https://imgs.xkcd.com/comics/making_progress.png')

# Example of a simple POST request
pload = {'username': 'Olivia', 'password': '123'}
response = requests.post('https://httpbin.org/post', data=pload)