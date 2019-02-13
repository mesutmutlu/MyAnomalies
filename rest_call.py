import requests
url = 'https://jira.digiturk.com.tr/rest/api/2/search'
data = '''{
    "expand": "names,schema",
    "startAt": 0,
    "maxResults": 50,
    "total": 1,
    "jql": "filter%3D24779",
}'''
response = requests.post(url, data=data)
print(response)