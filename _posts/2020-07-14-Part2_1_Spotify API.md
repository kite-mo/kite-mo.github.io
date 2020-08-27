---  
title:  "Part2-1 Spotfiy API"  
  
categories:  
 - Data Engineering
tags:  
 - Study, Data Engineering
 
---
패스트 캠퍼스에서 수강하는 데이터 엔지니어링 강의 내용의 정리본이다.

# Part2-1 Spotfiy API

### 목차

-  Step 1. Spotify App 생성 및 토큰 발급
-  Step 2. Python Request package
-  Step 3. API를 통해 데이터 요청
-  Step 4. Status Code
-  Step 5. Error Handling

## Step 1. Spotify App 생성 및 토큰 발급

API를 사용해 Resource를 얻기 위해서는 Authorization 과정을 통해 Access_token ( API Key )를 발급 받아야 한다. 우리가 수행할 과정은 아래와 같다.

![image](https://user-images.githubusercontent.com/59912557/89752032-9a6b4300-db0d-11ea-8520-5420014b8e39.png)

첫째로는 client_id 와 client_secrect을 발급받아야 한다. 

[https://developer.spotify.com/dashboard/applications](https://developer.spotify.com/dashboard/applications)

위 링크로 들어가 dashborad에 계정을 생성하여 ID, Password를 발급 받으면 된다.

![image](https://user-images.githubusercontent.com/59912557/89752119-051c7e80-db0e-11ea-92b0-a311869ec612.png)

발급 받은 이후엔 Spotify가 요구하는 방식으로 access_token을 이용해 데이터들을 얻어오면 된다. 상세 부분은 뒤에서 다루도록 하자.

## Step 2. Python Request package

파이썬을 사용하는 경우 API를 이용할 때, 해당 요청에 특화된 라이브러리인 requests library를 사용해야한다.

[https://requests.readthedocs.io/en/master/](https://requests.readthedocs.io/en/master/)
 
 해당 라이브러리에 관한 다큐멘테이션이다.

가장 많이 사용되는 함수로는 아래와 같다.

 - GET : 해당 리소스를 조회하고 정보를 가져오기/데이터 받아오기( 가장 많이 쓰임 ) 
	e.g.  requests.get(url, params, headers, ..)
 
 - HEAD : GET 방식과 동일하난 응답코드와 HEAD만 가져옴
 API 작동여부 확인에 사용

 - POST : 요청된 리소스를 생성
 requests.post(url, data = data, headers= headers)
 
 - PUT : 요청된 리소스를 업데이트(갱신)

 - DELETE : 요청된 리소스를 삭제

## Step 3. API를 통해 데이터 요청

![image](https://user-images.githubusercontent.com/59912557/89752586-fb941600-db0f-11ea-981f-7cb420cbe6e7.png)

우선 search sesource를 얻어볼 건데, request parameter 중 Authorization 이 필요하다. 위에서 얻은 정보들을 바탕으로 access_token을 얻어내는 함수를 구현해보자.

![image](https://user-images.githubusercontent.com/59912557/89752669-42820b80-db10-11ea-9063-48a90b935e7c.png)
spotify에서 제공하는 guide에서 위와 같이 진행하라고 나와있다. 

```python
import sys
import requests
import base64
import json

# dashboard 계정 생성시 발급되는 id, secrect number
client_id = '2f377076499cfs46fdb9a098fc84c6a74890'
client_secret = '427690c403dc4be383412f52361b99ff138cdss'

def get_headers(client_id, client_secret):
	
	endpoint = 'https://accounts.spotify.com/api/token'
	# base64 형식으로 인코딩이 필요하다고 되어있음
	encoded = base64.b64encoded('{}:{}'.format(client_id, client_secret).encode('utf-8')).decode('ascii')
	
	# header parpameter
	headers = {
		'Authorization' : 'Basic {}'.format(encoded)
	}
	
	# body parameter
	grant_type = {
		'grant_type' : 'client_credentials'
	}
	
	# requests 함수 사용
	r = requests.post(endpoint, headers = headers, data =grant_type)
	
	# string type인 r을 json을 이용해 dictionary 형태로 변환
	raw = json.loads(r.text)
	print(raw)
	
	# access_token
	access_token = raw['access_token']
	
	# "Authorization: Bearer NgCXRKc...MzYjw" 
	headers = {
		'Authorization' : 'Bearer {}'.format(access_token)
	}
	
	return headers
```

```python
   "access_token": "NgCXRKc...MzYjw",
   "token_type": "bearer",
   "expires_in": 3600,
```
이제 access_token도 얻었으니 API를 이용해 정보를 얻어보자.

![image](https://user-images.githubusercontent.com/59912557/89753806-9e4e9380-db14-11ea-86ba-27b12babfd6f.png)

SEARCH resource에선 q, type은 필수 파라미터이다. 해당 파라미터와 get 함수를 이용해 데이터를 얻어보자 

```python
def main():

	headers = get_headers(client_id, client_secret)
	endpoint = 'https://api.spotify.com/v1/search'
	
	# Search parameter
	params = {
		'q' : 'BTS',
		'type' : 'artist',
		# 최대 개수 제한
		'limit' : '5'
	}

	r = requests.get(endpoint, params = params, headers = headers)
	raw = json.loads(r.text)
	print(raw)

if __name__ == '__main__':
	main()
	 
```
```python
{'artists': {'href': 'https://api.spotify.com/v1/search?query=BTS&type=artist&offset=0&limit=5', 'items': [{'external_urls': {'spotify': 'https://open.spotify.com/artist/3Nrfpe0tUJi4K4DXYWgMUX'}, 'followers': {'href': None, 'total': 21291816}, 'genres': ['k-pop', 'k-pop boy group'], 'href': 'https://api.spotify.com/v1/artists/3Nrfpe0tUJi4K4DXYWgMUX', 'id': '3Nrfpe0tUJi4K4DXYWgMUX', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/83971573ae849bb366ff3d9d24623edd938805df', 'width': 640}, {'height': 320, 'url': 'https://i.scdn.co/image/7fe2be9666e4f5cf4a96316086aaa92ba5b6376d', 'width': 320}, {'height': 160, 'url': 'https://i.scdn.co/image/ef0cba018c61d4e9e3225cee946be3e1a03be75f', 'width': 160}], 'name': 'BTS', 'popularity': 93, 'type': 'artist', 'uri': 'spotify:artist:3Nrfpe0tUJi4K4DXYWgMUX'}, {'external_urls': {'spotify': 'https://open.spotify.com/artist/5RmQ8k4l3HZ8JoPb4mNsML'}, 'followers': {'href': None, 'total': 2651071}, 'genres': ['k-rap'], 'href': 'https://api.spotify.com/v1/artists/5RmQ8k4l3HZ8JoPb4mNsML', 'id': '5RmQ8k4l3HZ8JoPb4mNsML', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/820416216c8fb6c3dd265ab9908f46e63ee00158', 'width': 640}, {'height': 320, 'url': 'https://i.scdn.co/image/c39e8f8955f3b9f7b4c91351297f4aa763250711', 'width': 320}, {'height': 160, 'url': 'https://i.scdn.co/image/053da5852cf8ecdcfd6cc400a0596149e34a247e', 'width': 160}], 'name': 'Agust D', 'popularity': 74, 'type': 'artist', 'uri': 'spotify:artist:5RmQ8k4l3HZ8JoPb4mNsML'}, {'external_urls': {'spotify': 'https://open.spotify.com/artist/51sg5jUqKu2tkbmPlwPrNH'}, 'followers': {'href': None, 'total': 328880}, 'genres': [], 'href': 'https://api.spotify.com/v1/artists/51sg5jUqKu2tkbmPlwPrNH', 'id': '51sg5jUqKu2tkbmPlwPrNH', 'images': [], 'name': 'BTS World', 'popularity': 44, 'type': 'artist', 'uri': 'spotify:artist:51sg5jUqKu2tkbmPlwPrNH'}, {'external_urls': {'spotify': 'https://open.spotify.com/artist/1Dx8CcTQA8bWYen7zXsNW0'}, 'followers': {'href': None, 'total': 263}, 'genres': [], 'href': 'https://api.spotify.com/v1/artists/1Dx8CcTQA8bWYen7zXsNW0', 'id': '1Dx8CcTQA8bWYen7zXsNW0', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b273ff015e6e8e0c4cbd4fde4dee', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e02ff015e6e8e0c4cbd4fde4dee', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d00004851ff015e6e8e0c4cbd4fde4dee', 'width': 64}], 'name': 'BTSC', 'popularity': 29, 'type': 'artist', 'uri': 'spotify:artist:1Dx8CcTQA8bWYen7zXsNW0'}, {'external_urls': {'spotify': 'https://open.spotify.com/artist/5vV3bFXnN6D6N3Nj4xRvaV'}, 'followers': {'href': None, 'total': 176521}, 'genres': [], 'href': 'https://api.spotify.com/v1/artists/5vV3bFXnN6D6N3Nj4xRvaV', 'id': '5vV3bFXnN6D6N3Nj4xRvaV', 'images': [{'height': 640, 'url': 'https://i.scdn.co/image/ab67616d0000b2738dd6de651baf8860665f8003', 'width': 640}, {'height': 300, 'url': 'https://i.scdn.co/image/ab67616d00001e028dd6de651baf8860665f8003', 'width': 300}, {'height': 64, 'url': 'https://i.scdn.co/image/ab67616d000048518dd6de651baf8860665f8003', 'width': 64}], 'name': 'JIN', 'popularity': 55, 'type': 'artist', 'uri': 'spotify:artist:5vV3bFXnN6D6N3Nj4xRvaV'}], 'limit': 5, 'next': 'https://api.spotify.com/v1/search?query=BTS&type=artist&offset=5&limit=5', 'offset': 0, 'previous': None, 'total': 55}}
```
이런식으로 API를 이용해 본인이 원하는 정보를 얻어낼 수 있다.

## Step 4. Status Code

status code는 API를 이용하여 데이터를 request 했을 경우 수행 결과 상태를 나타내는 방식이다.  

![image](https://user-images.githubusercontent.com/59912557/89754473-01d9c080-db17-11ea-8e37-dd31eb34bc2a.png)

- 200 : 요청이 성공적인 경우 
- 401 : 토큰이 만료가 되었거나 잘 못 입력된 경우
- 429 : 지정된 요청 용량보다 많은 양이 요청되었을 경우 시간 제한을 두는 상태

## Step 5. Error Handling 

status code가 401, 429인 경우 어떻게 핸들링해야 할지 알아보자. 401 코드에는 토큰이 만료되었기 때문에 재발급 받는 처리를 해주고, 429는 제한된 시간동안 잠시 멈췄다가 다시 요청하는 처리를 해주자

```python
import logging

# error handling
r = requests.get("https://api.spotify.com/v1/search", params = params, headers = headers)

# check status code
if r.status_code != 200:
	# error 표시 
	logging.error(r.text)
	
	if r.status_code == 429:
		
		# 제한 시간 확인
		retry_after = json.loads(r.headers['Retry_After'])
		# 그 시간동안 잠시 멈추기
		time.sleep(int(retry_after))
	
		r = requests.get("https://api.spotify.com/v1/search", params = params, headers = header)

	elif r.status_code == 401:
		
		headers = get_headers(client_id, client_secrect)
		r = requests.get("https://api.spotify.com/v1/search", params = params, headers = header)

```
