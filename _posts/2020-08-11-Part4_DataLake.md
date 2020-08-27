---  
title:  "Part4_Data Lake"  
  
categories:  
 - Data Engineering
tags:  
 - Study, Data Engineering
 
---
패스트 캠퍼스에서 수강하는 데이터 엔지니어링 강의 내용의 정리본이다.

# Part4_Data Lake

### 목차

-  Step 1. Data Lake
-  Step 2. AWS S3 
-  Step 3. Data Lake 구축

## Step 1. Data Lake

'가공되지 않은 다양한 종류의 데이터를 한 곳에 모아둔 저장소의 집합'

시대의 변화에 따라 데이터 저장 방식이 진화를 했다. 그 이유는 빅데이터와 인공지능 기술의 중요성이 커지면서 다양한 영역의 다양한 데이터가 만나 새로운 가치를 만들어내기 시작했다.

이와 같이 빅데이터를 효율적으로 분석하고 사용하고자 다양한 영역의 Raw data를 한 곳에 모아 관리하고자 하는 것을 Data Lake 라고 한다.

### 1) Data Lake vs Data Warehouse

둘의 가장 큰 차이점은 Data Structure/Schema 이다.
기존 DB엔 structure가 존재하며 어떻게 쓸지 설계한 뒤 데이터를 수집했다. (e.g. RDB) 

그러나 현재는 데이터가 너무 방대하기에  데이터 사용 비용이 상당해졌고 여러 문제가 발생하기에 수집 과정이 전자가 되어버렸다.

- 기존 방식 : ETL(Extract -> Transform -> Load)
- 요즘 방식 : ELT(Extract -> Load in to data lake -> Transform)

그렇다면 데이터를 재가공 과정에서 시간이 너무나 소요되지 않을까? 과거에는 접근성이 떨어지는 단점들이 존재했지만 Hadoop, Spark 등의 빅데이터 처리 tool을 이용해 더 빠르게 접근할 수 있는 능력이 생겼다.
 
## Step 2. AWS S3

Amazon S3는 Amazon Simple Storage Service로 인터넷용 스토리지 서비스이다. S3에서 제공하는 인터페이스를 사용하여 웹에서 언제 어디서나 원하는 양의 데이터를 저장하고 검색이 가능하다. 

즉, S3는 '높은 내구성'과 '높은 가용성'을 '저렴한 가격'으로 제공하는 '인터넷 스토리지 서비스' 이다.

S3는 Bucket , Object, Key로 구성되어 있다.

- Bukcet : S3에 저장된 객체에 대한 컨테이너
- Object : 파일 및 파일정보(Data and Metadata)로 구성된 저장 단위
- Key : 버킷 내 객체의 고유한 식별자로 버킷 내 모든 객체는 정확히 하나의 키를 가짐

## Step 3. Date Lake 구축

### 1) S3 생성
AWS 서비스에서 S3를 선택하여 버킷을 만들면 된다. 버킷 이름만 설정하고 나머지 과정은 모두 Default 값으로 설정하면 완성이다.

![image](https://user-images.githubusercontent.com/59912557/90214103-77ed6880-de32-11ea-8a52-c37e1cab83ad.png)

![image](https://user-images.githubusercontent.com/59912557/90214132-876cb180-de32-11ea-952e-07016c224c64.png)


![image](https://user-images.githubusercontent.com/59912557/90214067-5ee4b780-de32-11ea-919f-a2e84203435d.png)

### 2) S3에 데이터 넣기

MySQL DB에 저장해둔 정보를 이용해 top_tracks와 audio_features data를 S3에 넣겠다.

```python
import sys
import requests
import base64
import json
import logging
import pymysql
import boto3
from datetime import datetime
import pandas as pd
import jsonpath

# 이전 과정에서 발급받은 정보들을 기입하면 된다
client_id = ''
client_secret = ''

host = ''
port = 3306
username = ''
database = ''
password = ''

def main():
	
	try:
		conn = pymysql.connect(host, user = username, passwd = password, db = database, port = port, use_unicode = True, charset = 'utf8')
		cursor = conn.cursor()
	except:
		logging.error('could not connect to RDS')
	
	# get access_token
	headers = get_headers(client_id, client_secret)
	
	# bring artists_id 
	cursor.execute('SELECT id FROM artists')
	
	## top_tracks 가져오기
	# jsonpath library를 이용해 json 형식에서 원하는 data만 가져오도록
	top_tracks_keys = {
		'id' = 'id',
		'name' = 'name',
		'popularity' = 'popularity',
		'external_url' = 'external_urls.spotify'
	}

	top_tracks = []
	# id 가져오기
	for (id, ) in cursor.fetchall():
		# endpoints
		URL = URL = 'https://api.spotify.com/v1/artists/{}/top-tracks'.format(id)
		# top tracks params
		params = {
			'country' = 'US'
		}
		# Use API
		r = requests.get(URL, headers = headers, parms = params)
		# string to dict
		raw = json.loads(r.text)
		
		top_tracks = []
		for i in raw['track']:
			top_track = {}

			for key, value in top_tracks_keys.items():
				# 위에서 설정한 원하는 key,value값만 update
				top_track.update({key: jsonpath.jsonpath(i, value)})
				top_track.update({'artist_id' : id})
				top_tracks.append(top_track)

	# top_tracks_id
	tracks_ids = [track['id'][0] for track in top_tracks]

	# S3 저장에 효율적인 parquet 형식으로 변환
	top_tracks = pd.DataFrame(top_tracks)
	top_tracks.to_parquet('top-tracks.parquet', engine = 'payarrow', compression = 'snappy')
	
	# S3 import
	s3 = boto3.resource('s3', region_name = 'ap-northeast-2')

	# utcnow 기준 시간을 strftime으로 변형
	# s3 폴더를 날짜 별로 구분하기 위해
	dt = datetime.utcnow().strftime('%Y-%m-%d')

	# object 생성
	object = s3.Object('버켓이름', 'top-tracks/dt={}/top-tracks.parquet'.format(dt))

	# binary file을 읽기 위해 rb 설정
	data = open('top-tracks.parquet', 'rb')
	# 데이터 쌓기
	object.put(Body = data)

	## audio_features 가져오기
	# batch 형식
	audio_batches = [track_ids[i:i+100] for i in range(0, len(track_ids),100)]
	
	audio_featues = []
	for audio in audio_batches:

		ids = ','join(audio)
        url = 'https://api.spotify.com/v1/audio-features/?ids={}'.format(ids)

		r = requests.get(url, headers= headers)
		raw = json.loads(r.text)

		audio_features.extend(raw['audio_features'])
	
	# DF to Parquet
	audio_featues = pd.DataFrame(audio_features)	
	audio_features.to_parquet('audio_features.parquet', engine = 'pyarrow', compression = 'snappy')

	# S3 import
	objects = s3.Object('버켓이름', 'audio_features/dt={}/top-tracks.parquet'.format(dt))
    data = open('audio_features.parquet', 'rb')
    objects.put(Body = data)

	## aritsts data 가져오기
	cursor.execute('SELECT * FROM artists')
	# array 형태여서 0으로 반환
	colnames = [d[0] for d in cursor.description]
	
	# make dictionary
	artists = [dict(zip(colnames, row)) for row in cursor.fetchall()]

	# dict to df
	df_artists = pd.DatatFrame(artists)
    df_artists.to_parquet('artists.parquet', engine = 'pyarrow', compression = 'snappy')
	
    object = s3.Object('버켓 이름', 'artists/dt={}/artists.parquet'.format(dt))
    # load artists.parquet as rb
    data = open('artists.parquet', 'rb')
    # insert into s3 storage
    object.put(Body = data)	
```

아래와 같이 설정한 폴더를 기준으로 나뉘어 저장된 것을 확인 할 수 있다.

![image](https://user-images.githubusercontent.com/59912557/90216849-a111f700-de3a-11ea-8f0b-f8e1577a0c7a.png)

![image](https://user-images.githubusercontent.com/59912557/90215537-d9173b00-de36-11ea-9070-a55a6f184978.png)


