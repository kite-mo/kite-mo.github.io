---  
title:  "Part3_DB_NoSQL"  
  
categories:  
 - Data Engineering
tags:  
 - Study, Data Engineering
 
---
패스트 캠퍼스에서 수강하는 데이터 엔지니어링 강의 내용의 정리본이다.

# Part3_DB_NoSQL

### 목차

-  Step 1. NoSQL
-  Step 2. Dynamodb
-  Step 3. 데이터 채우기
-  Step 4. 데이터 조회하기

##  Step 1. NoSQL

### NoSQL(Not Only SQL)
초고용량 데이터 처리 등 성능에 특화된 목적을 위해 비관계형 데이터 저장소에 비구조적인 데이터를 저장하기 위한 분산 저장 시스템

### 필요성
- 유연성 
	- 현재는 데이터가 매우 많이 나오고 있으며 유형도 다양하기에 유연성을 가지고 데이터를 저장할 수 있음

- Scalability 
	- SQL DB는 vertically scalable인 컬럼 기준으로 나누는 형식이며 전체 특정 스펙들을 지정하여 사용 
	- NoSQL DB는 horizontally scalable


##  Step 2. Dynamodb

Amazon DynamoDB는 종합 관리형 NoSQL 데이터베이스 서비스로, 원활한 확장성과 함께 빠르고 예측 가능한 성능을 제공한다.

DynamoDB를 통해 데이터 규모에 관계없이 데이터를 저장 및 검색하고, 어떤 수준의 요청 트래픽이라도 처리할 수 있는 데이터베이스 테이블을 생성이 가능하다.

### 파티션(Partition)
데이터가 많아짐에 따라 데이터 매니지먼트, 퍼포먼스 등 다양한 이유로 데이터를 나누는 개념 
- vertical partition : 테이블을 더 작은 테이블로 나누는 작업으로써(컬럼으로 나눔) 정규화 후에도 경우에 따라 컬럼을 나누는 파티션 작업을 실행

- horizontal partition : NoSQL DB에서 사용하는 방법으로 schema / structure 자체를 카피하여 데이터 자체를 shared key로 분리 (컬럼을 유지한 상태로 나눔)

- 파티션 키 (Partition Key)
DB 입장에서 어떤 값을 기준으로 값을 서칭하고 추가할지에 대한 기본키 개념이다. 파티션 키를 artist_id로 설정하면 해당 컬럼으로 들어오는 데이터가 파티션 키로 된다.

- 솔트 키 (Sort key)
파티션 키가 중복되지만 다른 정보는 상이할 경우가 있다. 해당 경우에 원하는 값을 찾기 위한 서브 키 개념이다.

##  Step 3. 데이터 채우기

이제 AWS 에서 Dynamodb를 생성해보자. 우선 우리가 사용할 테이블은 top_tracks 이다.  파티션 키는 artist_id 이고 솔트 키는 id 로 설정한다.

![image](https://user-images.githubusercontent.com/59912557/89859159-52b2ed00-dbdb-11ea-9d3e-35dfd1877b65.png)

![image](https://user-images.githubusercontent.com/59912557/89859371-d240bc00-dbdb-11ea-8322-f52d6714f13e.png)


![image](https://user-images.githubusercontent.com/59912557/89859836-e638ed80-dbdc-11ea-9aed-7b80cadca1fd.png)

이렇게 프리티어로 생성해야 과금이 생기지 않는다. 자 이제 생성한 Dynamodb를 파이썬이랑 연결을 시켜보자

### 1) Boto3 library
 Python 애플리케이션, 라이브러리 또는 스크립트를 Amazon S3, Amazon EC2, Amazon DynamoDB 등 AWS 서비스와 통합 시켜주는 library이다.
```python
import sys
import os
import boto3
import requests
import base64
import json
import logging
import pymysql

def main():

    try:
	    # connect to python and dynamodb
        dynamodb = boto3.resource('dynamodb', region_name = 'ap-northeast-2', endpoint_url = 'http://dynamodb.ap-northeast-2.amazonaws.com')
    except:
        logging.error('could not connect to dynamodb')
        sys.exit(1)
	
	try:
        conn = pymysql.connect(host, user = username, passwd = password, db = database, port = port, use_unicode = True, charset = 'utf8')
        cursor = conn.cursor()
    except:
        logging.error('could not connect to RDS')
        sys.exit(1) # 1 == fail
	
	# get access_token
	headers = get_headers(client_id, client_secret)

	# use dynamodb table
	table = dynamodb.Table('top_tracks')
	
	# get artist_id
	cursor.excute('SELECT id FROM artists')
	
	# get top tracks
	# GET https://api.spotify.com/v1/artists/{id}/top-tracks
	for (artist_id, ) cursor.fetchall():
	
		 url = 'https://api.spotify.com/v1/artists/{}/top-tracks'.format(artist_id)
		 params = {
				 'country' : 'US'
			 }
			 
		 r = requests.get(url, params,headers)	  
		 raw = json.load(r.text)

	     for track in raw['tracks']:
		     data = {
				 'artist_id' : artist_id
			 } 
			 data.update(track)
			 
			 # dynamodb 'top_tracks' table에 데이터 추가
			 table.put_item(
				Item = data
			 )
			 
```
추가된 데이터는 AWS Dynamodb에서 확인할 수 있다
![image](https://user-images.githubusercontent.com/59912557/89860469-7deb0b80-dbde-11ea-97f3-ab37b6fb8455.png)

## Step 4. 데이터 조회하기

python을 이용해서 적재한 데이터를 조회해보자

```python
import sys
import os
import boto3
# 조건을 사용할 경우 추가
from boto3.dynamodb.conditions import Key, Attr

def main():

    try:
	    # connect to dynamodb
        dynamodb = boto3.resource('dynamodb', region_name = 'ap-northeast-2',
        endpoint_url = 'http://dynamodb.ap-northeast-2.amazonaws.com')
    except:
        logging.error('could not connect to dynamodb')
        sys.exit(1)
	
	# open 'top_tracks' table
    table = dynamodb.Table('top_tracks')
	
	# get_item : 행 조회하기
	response = table.get_item(
			# partition key and sort key
			Key = {
				'artist_id' : '00FQb4jTyendYWaN8pK0wa',
	            'id' : '0Oqc0kKFsQ6MhFOLBNZIGX'
				}
		)
	print(response)

	response = table.query(
		# eq = equal
		KeyConditionExpression = Key('artist_id').eq('00FQb4jTyendYWaN8pK0wa'),
		# Attr = column, gt = greater
		FilterExpression = Attr('popularity).gt(70)
	)
	
	print(response)
	print(len(response)) 
```
```python
{'Item': {'is_playable': True, 'duration_ms': Decimal('202192'), 'external_ids': {'isrc': 'GBUM71903894'}, 'uri': 'spotify:track:0Oqc0kKFsQ6MhFOLBNZIGX', 'name': "Doin' Time", 'album': {'total_tracks': Decimal('14'), 'images': [{'width': Decimal('640'), 'url': 'https://i.scdn.co/image/ab67616d0000b273db58320165892f952a6ddb3f', 'height': Decimal('640')}, {'width': Decimal('300'), 'url': 'https://i.scdn.co/image/ab67616d00001e02db58320165892f952a6ddb3f', 'height': Decimal('300')}, {'width': Decimal('64'), 'url': 'https://i.scdn.co/image/ab67616d00004851db58320165892f952a6ddb3f', 'height': Decimal('64')}], 'artists': [{'name': 'Lana Del Rey', 'href': 'https://api.spotify.com/v1/artists/00FQb4jTyendYWaN8pK0wa', 'id': '00FQb4jTyendYWaN8pK0wa', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/00FQb4jTyendYWaN8pK0wa'}, 'uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa'}], 'release_date': '2019-08-30', 'name': 'Norman Fucking Rockwell!', 'album_type': 'album', 'release_date_precision': 'day', 'href': 'https://api.spotify.com/v1/albums/5XpEKORZ4y6OrCZSKsi46A', 'id': '5XpEKORZ4y6OrCZSKsi46A', 'type': 'album', 'external_urls': {'spotify': 'https://open.spotify.com/album/5XpEKORZ4y6OrCZSKsi46A'}, 'uri': 'spotify:album:5XpEKORZ4y6OrCZSKsi46A'}, 'popularity': Decimal('76'), 'artists': [{'name': 'Lana Del Rey', 'href': 'https://api.spotify.com/v1/artists/00FQb4jTyendYWaN8pK0wa', 'id': '00FQb4jTyendYWaN8pK0wa', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/00FQb4jTyendYWaN8pK0wa'}, 'uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa'}], 'disc_number': Decimal('1'), 'href': 'https://api.spotify.com/v1/tracks/0Oqc0kKFsQ6MhFOLBNZIGX', 'track_number': Decimal('5'), 'external_urls': {'spotify': 'https://open.spotify.com/track/0Oqc0kKFsQ6MhFOLBNZIGX'}, 'artist_id': '00FQb4jTyendYWaN8pK0wa', 'preview_url': None, 'is_local': False, 'id': '0Oqc0kKFsQ6MhFOLBNZIGX', 'explicit': True, 'type': 'track'}, 'ResponseMetadata': {'RequestId': 'RP331CQAIMTKSRB9S1JDO6M2B7VV4KQNSO5AEMVJF66Q9ASUAAJG', 'HTTPStatusCode': 200, 'HTTPHeaders': {'x-amzn-requestid': 'RP331CQAIMTKSRB9S1JDO6M2B7VV4KQNSO5AEMVJF66Q9ASUAAJG', 'x-amz-crc32': '3537049350', 'content-type': 'application/x-amz-json-1.0', 'content-length': '2138', 'date': 'Tue, 11 Aug 2020 05:29:31 GMT'}, 'RetryAttempts': 0}}

{'Items': [{'is_playable': True, 'duration_ms': Decimal('202192'), 'external_ids': {'isrc': 'GBUM71903894'}, 'uri': 'spotify:track:0Oqc0kKFsQ6MhFOLBNZIGX', 'name': "Doin' Time", 'album': {'total_tracks': Decimal('14'), 'images': [{'width': Decimal('640'), 'url': 'https://i.scdn.co/image/ab67616d0000b273db58320165892f952a6ddb3f', 'height': Decimal('640')}, {'width': Decimal('300'), 'url': 'https://i.scdn.co/image/ab67616d00001e02db58320165892f952a6ddb3f', 'height': Decimal('300')}, {'width': Decimal('64'), 'url': 'https://i.scdn.co/image/ab67616d00004851db58320165892f952a6ddb3f', 'height': Decimal('64')}], 'artists': [{'name': 'Lana Del Rey', 'href': 'https://api.spotify.com/v1/artists/00FQb4jTyendYWaN8pK0wa', 'id': '00FQb4jTyendYWaN8pK0wa', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/00FQb4jTyendYWaN8pK0wa'}, 'uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa'}], 'release_date': '2019-08-30', 'name': 'Norman Fucking Rockwell!', 'album_type': 'album', 'release_date_precision': 'day', 'href': 'https://api.spotify.com/v1/albums/5XpEKORZ4y6OrCZSKsi46A', 'id': '5XpEKORZ4y6OrCZSKsi46A', 'type': 'album', 'external_urls': {'spotify': 'https://open.spotify.com/album/5XpEKORZ4y6OrCZSKsi46A'}, 'uri': 'spotify:album:5XpEKORZ4y6OrCZSKsi46A'}, 'popularity': Decimal('76'), 'artists': [{'name': 'Lana Del Rey', 'href': 'https://api.spotify.com/v1/artists/00FQb4jTyendYWaN8pK0wa', 'id': '00FQb4jTyendYWaN8pK0wa', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/00FQb4jTyendYWaN8pK0wa'}, 'uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa'}], 'disc_number': Decimal('1'), 'href': 'https://api.spotify.com/v1/tracks/0Oqc0kKFsQ6MhFOLBNZIGX', 'track_number': Decimal('5'), 'external_urls': {'spotify': 'https://open.spotify.com/track/0Oqc0kKFsQ6MhFOLBNZIGX'}, 'artist_id': '00FQb4jTyendYWaN8pK0wa', 'preview_url': None, 'is_local': False, 'id': '0Oqc0kKFsQ6MhFOLBNZIGX', 'explicit': True, 'type': 'track'}, {'is_playable': True, 'duration_ms': Decimal('300683'), 'external_ids': {'isrc': 'GBUM71903147'}, 'uri': 'spotify:track:2mdEsXPu8ZmkHRRtAdC09e', 'name': 'Cinnamon Girl', 'album': {'total_tracks': Decimal('14'), 'images': [{'width': Decimal('640'), 'url': 'https://i.scdn.co/image/ab67616d0000b273db58320165892f952a6ddb3f', 'height': Decimal('640')}, {'width': Decimal('300'), 'url': 'https://i.scdn.co/image/ab67616d00001e02db58320165892f952a6ddb3f', 'height': Decimal('300')}, {'width': Decimal('64'), 'url': 'https://i.scdn.co/image/ab67616d00004851db58320165892f952a6ddb3f', 'height': Decimal('64')}], 'artists': [{'name': 'Lana Del Rey', 'href': 'https://api.spotify.com/v1/artists/00FQb4jTyendYWaN8pK0wa', 'id': '00FQb4jTyendYWaN8pK0wa', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/00FQb4jTyendYWaN8pK0wa'}, 'uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa'}], 'release_date': '2019-08-30', 'name': 'Norman Fucking Rockwell!', 'album_type': 'album', 'release_date_precision': 'day', 'href': 'https://api.spotify.com/v1/albums/5XpEKORZ4y6OrCZSKsi46A', 'id': '5XpEKORZ4y6OrCZSKsi46A', 'type': 'album', 'external_urls': {'spotify': 'https://open.spotify.com/album/5XpEKORZ4y6OrCZSKsi46A'}, 'uri': 'spotify:album:5XpEKORZ4y6OrCZSKsi46A'}, 'popularity': Decimal('71'), 'artists': [{'name': 'Lana Del Rey', 'href': 'https://api.spotify.com/v1/artists/00FQb4jTyendYWaN8pK0wa', 'id': '00FQb4jTyendYWaN8pK0wa', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/00FQb4jTyendYWaN8pK0wa'}, 'uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa'}], 'disc_number': Decimal('1'), 'href': 'https://api.spotify.com/v1/tracks/2mdEsXPu8ZmkHRRtAdC09e', 'track_number': Decimal('7'), 'external_urls': {'spotify': 'https://open.spotify.com/track/2mdEsXPu8ZmkHRRtAdC09e'}, 'artist_id': '00FQb4jTyendYWaN8pK0wa', 'preview_url': None, 'is_local': False, 'id': '2mdEsXPu8ZmkHRRtAdC09e', 'explicit': False, 'type': 'track'}, {'is_playable': True, 'duration_ms': Decimal('236053'), 'external_ids': {'isrc': 'GBUM71301823'}, 'uri': 'spotify:track:2nMeu6UenVvwUktBCpLMK9', 'name': 'Young And Beautiful', 'album': {'total_tracks': Decimal('1'), 'images': [{'width': Decimal('640'), 'url': 'https://i.scdn.co/image/ab67616d0000b273d7fb3e4c63020039d1cff6b2', 'height': Decimal('640')}, {'width': Decimal('300'), 'url': 'https://i.scdn.co/image/ab67616d00001e02d7fb3e4c63020039d1cff6b2', 'height': Decimal('300')}, {'width': Decimal('64'), 'url': 'https://i.scdn.co/image/ab67616d00004851d7fb3e4c63020039d1cff6b2', 'height': Decimal('64')}], 'artists': [{'name': 'Lana Del Rey', 'href': 'https://api.spotify.com/v1/artists/00FQb4jTyendYWaN8pK0wa', 'id': '00FQb4jTyendYWaN8pK0wa', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/00FQb4jTyendYWaN8pK0wa'}, 'uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa'}], 'release_date': '2013-01-01', 'name': 'Young And Beautiful', 'album_type': 'single', 'release_date_precision': 'day', 'href': 'https://api.spotify.com/v1/albums/1D92WOHWUI2AGQCCdplcXL', 'id': '1D92WOHWUI2AGQCCdplcXL', 'type': 'album', 'external_urls': {'spotify': 'https://open.spotify.com/album/1D92WOHWUI2AGQCCdplcXL'}, 'uri': 'spotify:album:1D92WOHWUI2AGQCCdplcXL'}, 'popularity': Decimal('76'), 'artists': [{'name': 'Lana Del Rey', 'href': 'https://api.spotify.com/v1/artists/00FQb4jTyendYWaN8pK0wa', 'id': '00FQb4jTyendYWaN8pK0wa', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/00FQb4jTyendYWaN8pK0wa'}, 'uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa'}], 'disc_number': Decimal('1'), 'href': 'https://api.spotify.com/v1/tracks/2nMeu6UenVvwUktBCpLMK9', 'track_number': Decimal('1'), 'external_urls': {'spotify': 'https://open.spotify.com/track/2nMeu6UenVvwUktBCpLMK9'}, 'artist_id': '00FQb4jTyendYWaN8pK0wa', 'preview_url': None, 'is_local': False, 'id': '2nMeu6UenVvwUktBCpLMK9', 'explicit': False, 'type': 'track'}, {'is_playable': True, 'duration_ms': Decimal('248934'), 'external_ids': {'isrc': 'GBUM71903153'}, 'uri': 'spotify:track:3RIgHHpnFKj5Rni1shokDj', 'name': 'Norman fucking Rockwell', 'album': {'total_tracks': Decimal('14'), 'images': [{'width': Decimal('640'), 'url': 'https://i.scdn.co/image/ab67616d0000b273db58320165892f952a6ddb3f', 'height': Decimal('640')}, {'width': Decimal('300'), 'url': 'https://i.scdn.co/image/ab67616d00001e02db58320165892f952a6ddb3f', 'height': Decimal('300')}, {'width': Decimal('64'), 'url': 'https://i.scdn.co/image/ab67616d00004851db58320165892f952a6ddb3f', 'height': Decimal('64')}], 'artists': [{'name': 'Lana Del Rey', 'href': 'https://api.spotify.com/v1/artists/00FQb4jTyendYWaN8pK0wa', 'id': '00FQb4jTyendYWaN8pK0wa', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/00FQb4jTyendYWaN8pK0wa'}, 'uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa'}], 'release_date': '2019-08-30', 'name': 'Norman Fucking Rockwell!', 'album_type': 'album', 'release_date_precision': 'day', 'href': 'https://api.spotify.com/v1/albums/5XpEKORZ4y6OrCZSKsi46A', 'id': '5XpEKORZ4y6OrCZSKsi46A', 'type': 'album', 'external_urls': {'spotify': 'https://open.spotify.com/album/5XpEKORZ4y6OrCZSKsi46A'}, 'uri': 'spotify:album:5XpEKORZ4y6OrCZSKsi46A'}, 'popularity': Decimal('71'), 'artists': [{'name': 'Lana Del Rey', 'href': 'https://api.spotify.com/v1/artists/00FQb4jTyendYWaN8pK0wa', 'id': '00FQb4jTyendYWaN8pK0wa', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/00FQb4jTyendYWaN8pK0wa'}, 'uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa'}], 'disc_number': Decimal('1'), 'href': 'https://api.spotify.com/v1/tracks/3RIgHHpnFKj5Rni1shokDj', 'track_number': Decimal('1'), 'external_urls': {'spotify': 'https://open.spotify.com/track/3RIgHHpnFKj5Rni1shokDj'}, 'artist_id': '00FQb4jTyendYWaN8pK0wa', 'preview_url': None, 'is_local': False, 'id': '3RIgHHpnFKj5Rni1shokDj', 'explicit': True, 'type': 'track'}, {'is_playable': True, 'duration_ms': Decimal('214912'), 'external_ids': {'isrc': 'GBUM71304610'}, 'uri': 'spotify:track:6PUIzlqotEmPuBfjbwYWOB', 'name': 'Summertime Sadness (Lana Del Rey Vs. Cedric Gervais) - Cedric Gervais Remix', 'album': {'total_tracks': Decimal('1'), 'images': [{'width': Decimal('640'), 'url': 'https://i.scdn.co/image/ab67616d0000b273f8cdf8e85bc179a475d581ba', 'height': Decimal('640')}, {'width': Decimal('300'), 'url': 'https://i.scdn.co/image/ab67616d00001e02f8cdf8e85bc179a475d581ba', 'height': Decimal('300')}, {'width': Decimal('64'), 'url': 'https://i.scdn.co/image/ab67616d00004851f8cdf8e85bc179a475d581ba', 'height': Decimal('64')}], 'artists': [{'name': 'Lana Del Rey', 'href': 'https://api.spotify.com/v1/artists/00FQb4jTyendYWaN8pK0wa', 'id': '00FQb4jTyendYWaN8pK0wa', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/00FQb4jTyendYWaN8pK0wa'}, 'uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa'}, {'name': 'Cedric Gervais', 'href': 'https://api.spotify.com/v1/artists/4Wjf8diP59VmPG7fi4y724', 'id': '4Wjf8diP59VmPG7fi4y724', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/4Wjf8diP59VmPG7fi4y724'}, 'uri': 'spotify:artist:4Wjf8diP59VmPG7fi4y724'}], 'release_date': '2013-02-01', 'name': 'Summertime Sadness [Lana Del Rey vs. Cedric Gervais] (Cedric Gervais Remix)', 'album_type': 'single', 'release_date_precision': 'day', 'href': 'https://api.spotify.com/v1/albums/1fXwOvaqIdkhp5F3fiFbCv', 'id': '1fXwOvaqIdkhp5F3fiFbCv', 'type': 'album', 'external_urls': {'spotify': 'https://open.spotify.com/album/1fXwOvaqIdkhp5F3fiFbCv'}, 'uri': 'spotify:album:1fXwOvaqIdkhp5F3fiFbCv'}, 'popularity': Decimal('74'), 'artists': [{'name': 'Lana Del Rey', 'href': 'https://api.spotify.com/v1/artists/00FQb4jTyendYWaN8pK0wa', 'id': '00FQb4jTyendYWaN8pK0wa', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/00FQb4jTyendYWaN8pK0wa'}, 'uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa'}, {'name': 'Cedric Gervais', 'href': 'https://api.spotify.com/v1/artists/4Wjf8diP59VmPG7fi4y724', 'id': '4Wjf8diP59VmPG7fi4y724', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/4Wjf8diP59VmPG7fi4y724'}, 'uri': 'spotify:artist:4Wjf8diP59VmPG7fi4y724'}], 'disc_number': Decimal('1'), 'href': 'https://api.spotify.com/v1/tracks/6PUIzlqotEmPuBfjbwYWOB', 'track_number': Decimal('1'), 'external_urls': {'spotify': 'https://open.spotify.com/track/6PUIzlqotEmPuBfjbwYWOB'}, 'artist_id': '00FQb4jTyendYWaN8pK0wa', 'preview_url': None, 'is_local': False, 'id': '6PUIzlqotEmPuBfjbwYWOB', 'explicit': False, 'type': 'track'}, {'is_playable': True, 'duration_ms': Decimal('190066'), 'external_ids': {'isrc': 'USUM71912501'}, 'uri': 'spotify:track:6zegtH6XXd2PDPLvy1Y0n2', 'name': 'Don’t Call Me Angel (Charlie’s Angels) (with Miley Cyrus & Lana Del Rey)', 'album': {'total_tracks': Decimal('11'), 'images': [{'width': Decimal('640'), 'url': 'https://i.scdn.co/image/ab67616d0000b273c891137d2513ecd496e9152e', 'height': Decimal('640')}, {'width': Decimal('300'), 'url': 'https://i.scdn.co/image/ab67616d00001e02c891137d2513ecd496e9152e', 'height': Decimal('300')}, {'width': Decimal('64'), 'url': 'https://i.scdn.co/image/ab67616d00004851c891137d2513ecd496e9152e', 'height': Decimal('64')}], 'artists': [{'name': 'Various Artists', 'href': 'https://api.spotify.com/v1/artists/0LyfQWJT6nXafLPZqxe9Of', 'id': '0LyfQWJT6nXafLPZqxe9Of', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/0LyfQWJT6nXafLPZqxe9Of'}, 'uri': 'spotify:artist:0LyfQWJT6nXafLPZqxe9Of'}], 'release_date': '2019-11-01', 'name': "Charlie's Angels (Original Motion Picture Soundtrack)", 'album_type': 'album', 'release_date_precision': 'day', 'href': 'https://api.spotify.com/v1/albums/4NBuascXb3uK0mFUYuJ63f', 'id': '4NBuascXb3uK0mFUYuJ63f', 'type': 'album', 'external_urls': {'spotify': 'https://open.spotify.com/album/4NBuascXb3uK0mFUYuJ63f'}, 'uri': 'spotify:album:4NBuascXb3uK0mFUYuJ63f'}, 'popularity': Decimal('77'), 'artists': [{'name': 'Ariana Grande', 'href': 'https://api.spotify.com/v1/artists/66CXWjxzNUsdJxJ2JdwvnR', 'id': '66CXWjxzNUsdJxJ2JdwvnR', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/66CXWjxzNUsdJxJ2JdwvnR'}, 'uri': 'spotify:artist:66CXWjxzNUsdJxJ2JdwvnR'}, {'name': 'Miley Cyrus', 'href': 'https://api.spotify.com/v1/artists/5YGY8feqx7naU7z4HrwZM6', 'id': '5YGY8feqx7naU7z4HrwZM6', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/5YGY8feqx7naU7z4HrwZM6'}, 'uri': 'spotify:artist:5YGY8feqx7naU7z4HrwZM6'}, {'name': 'Lana Del Rey', 'href': 'https://api.spotify.com/v1/artists/00FQb4jTyendYWaN8pK0wa', 'id': '00FQb4jTyendYWaN8pK0wa', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/00FQb4jTyendYWaN8pK0wa'}, 'uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa'}], 'disc_number': Decimal('1'), 'href': 'https://api.spotify.com/v1/tracks/6zegtH6XXd2PDPLvy1Y0n2', 'track_number': Decimal('3'), 'external_urls': {'spotify': 'https://open.spotify.com/track/6zegtH6XXd2PDPLvy1Y0n2'}, 'artist_id': '00FQb4jTyendYWaN8pK0wa', 'preview_url': None, 'is_local': False, 'id': '6zegtH6XXd2PDPLvy1Y0n2', 'explicit': False, 'type': 'track'}, {'is_playable': True, 'duration_ms': Decimal('218287'), 'external_ids': {'isrc': 'GBUM71903510'}, 'uri': 'spotify:track:7MtVPRGtZl6rPjMfLoI3Lh', 'name': 'Fuck it I love you', 'album': {'total_tracks': Decimal('14'), 'images': [{'width': Decimal('640'), 'url': 'https://i.scdn.co/image/ab67616d0000b273db58320165892f952a6ddb3f', 'height': Decimal('640')}, {'width': Decimal('300'), 'url': 'https://i.scdn.co/image/ab67616d00001e02db58320165892f952a6ddb3f', 'height': Decimal('300')}, {'width': Decimal('64'), 'url': 'https://i.scdn.co/image/ab67616d00004851db58320165892f952a6ddb3f', 'height': Decimal('64')}], 'artists': [{'name': 'Lana Del Rey', 'href': 'https://api.spotify.com/v1/artists/00FQb4jTyendYWaN8pK0wa', 'id': '00FQb4jTyendYWaN8pK0wa', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/00FQb4jTyendYWaN8pK0wa'}, 'uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa'}], 'release_date': '2019-08-30', 'name': 'Norman Fucking Rockwell!', 'album_type': 'album', 'release_date_precision': 'day', 'href': 'https://api.spotify.com/v1/albums/5XpEKORZ4y6OrCZSKsi46A', 'id': '5XpEKORZ4y6OrCZSKsi46A', 'type': 'album', 'external_urls': {'spotify': 'https://open.spotify.com/album/5XpEKORZ4y6OrCZSKsi46A'}, 'uri': 'spotify:album:5XpEKORZ4y6OrCZSKsi46A'}, 'popularity': Decimal('72'), 'artists': [{'name': 'Lana Del Rey', 'href': 'https://api.spotify.com/v1/artists/00FQb4jTyendYWaN8pK0wa', 'id': '00FQb4jTyendYWaN8pK0wa', 'type': 'artist', 'external_urls': {'spotify': 'https://open.spotify.com/artist/00FQb4jTyendYWaN8pK0wa'}, 'uri': 'spotify:artist:00FQb4jTyendYWaN8pK0wa'}], 'disc_number': Decimal('1'), 'href': 'https://api.spotify.com/v1/tracks/7MtVPRGtZl6rPjMfLoI3Lh', 'track_number': Decimal('4'), 'external_urls': {'spotify': 'https://open.spotify.com/track/7MtVPRGtZl6rPjMfLoI3Lh'}, 'artist_id': '00FQb4jTyendYWaN8pK0wa', 'preview_url': None, 'is_local': False, 'id': '7MtVPRGtZl6rPjMfLoI3Lh', 'explicit': True, 'type': 'track'}], 'Count': 7, 'ScannedCount': 10, 'ResponseMetadata': {'RequestId': 'PJBFM6A54OI10KP9C94QIMFKSVVV4KQNSO5AEMVJF66Q9ASUAAJG', 'HTTPStatusCode': 200, 'HTTPHeaders': {'x-amzn-requestid': 'PJBFM6A54OI10KP9C94QIMFKSVVV4KQNSO5AEMVJF66Q9ASUAAJG', 'x-amz-crc32': '3599223101', 'content-type': 'application/x-amz-json-1.0', 'content-length': '16463', 'date': 'Tue, 11 Aug 2020 05:32:25 GMT'}, 'RetryAttempts': 0}}
7
```







