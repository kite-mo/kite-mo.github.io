---  
title:  "2. 데이터 수집 및 탐색"  
  
categories:  
 - Python
tags:  
 - Study, Python
 
---

# 2. 데이터 수집 및 탐색
### 목차
-  Step 0. 데이터 베이스 기초
-  Step 1. SQL 문을 활용한 python에 데이터 적재
-  Step 2. pandas를 이용한 데이터 적재
-  Step 3. 웹 데이터 적재
-  Step 4. library를 이용한 데이터 적재

## Step 0. 데이터 베이스 활용 - SQL


이번 포스팅은 데이터가 저장되어 있는 Database로 부터 SQL 언어를 사용하여 python으로 데이터를 불러오는 방법에 대하여 공부하겠다.

아래 사진을 통해 간단한 SQL의 개념에 대해서 숙지하면 되겠다.

![SQL](https://user-images.githubusercontent.com/59912557/78899313-29a91100-7ab0-11ea-9b6a-305c62aac2b6.PNG)

##  Step 1. SQL 문을 활용한 python에 데이터 적재

우선 사용할 데이터인 boston.db를 아래와 같은 폴더에 저장해두었다.
![image](https://user-images.githubusercontent.com/59912557/78899644-a2a86880-7ab0-11ea-9e1a-0d510c3f0b2e.png)

### (1) SQL 언어 사용 준비

```python
# SQL 언어를 쓰기 위한 library
import sqlite3 
sqlite_file = './data/boston.db' 
# database file을 향해 SQL과 python간의 통로를 만들어 줌
conn = sqlite3.connect(sqlite_file)
# 연결된 통로를 통한 현재 위치를 갱신
cur = conn.cursor()
```

### (2) 조회 그리고 적재

이제 SQL과 python을 연결 했으니, SQL 문을 이용하여 데이터를 조회하고 그리고 적재를 할 것이다. 

간단히 이해하면 **조회**는 사용자가 원하는 데이터를 꺼내오기 위해 SQL 문을 통하여 명령하는 것이고, **적재**는 명령을 통해 조회한 데이터를 실제로 쌓는 과정이다.

```python
# boston 테이블로부터 모든 필드값(*)을 5개의 행을 조회
cur.execute("SELECT * FROM boston LIMIT 5;")
# data 란 객체에 조회된 데이터를 저장
data = cur.fetchall()
print(data)
```
```python
[(0, 0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.09, 1.0, 296.0, 15.3, 396.9, 4.98, 24.0), (1, 0.02731, 0.0, 7.07, 0.0, 0.469, 6.421, 78.9, 4.9671, 2.0, 242.0, 17.8, 396.9, 9.14, 21.6), (2, 0.02729, 0.0, 7.07, 0.0, 0.469, 7.185, 61.1, 4.9671, 2.0, 242.0, 17.8, 392.83, 4.03, 34.7), (3, 0.03237, 0.0, 2.18, 0.0, 0.458, 6.998, 45.8, 6.0622, 3.0, 222.0, 18.7, 394.63, 2.94, 33.4), (4, 0.06905, 0.0, 2.18, 0.0, 0.458, 7.147, 54.2, 6.0622, 3.0, 222.0, 18.7, 396.9, 5.33, 36.2)]
```
SQL 문을 활용하여 조건을 추가할 수 도 있다.
```python
# ZN 필드에서 0.0 값 이상만 가지는 데이터 추출
cur.execute("SELECT ZN FROM boston WHERE ZN > 0.0;")
data = cur.fetchall()
```
### (3) 조회된 데이터를 데이터프레임으로 저장

함수 fetchall()로 적재한 데이터의 형태는 list 이다. 데이터를 다루기 쉽게, 보기 쉽게하기 위해 dataframe으로 저장이 가능하다.

```python
import pandas as pd
# SQL로 읽어온 데이터를 dataframe 형태로 저장
df = pd.read_sql_query("SELECT * FROM boston;", conn)
print(df.shape) # 데이터 형태 확인
print(df.head) # 상위 5개 행 확인
print(df.describe().transpose()) # 데이터 요약 통계량 확인
```
```python
(506, 15)

  record     CRIM    ZN  INDUS  CHAS  ...    TAX  PTRATIO       B  LSTAT  MEDV
0       0  0.00632  18.0   2.31   0.0  ...  296.0     15.3  396.90   4.98  24.0
1       1  0.02731   0.0   7.07   0.0  ...  242.0     17.8  396.90   9.14  21.6
2       2  0.02729   0.0   7.07   0.0  ...  242.0     17.8  392.83   4.03  34.7
3       3  0.03237   0.0   2.18   0.0  ...  222.0     18.7  394.63   2.94  33.4
4       4  0.06905   0.0   2.18   0.0  ...  222.0     18.7  396.90   5.33  36.2

         count        mean         std  ...        50%         75%       max
record   506.0  252.500000  146.213884  ...  252.50000  378.750000  505.0000
CRIM     506.0    3.593761    8.596783  ...    0.25651    3.647423   88.9762
ZN       506.0   11.363636   23.322453  ...    0.00000   12.500000  100.0000
INDUS    506.0   11.136779    6.860353  ...    9.69000   18.100000   27.7400
CHAS     506.0    0.069170    0.253994  ...    0.00000    0.000000    1.0000
NOX      506.0    0.554695    0.115878  ...    0.53800    0.624000    0.8710
RM       506.0    6.284634    0.702617  ...    6.20850    6.623500    8.7800
AGE      506.0   68.574901   28.148861  ...   77.50000   94.075000  100.0000
DIS      506.0    3.795043    2.105710  ...    3.20745    5.188425   12.1265
RAD      506.0    9.549407    8.707259  ...    5.00000   24.000000   24.0000
TAX      506.0  408.237154  168.537116  ...  330.00000  666.000000  711.0000
PTRATIO  506.0   18.455534    2.164946  ...   19.05000   20.200000   22.0000
B        506.0  356.674032   91.294864  ...  391.44000  396.225000  396.9000
LSTAT    506.0   12.653063    7.141062  ...   11.36000   16.955000   37.9700
MEDV     506.0   22.532806    9.197104  ...   21.20000   25.000000   50.0000
```
위에서 언급했다싶이, SQL 문을 통해 조건을 추가할 수 있다. 그래서 SQL 문이 길어질 경우 """ ~~~ """ 을 통해 여러 줄을 작성이 가능하다.

```python
df = pd.read_sql_query("""
						SELECT record, ZN, AGE, TAX FROM boston
						WHERE ZN > 0.0 and record > 250;
						""", conn)
print(df.head)
print(df.shape)
``` 
```python
   record    ZN   AGE    TAX
0     251  22.0   8.9  330.0
1     252  22.0   6.8  330.0
2     253  22.0   8.4  330.0
3     254  80.0  32.0  315.0
4     255  80.0  19.1  315.0

(66, 4) 
```

### (4) 통로 연결 끊기

원하는 작업이 끝나면, SQL과 python을 이어주는 연결 통로를 제거해야한다.

```python
conn.close()
```

##  Step 2. pandas를 이용한 데이터 적재

이번에는 pandas를 이용하여 만든 데이터프레임을 직접 Database에 저장하는 방법을 공부해보자.

우선 boston DB로부터 10개의 행을 불러온다
```python
sqlite_file = './data/boston.db'
conn = sqlite3.connect(sqlite_file)

df = pd.read_sql_query("SELECT * FROM boston LIMIT 10;", conn)
print(df.shape) #(10, 15)
```
### DB 테이블 생성
```python
# boston.db 파일에 df에 저장된 테이블을 저장시킨다.
# if_exists = replace : 중복된다면 대체해라
df.to_sql("boston_updated", conn, if_exists = "replace")
# 생성된 테이블로부터 데이터 불러오기
df_1 = pd.read_sql_query("SELECT * FROM boston_updated LIMIT 5;", conn)
print(df_1) # (5,15)
```
### 생성된 DB 테이블 확인
```python
cur = conn.cursor()
# 조회 후 적재
# # sqlite_master을 이용하여 Database 안에 무슨 table 있는지 확인 가능
cur.execute("SELECT * FROM sqlite_master WHERE type 'table;")
data = cur.fetechall()
print(data)
```
```python
[('table', 'boston', 'boston', 2, 'CREATE TABLE "boston" (\n"record" INTEGER,\n  "CRIM" REAL,\n  "ZN" REAL,\n  "INDUS" REAL,\n  "CHAS" REAL,\n  "NOX" REAL,\n  "RM" REAL,\n  "AGE" REAL,\n  "DIS" REAL,\n  "RAD" REAL,\n  "TAX" REAL,\n  "PTRATIO" REAL,\n  "B" REAL,\n  "LSTAT" REAL,\n  "MEDV" REAL\n)'),
 ('table', 'boston_updated', 'boston_updated', 19, 'CREATE TABLE "boston_updated" (\n"index" INTEGER,\n  "record" INTEGER,\n  "CRIM" REAL,\n  "ZN" REAL,\n  "INDUS" REAL,\n  "CHAS" REAL,\n  "NOX" REAL,\n  "RM" REAL,\n  "AGE" REAL,\n  "DIS" REAL,\n  "RAD" REAL,\n  "TAX" REAL,\n  "PTRATIO" REAL,\n  "B" REAL,\n  "LSTAT" REAL,\n  "MEDV" REAL\n)')]
```
### DB 테이블 제거
```python
# DROP TABLE = 테이블 제거
cur.execute("DROP TABLE 'boston_updated'")
cur.execute("SELECT * FROM sqlite_master WHERE type = 'table';")
data = cur.fetchall()
print(data)
```
```python
# boston_updated 테이블이 제거되었음을 확인 가능
[('table', 'boston', 'boston', 2, 'CREATE TABLE "boston" (\n"record" INTEGER,\n  "CRIM" REAL,\n  "ZN" REAL,\n  "INDUS" REAL,\n  "CHAS" REAL,\n  "NOX" REAL,\n  "RM" REAL,\n  "AGE" REAL,\n  "DIS" REAL,\n  "RAD" REAL,\n  "TAX" REAL,\n  "PTRATIO" REAL,\n  "B" REAL,\n  "LSTAT" REAL,\n  "MEDV" REAL\n)')]
# 연결 끊기
conn.close()
```

##  Step 3. 웹 데이터 불러오기

데이터가 너무 방대한 파일을 폴더에 넣기 부담스러울 경우, url을 이용하여 데이터를 불러올 수 있다.

```python
url="https://raw.githubusercontent.com/cs109/2014_data/master/countries.csv"
# url도 read_csv로 읽을 수 있다.
df = pd.read_csv(url) 
print(df.head())
```
```python
    Country  Region
0   Algeria  AFRICA
1    Angola  AFRICA
2     Benin  AFRICA
3  Botswana  AFRICA
4   Burkina  AFRICA
```
## Step 4. library를 이용한 데이터 불러오기

파이썬에서는 dataset을 제공하는 여러 library들이 존재한다.

### sklearn
```python

from sklearn.datasets import load_iris
dataset = load_iris()
df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
# species 열 생성
df['species'] = dataset.target
print(df.head())
```
```python
   sepal length (cm)  sepal width (cm)  ...  petal width (cm)  species
0                5.1               3.5  ...               0.2        0
1                4.9               3.0  ...               0.2        0
2                4.7               3.2  ...               0.2        0
3                4.6               3.1  ...               0.2        0
4                5.0               3.6  ...               0.2        0
```

### Seaborn
```python
import seaborn as sns
df = sns.load_dataset("flights") # flights 라는 data-set load
print(df.head())
```
```python
   year     month  passengers
0  1949   January         112
1  1949  February         118
2  1949     March         132
3  1949     April         129
4  1949       May         121
```

