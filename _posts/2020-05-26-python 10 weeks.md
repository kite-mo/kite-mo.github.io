---  
title:  "8. 군집분석 - (2)"  
  
categories:  
 - Python
tags:  
 - Study, Python
---

# 8. 군집분석 - (2)

### 목차
-  Step 1. DBSCAN
-  Step 2. Spectral Clustering

군집분석에는 데이터 형태에 따라 크게 두 가지 종류의 분석방법으로 나뉜다.

- Compactness : 데이터들이 덩어리 형태로 군집을 띄는 경우 앞서 배웠던 분석 방법인
	- K-means clustering, Hierarchical clustering 

- Connectivity : 군집을 구성할 경우 데이터가 연결되어 있는 형태를 가지고 군집분석 하는경우
	- DBSCAN

- 두 가지 경우에도 모두 분류가 잘되는 
	- Spectral Clustering 

## Step 1. DBSCAN

DBSCAN의 분석 과정이 어떻게 이루어지는지 공부해보자.

간단히 소개하면 epsilon이라는 중심으로부터 원의 반지름 길이를 설정한다. 해당 원 안에 최소 포함 점의 개수인 minpoint를 설정하여 원 안에 들어오는 데이터들을 군집으로 묶어나가는 과정이다.

그림을 보며 자세히 공부를 해보자

![DBSCAN_설명](https://user-images.githubusercontent.com/59912557/84112543-e9d8b880-aa63-11ea-908a-688691fc3ead.PNG)

우선 랜덤으로 core point를 정한 뒤에, epsilon 안에서 포함되는 minpoint 개수만큼의 데이터를 군집을 형성한다.

![DBSCAN_설명2](https://user-images.githubusercontent.com/59912557/84112546-ea714f00-aa63-11ea-9281-c8499c89cc56.PNG)

그 다음 포함되 어있던 점으로 중심을 옮겨 반지름안에 포함되는 데이터의 개수를 카운트 해본다. minpoint 보다 적을 경우 해당 점은 borderpoint 라고 정의한다.

![DBSCAN_설명3](https://user-images.githubusercontent.com/59912557/84112549-ea714f00-aa63-11ea-9bbe-f9b20de62619.PNG)

다시 또 중심을 옮겨가 원을 형성 할 때, 반지름 안에 corepoint를 포함한다면 해당 점도 같은 군집으로 형성한다.

![DBSCAN_설명4](https://user-images.githubusercontent.com/59912557/84112532-e7765e80-aa63-11ea-9469-c40694ba416a.PNG)

이렇게 원의 중심을 옮겨가며 군집을 확장해간다.

![DBSCAN_설명5PNG](https://user-images.githubusercontent.com/59912557/84112536-e8a78b80-aa63-11ea-8dfa-7b5e477d9dbc.PNG)

그러나 위 그림처럼 아무 데이터도 포함하지 않은 경우가 생기는데, 이런 경우는 다른 군집으로 설정한다. 이런 방식으로 군집을 확장하기 때문에 데이터의 형태에 따라 군집을 설정할 때 매우 유용하다. 아래와 같이 말이다.

![DBSCAN_설명8](https://user-images.githubusercontent.com/59912557/84112541-e9d8b880-aa63-11ea-82cc-2c58f27386e5.PNG)

이제 이론을 익혔으니, 코딩을 통해 실습을 해보자

```python
import pandas as pd
import seaborn as sns

sns.set_context("paper", font_scale=1.5)
sns.set_style("white")

# 군집 분석을 위한 가상 자료 생성

from sklearn import datasets

def make_blobs():
	n_samples = 1500
	blobs = datasets.make_blobs(n_samples = n_samples,
								centers = 5,
								cluster_std = [3, 0.9, 1.9, 1.9, 1.3],
								randoms_states = 51)
								
	# array 형태이기에 dataframe으로 변경						
	df = pd.DataFrame(blobs[0], columns = ['Feature_1', 'Feature_2'])
	df.index.name = 'record'
	return df
	
df = make_blobs()
sns.lmplot(x = 'Feature_2', y = 'Feautre_1', data = df, fit_reg = False)
```
![0](https://user-images.githubusercontent.com/59912557/84113113-f4e01880-aa64-11ea-9771-e4aaf363e39c.png)

그래프로 그려본 결과 덩어리 형태의 모습을 띄는 compactness 유형의 데이터이다.

connectivity에 적합한 DBSCAN을 적용을 시켜보자

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps = 0.5, min_samples =5, metric = 'euclidean')
dbscan.fit(df)

df['DBSCAN Cluster Labels'] = dbscan.labels_

sns.lmplot(x = 'Feature_2', y = 'Feature_1', 
           hue = "DBSCAN Cluster Labels", data = df, fit_reg = False)
```
![1](https://user-images.githubusercontent.com/59912557/84113115-f4e01880-aa64-11ea-9312-7d97eb92b408.png)

그래프를 그려본 결과 총 21개의 군집으로 나뉘어짐을 확인했다. 이렇듯 DBSCAN은 덩어리 형태의 데이터에 적용하기엔 부적합하다. 

그럼 이젠 그래프가 특정 형태를 띄는 connectivity 경우를 적용시켜보자

```python
# connectivity의 데이터셋 생성
def make_moons():
	moons = datasets.make_moons(n_samples = 200, noise = 0.05, random_states = 0)
	df = pd.DataFrame(moons[0], columns = ['Feature_1', 'Feature_2']
	df.index.name = 'record'

	return df

moons = make_moons()
sns.lmplot(x = 'Feature_2', y = 'Feature_1', data = moons, fit_reg = False)
```
![2](https://user-images.githubusercontent.com/59912557/84113116-f578af00-aa64-11ea-87c3-18811dd07cd2.png)

그래프로 그려보니 이전 데이터와는 다르게 특정 형태를 띄는 데이터 분포를 확인할 수 있다. 그러나 x,y 축의 범위가 다르기에 이전에 배웠던 표준화를 실행해보자.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
moons_scaled = scaler.fit_transform(moons)

moons_scaled = pd.DataFrame(moons_scaled, columns = ['Feature_1', 'Feature_2']

sns.lmplot(x = 'Feature_2', y = 'Feature_1', data = moons_scaled, fit_reg = False)
```
![3](https://user-images.githubusercontent.com/59912557/84113118-f6114580-aa64-11ea-8ca0-a7f99f5baa88.png)

x, y축의 범위가 같아짐을 확인했다. 이제 해당 데이터로 DBSCAN을 적용시켜보자

```python
dbscan = DBSCAN(eps = 0.5, min_samples = 5, metric = 'euclidean')
dbscan.fit(moons_scaled)

moons_scaled['DBSCAN Cluster Labels'] = dbscan.labels_
sns.lmplot(x = 'Feature_2', y = 'Feature_1', hue = 'DBSCAN Cluster Labels', data = moons_scaled, fit_reg = False)
```

![4](https://user-images.githubusercontent.com/59912557/84113102-f27dbe80-aa64-11ea-969a-8562ee15c967.png)

초승달 형태의 군집을 형성한 결과를 확인했다. 이처럼 DBSCAN은 특정 형태의 모습을 띄는 데이터에 적합하다. 그렇담 compactness data에 적합한 k-means 적용한 경우는 어떨까?

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2)
kmeans.fit(moons_scaled)
moons_scaled['KMeans Cluster Labels'] = kmeans.labels_

sns.lmplot(x = 'Feature_2', y = 'Feature_1', 
           hue = "KMeans Cluster Labels", data = moons_scaled, fit_reg = False)
```
 ![5](https://user-images.githubusercontent.com/59912557/84113105-f3165500-aa64-11ea-87c1-6d65674e64c9.png)
 
위 결과와는 다르게 가운데 부분이 이상하게 나뉨을 확인할 수 있다. 이처럼 데이터에 따라서 군집분석 방법을 다양하게 적용할 수 있음을 알 수 있다.

이번엔 iris 데이터를 이용해서 군집 방법론들을 적용시켜보자

```python
iris = datasets.load_iris()
# species df 생성
species = pd.DataFrame(iris.target)
species.columns = ['species']

# iris attribute df 생성
data = pd.DataFrame(iris.data)
data.columns = ['Sepal.length','Sepal.width','Petal.length','Petal.width']

# petal 속성만 뽑기
petal = data.iloc[:, 2:]
```
petal 데이터를 이용해서 K-means 군집화 수행
```python
kmeans = KMeans(n_clusters = 2)
kmeans.fit(petal)
petal['KMeans Cluster Labels'] = kmeans.labels_

sns.lmplot(x = 'Petal.length', y = 'Petal.width', 
           hue = "KMeans Cluster Labels", data = petal, fit_reg = False)
```
![6](https://user-images.githubusercontent.com/59912557/84113107-f3aeeb80-aa64-11ea-97c1-5c7b1e414051.png)

대부분 군집 분류가 잘되었지만 이상하게 군집이 분류된 한점이 보인다. 그럼 DBSCAN을 적용시켜서 결과를 확인해보자

```python
# eps 를 0.5에서 0.9로 변경해봤다
dbscan = DBSCAN(eps = 0.9, min_samples = 5, metric = 'euclidean')
dbscan.fit(petal)

petal['DBSCAN Cluster Labels'] = dbscan.labels_

sns.lmplot(x = 'Petal.length', y = 'Petal.width', 
           hue = "DBSCAN Cluster Labels", data = petal, fit_reg = False)
``` 

![7](https://user-images.githubusercontent.com/59912557/84113109-f4478200-aa64-11ea-98a9-5e1b30b7f0b9.png)

DBSCAN 군집화 결과는 매우 깔끔하게 분류되었다. 

## Step 2. Spectral Clustering

Spectral clustering은 그래프 기반 군집 분석이다. 큰 개념으론 주성분 분석을 진행 후 k-mean clustering을 진행한다.

만약 변수가 5개인 경우, 5x5 유사도 행렬을 생성하고 해당 유사도 행렬을 기반으로 주성분 분석을 실행한다. 추출된 주성분에 의한 자료 구성 후 k-means clustering을 진행한다.

자세한 개념은 아래 링크를 참고하면 좋을듯 하다.

[https://ratsgo.github.io/machine%20learning/2017/04/27/spectral/](https://ratsgo.github.io/machine%20learning/2017/04/27/spectral/)

위에서 사용했던 compactness와 connectivity data 둘다 해당 방법론을 적용해 군집이 적절하게 생성되는지 확인해보자

```python
from sklearn.cluster import SpectralClustering

# compactness
df = make_blobs()

# assign_labels : 군집화 전략 선택
# n_init : 다양한 중심에서 kmeans algorithm이 반복되는 횟수
# n_neighbors : 인접행렬 생성시 몇개의 neighbor을 이용할지
clus = SpectralClustering(n_clusters = 5, random_state =42,
						  assing_labels = 'kmeans', n_init = 10,
						  affinity = 'nearest_neighbors', n_neighbors = 10)
							
clus.fit(df)
df['Spectral Cluster Labels'] = clus.labels_

sns.lmplot(x = 'Feature_2', y = 'Feature_1', 
           hue = "Spectral Cluster Labels", data=df, fit_reg=False)
```
![9](https://user-images.githubusercontent.com/59912557/84118441-f82bd200-aa6d-11ea-83b2-a8ddd35b3326.png)

총 5개의 군집으로 분류가 잘 되었음을 확인할 수 있다.

```python
# connectivity
df = make_moons()

clus = SpectralClustering(n_clusters = 5, random_state =42,
						  assing_labels = 'kmeans', n_init = 10,
						  affinity = 'nearest_neighbors', n_neighbors = 10)
							
clus.fit(df)
df['Spectral Cluster Labels'] = clus.labels_

sns.lmplot(x = 'Feature_2', y = 'Feature_1', 
           hue = "Spectral Cluster Labels", data=df, fit_reg=False)
```
![10](https://user-images.githubusercontent.com/59912557/84118445-f95cff00-aa6d-11ea-8c74-cd25a1784c6d.png)

마찬가지로 connectivity data 경우에도 군집화가 잘되었음을 확인할 수 있다.

그러나 해당 방법론은 계산 과정이 상당히 복잡하기에 데이터 용량이 거대한 경우에는 적용하기가 쉽지 않다.

