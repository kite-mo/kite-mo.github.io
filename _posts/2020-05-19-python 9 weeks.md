---  
title:  "7. 군집분석 - (1)"  
  
categories:  
 - Python
tags:  
 - Study, Python
---

# 7. 군집분석 - (1)

### 목차
-  Step 1. K-means clustering
-  Step 2. Hierarchical clustering
-  Step 3. Dendrogram을 이용한 그룹화 특징 파악

군집 분석은 각 개체의 유사성을 측정하여 유사성이 높은 대상 집단을 분류하고, 군집에 속한 개체들의 유사성과 서로 다른 군집에 속한 개체간의 상이성을 규명하는 통계 분석 방법이다.

군집 분석에서 군집을 분류하는 가장 기본적 가정은 군집 내에 속한 객체들의 특성은 동질적이고, 서로 다른 군집에서 속한 개체들간의 특성은 서로 이질적이 되도록 각각의 객체를 분류해야 한다는 것이다. 따라서 군집 분석의 알고리즘은 군집 내 구성원의 동질성과 군집 간 구성원의 이질성을 최대하는 방법이다.

크게 계층적 군집 분석과 비 계층적 군집 분석으로 나뉘며, 이번 시간엔 비 계층적 군집 분석의 대표적방법론인 K-means clustering과 계층적 군집 분석에대해 공부해보자.

## Step 1. K-means clsutering

비 계층적 군집 분석은 구하고자 하는 군집의 수를 정한 상태에서 설정된 군집의 중심에 가장 가까운 개체를 하나씩 포함해 가는 방식으로 군집을 형성해가는 방법이다.

이제 예제 코드를 기반으로 예시들며 공부해보도록 하자.

```python
import pandas pd
import seaborn as sns

# 데이터들의 산포를 확인하기 위해 plot을 그릴 도화지 설정
sns.set_context('paper', font_scale =1.5
sns.set_style('white')

# data 생성을 위해 import
from sklearn import datasets

# 군집화 예제 데이터 생성 함수
def make_blobs():
	n_samples = 1500
	
	# 분류용 가상 데이터 생성
	# n_samples : 생성 데이터 개수 설정
	# centers : 중심 개수 설정
	# cluster_std : 중심으로부터 데이터 표준편차
	# random_state : 난수 seed 설정
	blobs = datasets.make_blobs(n_samples = n_samples, centers = 5,
								cluster_std = [3, 0.9, 1.9, 1.9, 1.3],
								random_state = 51)
	# array 형태이기에 [0] 인덱싱
	df = pd.DataFrame(blobs[0], columns = ['Feature_1', 'Feature_2'])
	df.index.name = 'record'
	return df

df = make_blobs()
print(df.head(10))
```
```python
        Feature_1  Feature_2
record                      
0       11.492294 -10.236187
1        4.376245  -9.152790
2       -2.193675   3.212265
3       -2.976039   3.037043
4       -2.963703   2.336960
5       -8.763687   9.609996
6        2.229265  -1.112749
7       -5.253198  10.006150
8       -2.502068   2.312068
9       -5.160658  -0.420058
```
임의로 생성한 데이터의 그래프를 그려보자

```python
sns.lmplot(x = 'Feature_2', y = 'Feature_1', data = df, fir_reg =False)
```

![1](https://user-images.githubusercontent.com/59912557/84100444-d9feab80-aa46-11ea-8483-feddc5320a3a.png)

뭉쳐있는 군집이 눈으로도 대강 확인이 가능하게 데이터가 생성됨을 확인할 수 있다. 이제 K-means 알고리즘을 이용해서 군집화를 진행해보자

```python
from sklearn.cluster import KMeans

# 정확히 몇개의 군집이 존재하는지 모르기에 임의로 값을 설정하여 진행한다

### k = 2
# tol : 목적함수의 변화값 임계치
# max_iter : 최대 반복수
clus = Kmeans(n_clusters = 2, tol = 0.0004, max_iter = 300)
# fit() : 데이터를 모델에게 학습
# transform() : fitted model로부터 output 반환
clus.fit(df)
# 군집 번호 열 생성
df['K-means Cluster Labels'] = clus.lables_
df.head()

sns.lmplot(x = 'Feature_2',  y = 'Feature_1', hue = 'K-means Cluster Labels', data = df, fit_reg = False)
```
```python
        Feature_1  Feature_2  K-means Cluster Labels
record                                              
0       11.492294 -10.236187                       1
1        4.376245  -9.152790                       1
2       -2.193675   3.212265                       0
3       -2.976039   3.037043                       0
4       -2.963703   2.336960                       0
```
![2](https://user-images.githubusercontent.com/59912557/84100446-d9feab80-aa46-11ea-99c0-651c4c438ec9.png)

위와 같이 군집이 2개가 생성됨을 확인할 수 있다. 그럼 예제 데이터 생성시 5개의 군집이 생기게 만들었으니 k = 5로 설정해서 진행해보자
```python
k = 5
clus = Kmeans(n_clusters = k, tol = 0.0004, max_iter = 300)
clus.fit(df)

df['K-means Cluster Labels'] = clus.labels_
print(df.head())

sns.lmplot(x = 'Feature_2', y = 'Feature_1', 
           hue = "K-means Cluster Labels", data = df, fit_reg = False)
```
```python
 Feature_1  Feature_2  K-means Cluster Labels
record                                              
0       11.492294 -10.236187                       2
1        4.376245  -9.152790                       2
2       -2.193675   3.212265                       0
3       -2.976039   3.037043                       0
4       -2.963703   2.336960                       0
```

![5](https://user-images.githubusercontent.com/59912557/84100449-db2fd880-aa46-11ea-85ab-f8a88b5f8a48.png)

위와 같이 군집이 5개로 나뉘는 결과를 확인할 수 있다. 

우리는 이전에 군집이 5개인 데이터를 생성했기에 최적의 군집을 알았지만, 알지 못하는 경우에는 어떻게 결정하면 좋을까

### (1) 적절한 군집 수 결정

적절한 군집 수를 결정하기 위해 군집의 평가 측도인 silhouette score가 존재한다. 보통 해당 점수가 1에 가까울 수록 가장 적절한 군집 수라고 보통 경험적으로 판단한다

```python
from sklearn import metrics

n_clusters = [2,3,4,5,6,7,8]
for k in n_clusters:
	kmeans = KMeans(n_cluster = k, random_state = 42).fit(df)
	cluster_lables = kmeans.labels_
	S = metrics.silhoutte_score(df, cluster_labels)
	print("n_clusters = {:d}, silhoutte score {:1f}".format(k,S))
```
```python
n_clusters = 2, silhouette score 0.427403
n_clusters = 3, silhouette score 0.456512
n_clusters = 4, silhouette score 0.541098
n_clusters = 5, silhouette score 0.582869
n_clusters = 6, silhouette score 0.558668
n_clusters = 7, silhouette score 0.556478
n_clusters = 8, silhouette score 0.512264
```
위 결과로 군집 수가 5개인 경우 silhoutte score가 1에 가장 가깝기에 적합한 군집수임을 확인할 수 있다

## Step 2. Hierarchical clsutering

이번엔 계층적 군집 분석에 대해 공부해보자. 비계층적 군집 분석과 차이점은 군집이 형성되는 과정을 확인할 수 있다는 점이다. 바로 덴드로그램을 통해서다. 

개별 데이터간의 거리에 의하여 가장 가까이 있는 대상드로부터 시작하여 결합해 감으로써 나무 모양의 계층구조를 형성해가는 방법으로 덴드로그램을 그려줌으로써 군집의 형성 과정을 확인이 가능하다

예제 코드를 통해 해당 과정을 공부해보자

```python
# 과정을 보여줄 예제 데이터 생성
data = {'x' : [1,2,2,4,5], 'y' : [1,1,4,3,4]}
data = pd.DataFrame(data)

sns.lmplot(x = 'x', y = 'y', data = data, fit_reg = False)
```

![7](https://user-images.githubusercontent.com/59912557/84100441-d8cd7e80-aa46-11ea-892c-0326dfa5adf7.png)

이제 데이터간의 거리를 이용하여 군집을 묶을건데 계층적 군집 분석에서 사용되는 거리측도는 아래와 같이 여러가지가 존재한다.

![중심과 각 점들의 거리 측도](https://user-images.githubusercontent.com/59912557/84103798-08808480-aa4f-11ea-95dd-d18ecd897555.PNG)

우리는 가장 기본적으로 사용되는 유클리디안 거리를 이용하여 분석을 해보려한다.

```python
# 거리를 구하기 위한 library
from scipy.spatial.distance import pdist, squrefrom

distance = pdist(data.values, metric = 'euclidean') ** 2

# 각 데이터간 거리를 쉽게 파악하기 위해 정방행렬로 표현
dist_matrix = sqaureform(distance)
print(dist_matrix)
```
```python
[[ 0.  1. 10. 13. 25.]
 [ 1.  0.  9.  8. 18.]
 [10.  9.  0.  5.  9.]
 [13.  8.  5.  0.  2.]
 [25. 18.  9.  2.  0.]]
```
데이터간 거리를 확인해본 결과 첫번째, 두번째 데이터가 가장 가깝기에 하나의 군집으로 묶는다

```python
data['cluster'] = [0,0,1,2,3]
# 묶인 군집을 기반으로 군집화를 계속 진행하기 위함
data_0 = data.groupby('cluster').mean()
print(data_0.values)
```
```python
[[1.5 1. ]
 [2.  4. ]
 [4.  3. ]
 [5.  4. ]]
```
군집 0으로 묶인 x,y 평균이 가장 작음을 확인할 수 있다. 계속 순차적으로 진행해보자

```python
distances = pdist(data_0.values, metric = 'euclidea') ** 2
dist_matrix = squareform(distances)
print(dist_matrix)
```
```python
[[ 0.    9.25 10.25 21.25]
 [ 9.25  0.    5.    9.  ]
 [10.25  5.    0.    2.  ]
 [21.25  9.    2.    0.  ]]
```
 세 번째와 네 번째 데이터의 거리가 가장 가까움을 확인했다

```python
data['cluster'] = [0,0,1,2,2]
data_0 = data0.groupby('cluster').means()

# 다음 단계 진행
distances = pdist(data0.values, metric = 'euclidean')**2
dist_matrix = squareform(distances)
print(dist_matrix)
```
```python
[[ 0.    9.25 15.25]
 [ 9.25  0.    6.5 ]
 [15.25  6.5   0.  ]]
```
두 번째와 세 번째 데이터가 가까움을 확인했다

```python
data['cluster] = [0,0,1,1,1]
```
이렇듯 계층적 군집 분석 방법을 이용해 최종 두개의 군집으로 묶이는 과정을 확인했다. 이번엔 library를 사용하여 분석해보자

```python
from scipy.cluster.hierarchy import linkage, dendrogram

data = {'x' : [1,2,2,4,5], 'y' : [1,1,4,3,4]}
data = pd.DataFrame(data)

z = linkage(data, 'single')
dendrogram(z)
```
![8](https://user-images.githubusercontent.com/59912557/84104994-fbb16000-aa51-11ea-84af-507159462c24.png)

위에서 하나하나 순차적으로 계산했던 과정이 덴드로그램을 통해 직관적으로 이해가 된다.

다만 계산의 수가 많다면 결과 해석이 복잡해지는 단점을 가진다. 아래와 같이 말이다.

![hca_dendrogram](https://user-images.githubusercontent.com/59912557/84105143-6793c880-aa52-11ea-9148-832734b7a6a2.jpg)

그럼 이젠 K-means clustering에서 쓰였던 데이터셋을 이용하여 계층적 군집 분석 방법에 적용시켜보자. 그 후에 어떤 군집 방법론이 더 뛰어난지도 확인할 거다.

```python
from sklearn.cluster impot AgglomerativeClustering

# affinity는 similiarity와 같은 의미이며 데이터끼리의 거리 계산 측도 파라미터이다
# linkage = 연결에 사용되는 거리 측도
clus = AgglomerativeClustering(n_clusters =5, affinity = 'euclidean', linkage = 'ward')

clus.fit(df)
df['HCA Cluster Labels'] = clus.labels_

n_clusters = [2,3,4,5,6,7,8]

for num in n_clusters:
	HCA = AgglomerativeClustering(n_clusters =5, affinity = 'euclidean', linkage = 'ward')
	Cluster_labels = HCA.fit_predict(df)
	S = metrics.silhouette_score(df, cluster_labels)
	print('n_clusters = {:d}, silhoutte score = {:1f}'.format(k,S))
```
```python
n_clusters = 2, silhouette score 0.480602
n_clusters = 3, silhouette score 0.462736
n_clusters = 4, silhouette score 0.559403
n_clusters = 5, silhouette score 0.600321
n_clusters = 6, silhouette score 0.567600
n_clusters = 7, silhouette score 0.563992
n_clusters = 8, silhouette score 0.510573
```
계층적 군집 분석을 사용한 경우, 군집 수가 5개 일때 k-means의 실루엣 척도보다 더 높은 점수가 나왔다

