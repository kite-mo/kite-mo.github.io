---  
title:  "6. 차원 축소법 - 주성분 분석(PCA)"  
  
categories:  
 - Python
tags:  
 - Study, Python
---

# 6. 차원 축소법 - 주성분 분석(PCA)
### 목차
-  Step 1. 주성분 분석 - 임의 데이터 적용
-  Step 2. 주성분 분석 - sklearn Cancer 데이터 적용
-  Step 3. 주성분 분석 - sklearn Iris 데이터 적용

## Step 1. 주성분 분석 - 임의 데이터 적용
차원 축소(변환) 방법 중 주성분 분석은 통계학에서 제안된 방법이다. 보통 우리가 분석하는 데이터는 많은 변수가 존재한다. 예를들어 30개의 변수가 있으면 30 차원으로 존재한다 말한다. 30 차원을 2 차원으로 축소시켜 사람이 인지할 수 있는 평면상으로 보여주는 것이 차원 축소의 개념이다. 

임의 데이터를 이용하여 주성분 분석(PCA)의 기본적인 개념을 알아보자 

```python
from sklearn.decomposition import PCA
import matplolib.pyplot as plt
import numpy as np

# 일정한 난수를 뽑기 위해 seed 설정
rnd = np.random.RandomState(5)
# 300개의 관측치를 가지는 정규분포를 따르는 2개의 변수 생성
X_ = rnd.normal(size = (300, 2))
print(X_[:10])
print(np.var(X_, axis = 0)
# c : 값에 따른 색 그라데이션, s = 점의 크기 
plt.scatter(X_[:,0], X_[:,1], c = X_[:,0], linewidths = 0, s = 60, cmap = 'viridis')
```
```python
[[ 0.44122749 -0.33087015]
 [ 2.43077119 -0.25209213]
 [ 0.10960984  1.58248112]
 [-0.9092324  -0.59163666]
 [ 0.18760323 -0.32986996]
 [-1.19276461 -0.20487651]
 [-0.35882895  0.6034716 ]
 [-1.66478853 -0.70017904]
 [ 1.15139101  1.85733101]
 [-1.51117956  0.64484751]]

# 1에 가까운 분산 결과 
[0.96498287 0.96062767]
```
0을 중심으로 분포 형태를 띄는 정규분포 모양임을 확인할 수 있다.
아래 그림을 살펴보면 현재 변수가 x,y 축에 위치하기에 해당 기준으로 분산이 모두 큼을 확인할 수 있다
![1](https://user-images.githubusercontent.com/59912557/82978161-db7bad00-a01e-11ea-80a9-fef4ccd6b841.png)

이제 해당 데이터들을 기울기, 절편이 존재하도록 선형변환을 진행한다. 
- Y = BX + a
- X: 300*2 행렬 : 현재 가지고 있는 데이터
- B: 2*2 기울기 행렬
- a: 절편 벡터

```python
# X_ 행렬 형태와 맞춰주기 위해 2,2 생성
B = rnd.normal(size = (2,2))
a = rnd.normal(size = 2)

# np.dot : 행렬의 곱셈 연산
X_blob = np.dot(X_, B) + a
print(X_blob[:10])
print(np.var(X_blob, axis = 0))

plt.scatter(X_blob[:,0], X_blob[:, 1], c = X_blob[:,0], linewidths = 0 , s = 60, cmap = 'virdis')
```
```python
[[ 0.01018999  2.02672882]
 [-3.18962576  2.40158348]
 [-2.22462566  3.30904327]
 [ 2.48001701  1.62758254]
 [ 0.40223078  1.98665131]
 [ 2.36415919  1.85197933]
 [-0.09114131  2.55032641]
 [ 3.80816708  1.43034139]
 [-4.23579848  3.66839364]
 [ 1.63718644  2.39394337]]

# 선형 변환 후, 분산이 증가함을 확인 
[4.18532335 0.48696538]
```
![2](https://user-images.githubusercontent.com/59912557/82978162-dcacda00-a01e-11ea-815c-fcbbc227ed60.png)

위와 같이 선형변환을 할 경우엔,  기존의 x 축이 현재 그림의 사선으로 존재하면 분산이 더욱 커지고, y축은 그에 대한 직각선 이라면 분산이 작아진다. 앞서 공부했던 분산의 크기에 의한 변수 선택법에 의하면 y축에 대한 분산이 작기에 해당 변수는 가지고 있는 예측 정보량이 작다고 판단할 수 있다. 그렇게 해당 변수는 제거가 가능하며 이것이 주성분 분석의 기본적인 아이디어이다.

이제 실제로 주성분 분석을 실행해보자

```python
pca = PCA()
X_pca = pca.fit_transform(X_blob)
print(X_pca[:10])
print(pca.mean_)
print(X_pca.var(axis=0))
``` 
```python
[[-0.11410926 -0.21437475]
 [-3.29195116 -0.74414614]
 [-2.6173721   0.39587647]
 [ 2.36929377  0.08905996]
 [ 0.27361257 -0.14384839]
 [ 2.19560327  0.27238634]
 [-0.35705127  0.26039009]
 [ 3.69990561  0.26894555]
 [-4.64914578  0.1817632 ]
 [ 1.34659086  0.59080793]]

# pca 결과의 원점
[0.17941386 2.20091474]
# 첫번째 변수의 분산은 증가하고, 두번째는 감소함을 확인
[4.52366539 0.14862335]

```

이제 pca를 적용한 2 개의 변수를 이용해 주성분 분석 결과를 시각화로 표현해보자

```python
# 표준 편차
S = X_pca.std(axis=0)

# 2 개의 그림을 그리기 위함
fig, axes = plt.subplot(1, 2, figsize = (10,10)) 
# 각각의 그림마다 다른 축을 지정하기 위해 축 객체 생성
axes = axes.reavel()

axes[0].scatter(X_blob[:,0], X_blob[:, 1], c = X_blob[:, 0], linewidths = 0, s = 60, cmap = 'viridis')
axes[0].set_title("Original data")
axes[0].set_xlabel("feature 1")
axes[0].set_ylabel("feature 2")

# 변수간 축의 수직관계를 보여주기 위한 화살표 그림 추가
axes[0].arrow(pca.mean_[0], pca.mean_[1], S[0] * pca.components_[0, 0],
              S[0] * pca.components_[0, 1], width = .1, head_width = .3,
              color = 'k')
axes[0].arrow(pca.mean_[0], pca.mean_[1], S[1] * pca.components_[1, 0],
              S[1] * pca.components_[1, 1], width = .1, head_width = .3,
              color = 'k')

axes[0].text(-1.5, -.5, "Component 2", size = 14)
axes[0].text(-4, -4, "Component 1", size = 14)
axes[0].set_aspect('equal') # x,y 축 비율을 같게 표현

# 위의 그림을 수평으로 틀어서 나타낸 그림
axes[1].set_title("Transformed data")
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c = X_pca[:, 0], 
                linewidths = 0, s = 60, cmap = 'viridis')
axes[1].set_xlabel("First principal component")
axes[1].set_ylabel("Second principal component")
axes[1].set_aspect('equal')
axes[1].set_ylim(-6, 6)
```
![3](https://user-images.githubusercontent.com/59912557/82978163-dd457080-a01e-11ea-9059-8883a99c5146.png)

위 그림으로 살펴보면 주성분 분석 결과인 component1은 분산이 크지만, component2는 분산이 작음을 확인할 수 있다. 그럼 pca 주성분의 개수를 1개로만 설정하여 pca를 적용해보자

```python
fig, axes = plt.subplots(1, 2, figsize = (10, 10))

pca = PCA(n_components=1)
pca.fit(X_blob) # class 형태로 변경
X_inverse = pca.inverse_transform(pca.transform(X_blob)) # pca 적용

axes[0].set_title("Transformed data w/ second component dropped")
axes[0].scatter(X_pca[:, 0], np.zeros(X_pca.shape[0]), c = X_pca[:, 0],
                linewidths = 0, s = 60, cmap = 'viridis')
axes[0].set_xlabel("First principal component")
axes[0].set_aspect('equal') # x축과 y축 비율을 같게
axes[0].set_ylim(-8, 8) # y축의 범위 설정

axes[1].set_title("Back-rotation using only first component")
axes[1].scatter(X_inverse[:, 0], X_inverse[:, 1], c = X_pca[:, 0],
                linewidths = 0, s = 60, cmap = 'viridis')
axes[1].set_xlabel("feature 1")
axes[1].set_ylabel("feature 2")
axes[1].set_aspect('equal')
axes[1].set_xlim(-8, 4)
axes[1].set_ylim(-8, 4)
```
![4](https://user-images.githubusercontent.com/59912557/82978164-dd457080-a01e-11ea-8a24-8cec41e4e10d.png)

보다싶이 데이터가 골고루 퍼져있어 분산 즉, 해당 변수의 설명력을 잘 표현함을 확인할 수 있다. 

## Step 2. 주성분 분석 - sklearn Cancer 데이터 적용

sklearn 에서 제공하는 암에 관련된 dataset을 이용해 pca를 적용해보자

```python
from sklearn.dataset import load_breast_cancer

cancer = load_breast_cancer

# 자료 탐색
print(cancer.target)
print(cancer.data.shape)
```

```python
# 0 = 악성, 1 = 양성
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 1 0 0 0 0 0 0 0 0 1 0 1 1 1 1 1 0 0 1 0 0 1 1 1 1 0 1 0 0 1 1 1 1 0 1 0 0
 1 0 1 0 0 1 1 1 0 0 1 0 0 0 1 1 1 0 1 1 0 0 1 1 1 0 0 1 1 1 1 0 1 1 0 1 1
 1 1 1 1 1 1 0 0 0 1 0 0 1 1 1 0 0 1 0 1 0 0 1 0 0 1 1 0 1 1 0 1 1 1 1 0 1
 1 1 1 1 1 1 1 1 0 1 1 1 1 0 0 1 0 1 1 0 0 1 1 0 0 1 1 1 1 0 1 1 0 0 0 1 0
 1 0 1 1 1 0 1 1 0 0 1 0 0 0 0 1 0 0 0 1 0 1 0 1 1 0 1 0 0 0 0 1 1 0 0 1 1
 1 0 1 1 1 1 1 0 0 1 1 0 1 1 0 0 1 0 1 1 1 1 0 1 1 1 1 1 0 1 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 1 1 1 1 1 1 0 1 0 1 1 0 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1
 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 1 1 0 0 0 1 1
 1 1 0 1 0 1 0 1 1 1 0 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 0 0 1 0 0
 0 1 0 0 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1
 1 0 1 1 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 0 1 1 1 1 1 0 1 1
 0 1 0 1 1 0 1 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 0 1
 1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1 0 0 1 0 1 0 1 1 1 1 1 0 1 1 0 1 0 1 0 0
 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 0 0 0 0 0 0 1]

(569, 30) # 30개 변수, 569 인스턴스
```

각 변수별 단위와 범위가 다르기에 표준화 진행 후
 악성과 양성간의 차이를 보기 위해 data splite 진행하여 시각화 실행 

```python
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

# 컬러 리스트
cm3 = ListedColormap(['#0000aa', '#ff2020', '#50ff50'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(cancer.data)

malignant = X_scaled[cancer.target == 0] # 악성
bengin = X_scaled[cancer.target == 1] # 양성

fig, axes = plt.subplots(15, 2, figsize = (10,20))
ax = axes.ravel()

for i in range(30):
	_, bins = np.histogram(X_scaled[:, i], bins = 50)
	ax[i].hist(malignant[:,i], bins = bins, color = cm3(0), alpha = .5)
	ax[i].hist(bengin[:,i],bins = bins, color = cm3(2), alpha = .5)
	ax[i].set_title(cancer.feature_names[i])
	ax[i].set_yticks(()) # y축 단위 제거
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["malignant", "benign"], loc="best")
fig.tight_layout()
```

![6](https://user-images.githubusercontent.com/59912557/82978169-df0f3400-a01e-11ea-9e2e-7b84aa7cc76d.png)

위 그림을 보면 변수별 음성/양성 분포가 비슷한 변수도 있지만, 'worst concave points'와 같이 음성/양성을 매우 잘 구별할 수 있는 변수도 있음이 확인이 가능하다.

이제 이 데이터 셋을 주성분 분석으로 30 차원에서 2 차원으로 축소시켜보자

```python

## 주성분 분석: 2개 주성분(2차원)
pca = PCA(n_components = 2)
X_pca = pca.fit_transform(X_scaled)

print("Original shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))

print("PCA component shape: {}".format(pca.components_.shape))
print("PCA components:\n{}".format(pca.components_))
```
```python
Original shape: (569, 30)
Reduced shape: (569, 2) # 2개의 변수로 줄었음을 확인할 수 있다

PCA component shape: (2, 30)

# 첫번째 주성분은 모두 양수이기에 가중 평균의 개념 즉 x축
# 두번째 주성분은 양수 음수가 섞여 있기에 선형 대비로 설명이 가능한 y축
PCA components:
[[ 0.21890244  0.10372458  0.22753729  0.22099499  0.14258969  0.23928535
   0.25840048  0.26085376  0.13816696  0.06436335  0.20597878  0.01742803
   0.21132592  0.20286964  0.01453145  0.17039345  0.15358979  0.1834174
   0.04249842  0.10256832  0.22799663  0.10446933  0.23663968  0.22487053
   0.12795256  0.21009588  0.22876753  0.25088597  0.12290456  0.13178394]
 [-0.23385713 -0.05970609 -0.21518136 -0.23107671  0.18611302  0.15189161
   0.06016536 -0.0347675   0.19034877  0.36657547 -0.10555215  0.08997968
  -0.08945723 -0.15229263  0.20443045  0.2327159   0.19720728  0.13032156
   0.183848    0.28009203 -0.21986638 -0.0454673  -0.19987843 -0.21935186
   0.17230435  0.14359317  0.09796411 -0.00825724  0.14188335  0.27533947]]
```

숫자로만 판단하기 어렵기에 그림으로 그려서 판단 또한 가능하다

```python
plt.matshow(pca.components_, cmap = 'viridis')
plt.yticks([o, 1], ["First components", "Second components"])
plt.xticks(range(len(cancer.feature_names)),
           cancer.feature_names, rotation = 60, ha = 'left')
plt.xlabel("Feature")
plt.ylabel("Principal components")
```
![8](https://user-images.githubusercontent.com/59912557/82978172-dfa7ca80-a01e-11ea-89a7-26d6c0229144.png)
첫번째 행은 모두 초록색 계열이기에 양수이며 가중 평균임을 확인 가능하고, 두번째 행은 음수 양수 섞여있기에 선형 대비 축임을 확인이 가능하다.

그럼 로지스틱 회귀분석을 사용할 때, 모든 변수를 사용할 경우와 주성분 만 사용할 경우 어느것이 성능이 더 좋을지 확인해 보자

```python
from sklearn.linear_model import LogisticRegression

# 모든 변수
# random_state = 머신러닝 학습을 위해 셔플을 수행하는 랜던 넘버 시드
logistic = LogisticRegression(random_state = 0).fit(cancer.data, cancer.target)
logistic.predict(cancer.data)

print(logistic.predict_proba(cancer.data))
print(logistic.score(cancer.data, cancer.target)
```

```python
# 항목 하나가 예측률이 100퍼인 과적합 현상
[[1.00000000e+00 3.94473969e-14]
 [9.99999889e-01 1.10652307e-07]
 [9.99998883e-01 1.11719115e-06]
 ...
 [9.92517189e-01 7.48281109e-03]
 [1.00000000e+00 1.71145786e-10]
 [4.56551603e-02 9.54344840e-01]]

# 총 예측률
0.942
```
이젠 주성분 2개로 이용하여 로지스틱 분석을 실행해보자

```python
logistic = LogisticRegression(random_state = 0).fit(X_pca, cancer.target)
print(logistic.predict_proba(X_pca))
print(logistic.score(X_pca, cancer.target)) # 0.956
```
```python
[[9.99999889e-01 1.11185767e-07]
 [9.99818356e-01 1.81644225e-04]
 [9.99995811e-01 4.18908803e-06]
 ...
 [9.85781568e-01 1.42184324e-02]
 [9.99999993e-01 7.45581034e-09]
 [2.10795776e-05 9.99978920e-01]]

0.956
```
오히려 30 개의 변수가 아닌 2 개의 주성분을 이용하는 것이 분류 정확도가 더 높게 나옴을 알 수 있다.
