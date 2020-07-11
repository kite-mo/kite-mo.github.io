---  
title:  "10. 로지스틱 회귀분석"  
  
categories:  
 - Python
tags:  
 - Study, Python
---

# 10. 로지스틱 회귀분석 

### 목차

-  Step 0. 로지스틱 회귀분석 개념
-  Step 1. 로지스틱 회귀분석 적용

## Step 0. 로지스틱 회귀분석 개념

![분류1](https://user-images.githubusercontent.com/59912557/84455034-3bb55480-ac97-11ea-83ef-151d3ae0e6c1.PNG)

분류문제는 종속변수가 범주형인 경우 적용하는 방법론이다. 대표적으로 기본적인 모형으론 로지스틱 회귀모형과 의사결정나무 모형이 있다

이번에는 로지스틱 선형회귀모형에 대해 공부를 해보자

![분류2](https://user-images.githubusercontent.com/59912557/84455036-3ce68180-ac97-11ea-832d-e3af6880d0cd.PNG)

가장 기본적인 분류 문제로는 이항 분류(binary classification)이 있다. 자료구조는 위와 같다. $Yi$는 확률변수로 해당 경우가 발생할 확률이다.  

![분류3](https://user-images.githubusercontent.com/59912557/84455037-3d7f1800-ac97-11ea-898a-c3b18483013e.PNG)

첫번째는 $Xi$인 벡터와 $B$ 의 모수를 곱한 선형식이다.  확률은 0 ~ 1의 범위를 가져야 하는데 해당 식의 문제점은 확률이 양쪽 범위를 모두 벗어난다.

그래서 확률에 로그를 취한 두번째 식으로 발전 시켰지만, 확률이 0 미만 즉, 음의 방향으로 무한대로 작아질 수 있다.

그래서 (성공확률)/(실패확률)로 표현하는 odds 개념을 발견했지만, 해당 확률이 1 보다 커질 수 있는 문제점이 있다.

![분류4](https://user-images.githubusercontent.com/59912557/84455757-145f8700-ac99-11ea-98ad-c48697ccdce6.PNG)


그래서 최종적으로 찾은 식인 오즈에 로그를 취한 로지스틱 선형회귀모형이다. 확률의 범위는 0 ~ 1로 표현이 가능하다. 

![분류5](https://user-images.githubusercontent.com/59912557/84455040-3e17ae80-ac97-11ea-8c51-b83cb30ae505.PNG)

위에서 도출해낸 로지스틱 회귀모형을 이용해서 최대우도추정식을 유도하면 위 그림 첫 번째 식 처럼 나타낼 수 있다. 그러나 구해야할 모수가 안보이기에 모수 $B$를 나타내기 위해 두 번째 식으로 표현이 가능하다. 

$ni$는 몇번 시도했는지를 나타내는 식이므로 상수항 취급하여 마지막 항은 없는셈 치고 계산해도 무관하다.

![분류6](https://user-images.githubusercontent.com/59912557/84455041-3eb04500-ac97-11ea-8160-b13457ddef45.PNG)

선형 회귀식에서 기울기를 갱신하듯이 마찬가지로 기울기 갱신이 필요하다. 위에서 구한 로그 우도함수를 $B$로 편미분하여 기울기를 구한 뒤, 학습률을 곱해주며 기울기를 갱신해주는 최대경사도법을 적용시켜 학습시키는 알고리즘이다.

이제 위 개념들을 코딩으로 함수로 표현하여 데이터를 적용시켜보도록 하자

##  Step 1. 로지스틱 회귀분석 적용

```python
from numpy as np
from sklearn.datasets import load_iris

# iris data 불러오기
iris = load_iris()

# 독립변수
X = iris.data

# binaray classification을 위해 0,1로 나누기
Y = (iris.target !=0) * 1
# 절편 생성을 위해 X의 크기 확인
print(X.shape)
```
```python
(150, 4)
```
선형식을 만들기 위해 기존 데이터 크기와 맞는 절편 생성을 해보기전에 기초적인 np 의 함수들을 공부하자

```python
# 0으로 이뤄진 5개 행 생성
np.zeros(5)

# 2,2 행렬 만들기
tx = np.asarray([[1,2], [3,4]])

# tx 행 크기에 적합한 1로 이뤄진 array 만들기
b = np.ones(tx.shape[0])

# 선형식 만들기 - 절편 x 행렬
np.dot(tx,b)

# 전치행렬 만들기
tx.T
```
기본적인 np 함수들을 공부했으니, 이제 iris 데이터에 적용시켜보자

```python
# 절편 생성 (150,1) 크기
intercept = np.ones((X.shape[0], 1))

# concat 하기
X = np.concatenate((intercept, X), axis =1)
print(X[:10])

# 초기 모수값 설정 (베타)
beta = np.zeros(X.shape[1])
print(beta)
```
```python
[[1.  5.1 3.5 1.4 0.2]
 [1.  4.9 3.  1.4 0.2]
 [1.  4.7 3.2 1.3 0.2]
 [1.  4.6 3.1 1.5 0.2]
 [1.  5.  3.6 1.4 0.2]
 [1.  5.4 3.9 1.7 0.4]
 [1.  4.6 3.4 1.4 0.3]
 [1.  5.  3.4 1.5 0.2]
 [1.  4.4 2.9 1.4 0.2]
 [1.  4.9 3.1 1.5 0.1]]
 
 array([0., 0., 0., 0., 0.])
```

이제 분석해야할 기본적인 데이터셋을 구축을 했다. 앞서 이론에서 배웠던 수식들을 함수로 표현해보자

```python
# 로지스틱 함수
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

# 예측확률 반환 
def predict_prob(X, beta):
	return sigmoid(np.dot(X,beta))

# negative log probability 함수 = cost function, object function ..
def negative_log_likelihood(p,y):
	# 이론식처럼 합을 이용하여 값이 너무 커지기에 평균으로 대체
	return - (y * np.log(p) + (1 - y) * np.log(1-p)).mean()

# gradient 구하기
def gradient(p, y):
	# p-y 행렬과 곱하기 위해 X 전치
	# 위 식과 마찬가지로 평균으로 구하기 위해 y.size로 나눠주기 
	return - np.dot(X.T, (p - y)) / y.size

# 예측하기
def predict(X, beta, threshold):
	return predict_prob(X, beta) >= threshold
```
이제 위 함수들을 조합해서 로지스틱의 최종 알고리즘을 구축하자

```python
num_iter = 1000 # 최대반복수
eta = 0.1 # 학습률

# 최대경사도법
for i in range(num_iter):
	
	# beta의 초기값으로 선형식 만들기
	z = np.dot(X, beta)
	p = sigmoid(z)

	# beta 갱신
	beta += eta * gradient(p, y)
	
	# 갱신된 beta로 새로 계산
	z = np.dot(X,beta)
	p = sigmoid(z)
	
	# cost function의 변화 확인
	if (i+1)%100 == 0:
		print('At iteration %d: loss = %.4f' %((i+1), negative_log_likelihood(p,y)))

preds = predict(X, beta, 0.5)

# accuracy
(preds == y).mean()
```

