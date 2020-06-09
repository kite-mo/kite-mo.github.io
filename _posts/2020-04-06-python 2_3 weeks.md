---  
title:  "1. 데이터 요약 및 시각화 in python"  
  
categories:  
 - Python
tags:  
 - Study, Python
 
---

# 1. 데이터 요약 및 시각화 in python
### 목차
-  Step 0. 개요
-  Step 1. pandas를 활용한 데이터 관리
-  Step 2. seaborn을 이용한 데이터 탐색
-  Step 3. 그룹화와 예측, 분류

##   Step 0. 개요
----
학교 전공 시간에 배우는 파이썬을 이용한 데이터마이닝 수업을 복습 겸 추가적 공부를 하기 위해 블로그로 정리를 하려한다.

우선 python을 사용하기 위해서는 아래 사항이 필요하다.

- 파이썬 공식 홈페이지에서의 다운로드
	- [https://www.python.org](https://www.python.org/)
	
- 아나콘다 설치	
	- 파이썬 및 주로 사용되는 1400개의 library와 data set들이 포함되어 있어 여러 패키지들을 설치하는 수고를 덜어줌
	- https://www.anaconda.com/download/  
 - 파이썬 interpreter 설치
	 - 컴파일 없이 코드를 바로 실행할 수 있는 interpreter 언어의 실행 도구라고 이해하면 되겠다.
	 - interpreter의 종류는 다양하지만, 해당 수업에서는 anaconda를 설치하면 제공되는 **spyder** 를 이용한다 
	 - 그리고 보통 pycharm이라는 interpreter를 사용한다

## Step 1. pandas를 활용한 데이터 관리
---
위 과정을 모두 수행했다면 데이터 분석 라이브러리로, 행과 열로 이루어진 dataframe, series 데이터 객체를 제공하여 안정적으로 대용량의 데이터들을 처리하는데 매우 편리한 도구인 pandas를 이용해보자. 사용 데이터는 Iris 데이터이다. 
사전에 데이터를 다운받아 working directory에 넣어놨다.

```python
import pandas as pd
df = pd.read_csv("./data/iris.csv") # data load
```
### 1) 데이터 상태 확인
이제 해당 데이터가 어떤 형태로 이뤄져 있는지 확인을 한다. 

```python
print("shape of data in (rows, columns) is " + str(df.shape)) # 데이터의 행, 열의 개수 확인
print(df.head()) # 상위 5개의 데이터 확인
df.describe().transpose() # 전치된 데이터 요약 결과 확인 
```
```python
shape of data in (rows, columns) is (150, 5)
```
```python
   sepal length in cm  sepal width in cm  ...  petal width in cm  species
0                 5.1                3.5  ...                0.2   setosa
1                 4.9                3.0  ...                0.2   setosa
2                 4.7                3.2  ...                0.2   setosa
3                 4.6                3.1  ...                0.2   setosa
4                 5.0                3.6  ...                0.2   setosa
```
```python
                    count      mean       std  min  25%   50%  75%  max
sepal length in cm  150.0  5.843333  0.828066  4.3  5.1  5.80  6.4  7.9
sepal width in cm   150.0  3.054000  0.433594  2.0  2.8  3.00  3.3  4.4
petal length in cm  150.0  3.758667  1.764420  1.0  1.6  4.35  5.1  6.9
petal width in cm   150.0  1.198667  0.763161  0.1  0.3  1.30  1.8  2.5
```
### 2) 데이터 attribute 확인
```python
print(df.columns) # check features name, 열 이름 확인
print(df.index) # check index numbers
print(df.dtypes) # check features type
print(df.values) # check features value
print(type(df.values)) # check type of df.values
```
```python
Index(['sepal length in cm', 'sepal width in cm', 'petal length in cm','petal width in cm', 'species'],
      dtype='object')
```
```python
RangeIndex(start=0, stop=150, step=1) # 1씩 증가하는 0~150 까지의 인덱스
```
```python
sepal length in cm    float64
sepal width in cm     float64
petal length in cm    float64
petal width in cm     float64
species                object
```
```python
[[5.1 3.5 1.4 0.2 'setosa']
 [4.9 3.0 1.4 0.2 'setosa']
 [4.7 3.2 1.3 0.2 'setosa']
 [4.6 3.1 1.5 0.2 'setosa']
 [5.0 3.6 1.4 0.2 'setosa']
 [5.4 3.9 1.7 0.4 'setosa']
 [4.6 3.4 1.4 0.3 'setosa']
 [5.0 3.4 1.5 0.2 'setosa']
 [4.4 2.9 1.4 0.2 'setosa']
 [4.9 3.1 1.5 0.1 'setosa']
 [5.4 3.7 1.5 0.2 'setosa']
 [4.8 3.4 1.6 0.2 'setosa']
 [4.8 3.0 1.4 0.1 'setosa']
 [4.3 3.0 1.1 0.1 'setosa']
 [5.8 4.0 1.2 0.2 'setosa']
 [5.7 4.4 1.5 0.4 'setosa']
 [5.4 3.9 1.3 0.4 'setosa']
 [5.1 3.5 1.4 0.3 'setosa']
 [5.7 3.8 1.7 0.3 'setosa']
 [5.1 3.8 1.5 0.3 'setosa']
 [5.4 3.4 1.7 0.2 'setosa']
 [5.1 3.7 1.5 0.4 'setosa']
 [4.6 3.6 1.0 0.2 'setosa']
 [5.1 3.3 1.7 0.5 'setosa']
 [4.8 3.4 1.9 0.2 'setosa']
 [5.0 3.0 1.6 0.2 'setosa']
 [5.0 3.4 1.6 0.4 'setosa']
 [5.2 3.5 1.5 0.2 'setosa']
 [5.2 3.4 1.4 0.2 'setosa']
 [4.7 3.2 1.6 0.2 'setosa']
 [4.8 3.1 1.6 0.2 'setosa']
 [5.4 3.4 1.5 0.4 'setosa']
 [5.2 4.1 1.5 0.1 'setosa']
 [5.5 4.2 1.4 0.2 'setosa']
 [4.9 3.1 1.5 0.1 'setosa']
 [5.0 3.2 1.2 0.2 'setosa']
 [5.5 3.5 1.3 0.2 'setosa']
 [4.9 3.1 1.5 0.1 'setosa']
 [4.4 3.0 1.3 0.2 'setosa']
 [5.1 3.4 1.5 0.2 'setosa']
 [5.0 3.5 1.3 0.3 'setosa']
 [4.5 2.3 1.3 0.3 'setosa']
 [4.4 3.2 1.3 0.2 'setosa']
 [5.0 3.5 1.6 0.6 'setosa']
 [5.1 3.8 1.9 0.4 'setosa']
 [4.8 3.0 1.4 0.3 'setosa']
 [5.1 3.8 1.6 0.2 'setosa']
 [4.6 3.2 1.4 0.2 'setosa']
 [5.3 3.7 1.5 0.2 'setosa']
 [5.0 3.3 1.4 0.2 'setosa']
 [7.0 3.2 4.7 1.4 'versicolor']
 [6.4 3.2 4.5 1.5 'versicolor']
 [6.9 3.1 4.9 1.5 'versicolor']
 [5.5 2.3 4.0 1.3 'versicolor']
 [6.5 2.8 4.6 1.5 'versicolor']
 [5.7 2.8 4.5 1.3 'versicolor']
 [6.3 3.3 4.7 1.6 'versicolor']
 [4.9 2.4 3.3 1.0 'versicolor']
 [6.6 2.9 4.6 1.3 'versicolor']
 [5.2 2.7 3.9 1.4 'versicolor']
 [5.0 2.0 3.5 1.0 'versicolor']
 [5.9 3.0 4.2 1.5 'versicolor']
 [6.0 2.2 4.0 1.0 'versicolor']
 [6.1 2.9 4.7 1.4 'versicolor']
 [5.6 2.9 3.6 1.3 'versicolor']
 [6.7 3.1 4.4 1.4 'versicolor']
 [5.6 3.0 4.5 1.5 'versicolor']
 [5.8 2.7 4.1 1.0 'versicolor']
 [6.2 2.2 4.5 1.5 'versicolor']
 [5.6 2.5 3.9 1.1 'versicolor']
 [5.9 3.2 4.8 1.8 'versicolor']
 [6.1 2.8 4.0 1.3 'versicolor']
 [6.3 2.5 4.9 1.5 'versicolor']
 [6.1 2.8 4.7 1.2 'versicolor']
 [6.4 2.9 4.3 1.3 'versicolor']
 [6.6 3.0 4.4 1.4 'versicolor']
 [6.8 2.8 4.8 1.4 'versicolor']
 [6.7 3.0 5.0 1.7 'versicolor']
 [6.0 2.9 4.5 1.5 'versicolor']
 [5.7 2.6 3.5 1.0 'versicolor']
 [5.5 2.4 3.8 1.1 'versicolor']
 [5.5 2.4 3.7 1.0 'versicolor']
 [5.8 2.7 3.9 1.2 'versicolor']
 [6.0 2.7 5.1 1.6 'versicolor']
 [5.4 3.0 4.5 1.5 'versicolor']
 [6.0 3.4 4.5 1.6 'versicolor']
 [6.7 3.1 4.7 1.5 'versicolor']
 [6.3 2.3 4.4 1.3 'versicolor']
 [5.6 3.0 4.1 1.3 'versicolor']
 [5.5 2.5 4.0 1.3 'versicolor']
 [5.5 2.6 4.4 1.2 'versicolor']
 [6.1 3.0 4.6 1.4 'versicolor']
 [5.8 2.6 4.0 1.2 'versicolor']
 [5.0 2.3 3.3 1.0 'versicolor']
 [5.6 2.7 4.2 1.3 'versicolor']
 [5.7 3.0 4.2 1.2 'versicolor']
 [5.7 2.9 4.2 1.3 'versicolor']
 [6.2 2.9 4.3 1.3 'versicolor']
 [5.1 2.5 3.0 1.1 'versicolor']
 [5.7 2.8 4.1 1.3 'versicolor']
 [6.3 3.3 6.0 2.5 'virginica']
 [5.8 2.7 5.1 1.9 'virginica']
 [7.1 3.0 5.9 2.1 'virginica']
 [6.3 2.9 5.6 1.8 'virginica']
 [6.5 3.0 5.8 2.2 'virginica']
 [7.6 3.0 6.6 2.1 'virginica']
 [4.9 2.5 4.5 1.7 'virginica']
 [7.3 2.9 6.3 1.8 'virginica']
 [6.7 2.5 5.8 1.8 'virginica']
 [7.2 3.6 6.1 2.5 'virginica']
 [6.5 3.2 5.1 2.0 'virginica']
 [6.4 2.7 5.3 1.9 'virginica']
 [6.8 3.0 5.5 2.1 'virginica']
 [5.7 2.5 5.0 2.0 'virginica']
 [5.8 2.8 5.1 2.4 'virginica']
 [6.4 3.2 5.3 2.3 'virginica']
 [6.5 3.0 5.5 1.8 'virginica']
 [7.7 3.8 6.7 2.2 'virginica']
 [7.7 2.6 6.9 2.3 'virginica']
 [6.0 2.2 5.0 1.5 'virginica']
 [6.9 3.2 5.7 2.3 'virginica']
 [5.6 2.8 4.9 2.0 'virginica']
 [7.7 2.8 6.7 2.0 'virginica']
 [6.3 2.7 4.9 1.8 'virginica']
 [6.7 3.3 5.7 2.1 'virginica']
 [7.2 3.2 6.0 1.8 'virginica']
 [6.2 2.8 4.8 1.8 'virginica']
 [6.1 3.0 4.9 1.8 'virginica']
 [6.4 2.8 5.6 2.1 'virginica']
 [7.2 3.0 5.8 1.6 'virginica']
 [7.4 2.8 6.1 1.9 'virginica']
 [7.9 3.8 6.4 2.0 'virginica']
 [6.4 2.8 5.6 2.2 'virginica']
 [6.3 2.8 5.1 1.5 'virginica']
 [6.1 2.6 5.6 1.4 'virginica']
 [7.7 3.0 6.1 2.3 'virginica']
 [6.3 3.4 5.6 2.4 'virginica']
 [6.4 3.1 5.5 1.8 'virginica']
 [6.0 3.0 4.8 1.8 'virginica']
 [6.9 3.1 5.4 2.1 'virginica']
 [6.7 3.1 5.6 2.4 'virginica']
 [6.9 3.1 5.1 2.3 'virginica']
 [5.8 2.7 5.1 1.9 'virginica']
 [6.8 3.2 5.9 2.3 'virginica']
 [6.7 3.3 5.7 2.5 'virginica']
 [6.7 3.0 5.2 2.3 'virginica']
 [6.3 2.5 5.0 1.9 'virginica']
 [6.5 3.0 5.2 2.0 'virginica']
 [6.2 3.4 5.4 2.3 'virginica']
 [5.9 3.0 5.1 1.8 'virginica']]
```
```python
<class 'numpy.ndarray'>
```
특정 열만 호출하여 확인하기 - df['columns names']
```python
print(df['species'])
```
```python
0         setosa
1         setosa
2         setosa
3         setosa
4         setosa
   
145    virginica
146    virginica
147    virginica
148    virginica
149    virginica
Name: species, Length: 150, dtype: object
```
### 3) slicing and selecting
slicing : 원하는 부분만 잘라서 값을 가져오기 
```pyhton
df([0:50:1]) # 0~49 까지 1씩 증가하는 행 호출
```
selecting 함수인 iloc 활용
- iloc[: , : ] : 특정 행과 열을 지정하여 호출 가능
```python
df.iloc[:, 1:3] # 첫번째 열부터 세번째 열 직전까지 모든 행 가져오기
df.iloc[0:50, :] # 모든 열의 0~49번째 행까지 가져오기
```
### 4) 데이터 요약치 

```python
df.min() # 최소값
df.max() # 최대값
df.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).values # 데이터 최소, 제1사분위수, 중앙값, 제3사분위수, 최대값 확인
```
### 5) pandas를 이용한 시각화

```python
import pandas.plotting as pp
pp.boxplot(df)
pp.scatter_matrix(df)
```
![pp boxplot](https://user-images.githubusercontent.com/59912557/77726304-f9d32580-703a-11ea-8447-0c17f2e079d2.png)
![pp scatter_matrix](https://user-images.githubusercontent.com/59912557/77726312-fb9ce900-703a-11ea-9d0e-6bc17de19cc6.png)
데이터의 분포는 확인이 가능하나 모두 같은 색으로 표현이되어 범주별 확인이 어렵다. 그래서 데이터 시각화에 특화된 library인 seaborn을 이용한다.

## step 2. seaborn을 이용한 데이터 탐색
---
```python
import seaborn as sns
sns.boxplot(x = 'species', y = 'petal length in cm', data =df)
```
![sns boxplot](https://user-images.githubusercontent.com/59912557/77726627-b88f4580-703b-11ea-8354-4ea10cd0fa93.png)
데이터의 분포가 어떻게 퍼져있는지 직관적으로 파악하기가 어렵다. 그래서 violinplot을 이용한다

```python
sns.violinplot(x ='species', y = 'petal length in cm', data = df)
```
![sns violinplot](https://user-images.githubusercontent.com/59912557/77726615-ae6d4700-703b-11ea-800b-b2be7bbad68c.png)
setosa는 정규분포 모양을 띄지만, vericolor와 virginica는 한쪽으로 분포가 치우쳐짐을 확인할 수 있다. 

seaborn을 이용하여  산점도 그래프도 그릴 수 있다.
```python
sns.lmplot(x = 'petal length in cm', y = 'petal width in cm', hue = "species",
 data = df, fit_reg = False, palette = 'bright', markers = ['o', 'x', 'v'])
# hue = 범주의 기준, markers = 각 범주의 표현 모양 설정
```
![sns lmplot](https://user-images.githubusercontent.com/59912557/77726610-aca38380-703b-11ea-9244-3fb15aeaac6f.png)

이번엔 히스토그램과 산점도 그래프를 동시에 그려보자
```python
sns.pairplot(df, hue = 'species', diag_kind = 'hist',
 palette = 'bright', markers = ['o', 'x', 'v'])
```
![sns pairplot hist](https://user-images.githubusercontent.com/59912557/77726612-ae6d4700-703b-11ea-9a2a-9a5d5d65e2b4.png)

위에서 pandas를 이용하여 그렸던 그래프와 똑같지만 색깔과 표현을 구분해줌으로써 각 범주별 산포와 유사성을 확인할 수 있다.


## Step 3. 데이터 정규화 및 요약
---

데이터 분석을 할 경우에 feature 간의 데이터 범위가 매우 상이할 경우 분석에 어려움이 있다. 자세한 내용은 앞선 포스팅인 아래 링크에서 확인하자
(https://kite-mo.github.io/machine%20learning/2020/03/10/bGradient/)

### 1) normalization

직접 정규화 함수를 만들어 feature scaling을 실행해보자.
정규화 공식은  (X - Mean) / std 이다.

```python
def normalize(df):
	result = df.copy() # df 복사
	for feature in df.columns: # 각 feature 이름 별로 반복
		if feature != 'species': # 범주형 자료 제외
			mean_val = df[feature].mean() # 각 열의 평균
			std_val = df[feature].std() # 각 열의 표준편차
			result[feature] = (df[feature] - mean_val)/std_val
	return result

ndf = normalize(df)
print(ndf.iloc[:,0:3].tail())# 하위 5개 데이터 확인
```
```python
    sepal length in cm  sepal width in cm  petal length in cm
145            1.034539          -0.124540            0.816888
146            0.551486          -1.277692            0.703536
147            0.793012          -0.124540            0.816888
148            0.430722           0.797981            0.930239
149            0.068433          -0.124540            0.760212
```
기존 데이터 range에서 축소됐음을 확인할 수 있다.

### 차원축소를 통한 데이터 요약
벡터의 차원, 변수의 개수가 많아지게 되면 생기는 문제점들이 존재한다.
- 모델링에 필요한 학습 집합의 크기가 커짐
- 노이즈들이 포함되므로 예측, 분류에 안좋은 영향을 끼침
- 모델의 학습 속도가 느리며, 성능 저하

그래서 차원 축소를 통해 특징 변수들을 효과적으로 다루며, 높은 차원이 가지는 문제점들을 해결할 수 있다.
여기서 데이터 차원을 낮춘다는 것은 현재 데이터가 존재하는 차원에서 그보다 낮은 차원으로 데이터들을 맵핑 시킨다는 의미이다.  e.g. 10개의 변수를 3개의 변수로 축소

대표적인 두 가지 방법을 소개한다.
### 1) PCA(주성분 분석)
고차원의 정보를 유지하면서 저차원으로 차원을 비지도적 방식(종속변수 X)으로 축소하는 다변량 데이터 처리방법이다.
데이터를 분산이 가장 커지는 새로운 첫 번째 축으로 맵핑시키고 두번째로 커지는 축으로 맵핑시키는 등 원하는 차수만큼의 새로운 축으로 만들어 새로운 좌표계로 데이터를 선형 변환한다.

python library인 sklearn을 이용하면 쉽게 구현이 가능하다.
```python
from sklearn.decomposition import PCA
pca = PCA(n_components = 2) #차원수 설정
out_pca = pca.fit_transform(df[[
'sepal length in cm',
'sepal width in cm',
'petal length in cm',
'petal width in cm']]) # 4개의 차원을 축소
print(out_pca[:4])
```
```python
# 4개의 차원이 2개로 축소되었음을 확인 
[[-2.68420713  0.32660731]
 [-2.71539062 -0.16955685]
 [-2.88981954 -0.13734561]
 [-2.7464372  -0.31112432]]
```
concat 함수를 이용하여 기존의 species 열과 합칠 수 있다.
```python
df_pca = pd.DataFrame(data = out_pca, columns = ['pca1', 'pca2']
df_pca = pd.concat([df_pca, df[['species']]], axis = 1) # axis = 1 : 열 기준
print(df_pca.head())
```
```python
    pca1        pca2   species
0 -2.684207  0.326607  setosa
1 -2.715391 -0.169557  setosa
2 -2.889820 -0.137346  setosa
3 -2.746437 -0.311124  setosa
4 -2.728593  0.333925  setosa
```

### 2) LDA(판별 분석)

지도적 방식인 차원 축소법으로 분류돼 있는 데이터를 가장 잘 분별 할 수있는 새로운 축에 맵핑하는 방법이다. PCA와 다른 점은 데이터가 분류되어 있어야 한다는 점이다.

마찬가지로 sklearn을 이용하여 구현이 가능하다.
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components = 2) # 2개의 차원으로 축소
out_lda = lda.fit_transform(X = df.iloc[:,:4], y =df['species']) # 지도학습이기에 y를 설정
df_lda = pd.DataFrame(data = out_lda, columns = ['lda1', 'lda2'])
df_lda = pd.concat([df_lda, df[['species']]], axis =1)
print(df_lda.head())
```
```python
       lda1    lda2    species
0  8.084953  0.328454  setosa
1  7.147163 -0.755473  setosa
2  7.511378 -0.238078  setosa
3  6.837676 -0.642885  setosa
4  8.157814  0.540639  setosa
```
