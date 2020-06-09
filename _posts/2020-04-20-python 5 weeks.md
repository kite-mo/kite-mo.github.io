---  
title:  "3. 데이터 시각화"  
  
categories:  
 - Python
tags:  
 - Study, Python
---

# 3. 데이터 시각화
### 목차
-  Step 1. seaborn의 기본 시각화 
-  Step 2. 산포도, 히스토그램
-  Step 3. 바이올린, 페어 그림 등      

오늘은 데이터 시각화에 대해서 공부하는 시간이다. 마찬가지로 iris 데이터를 이용하여 실습을 할 것이다.

```python
import pandas as pd

#load iris
iris = pd.read_csv("../data/iris.csv")
print(iris.columns)
```
```python
Index(['sepal length in cm', 'sepal width in cm', 'petal length in cm',
       'petal width in cm', 'species'],
      dtype='object')
```
현재 컬럼명이 복잡하기 때문에 조금 더 간단한 컬럼명으로 바꿔줘야겠다
```python
iris.rename(columns = 
{iris.columns[0] : 'Sepal.Length',
iris.columns[1] : 'Sepal.Width',
iris.columns[2] : 'Petal.Length',
iris.columns[3] : 'Petal.Width',
iris.columns[4] : 'Species'})

print(iris.columns)
```
컬럼명이 바뀐 것을 확인할 수 있다.
```python
Index(['Sepal.Length', 'Sepal.Width', 'Petal.Length', 
'Petal.Width','Species'],
      dtype='object')
```
이번엔 특정 컬럼에서 존재하는 데이터의 개수를 확인을 해보자. species에 존재하는 값들은 범주형 데이터이기에 해당 범주당 데이터가 몇개 존재하는지 확인이 가능하다
```python
iris['Species'].value_count()

virginica     50
setosa        50
versicolor    50
Name: Species, dtype: int64
```
자 그럼 이제 본격적으로 해당 데이터셋을 이용하여 시각화를 해보자! 

## Step 1. seaborn의 기본 시각화 

파이썬 시각화 library 중 가장 많이 사용되는 seaborn을 사용할 것이다. 

데이터 분석에서 수치, 통계치를 통한 분석도 중요하나 시각화를 이용하면 더 쉽게 해당 데이터의 전체적인 개요를 확인이 가능하다.

```python
import seaborn as sns

# seaborn 초기 셋팅
# style = 배경, palette = 다양한 색 설정, color_codes 설정
sns.set(style = 'white', palette = 'muted', color_codes = True)
```

### 히스토그램
시각화하면 가장 먼저 떠오르는 히스토그램을 그려보자

```python
# 히스토그램과 밀도 그래프를 동시에 그리는 함수이기에
# kde(밀도 그래프)를 False로 설정
sns.displot(iris['Sepal.Length'], kde = False, color = 'b')
```
![1](https://user-images.githubusercontent.com/59912557/79420469-f3115000-7ff3-11ea-8e05-dcffb491f6cb.png)

구간의 개수를 보면 총 8개로 기본 셋팅이 되어있음을 확인이 가능하다. 이런 구간을 bin 이라고 지칭하며 bin size에 따라 히스토그램의 그림을 다양한 각도에서 해석이 가능하다

```python
# 15개의 bin 
sns.displot(iris['Sepal.Length'], kde = False, bins = 15, color = 'b')
```
![2](https://user-images.githubusercontent.com/59912557/79420470-f3a9e680-7ff3-11ea-901c-5a16fc2142ba.png)
이렇듯 구간의 개수만 늘렸을 뿐인데도 그림의 모양이 달라졌음을 확인이 가능하다. 이렇듯 bin의 개수를 정하는 문제는 중요하기에 공식을 이용하여 설정하는 방법도 존재한다.

```python
# sturges 방법론
sns.displot(iris['Sepal.Length'], kde = False, bins = 'sturges', color = 'r')
# fd 방법론
sns.displot(iris['Sepal.Length'], kde = False, bins = 'fd', color = 'b')
```
![3,4](https://user-images.githubusercontent.com/59912557/79420463-f1478c80-7ff3-11ea-9924-da7ec5910923.png)

이번엔 matplotlib라는 시각화 library를 이용하여 그림을 그려보자

```python
import matplotlib.pyplot as plt

# 그림 사이즈 설정 height = 8, width = 5
plt.figure(figsize = (8,5))
sns.displot(iris['Sepal.Length'], kde = False, color = 'b')
# 그림 호출
plt.show()
```
![5](https://user-images.githubusercontent.com/59912557/79420466-f278b980-7ff3-11ea-9067-0a76ff397942.png)

이전의 그림과 달리 크기가 달라짐을 확인 할 수 있다. 이번엔 밀도 그래프를 그려보자 

```python
plt.figure(figsize = (8,5))
# rug로 실제 데이터가 어디에 위치하는지 표현 가능
# kde_kws = {"shade" = True}) : 배경색 허용 
sns.displot(iris['Sepal.Length'], hist = False, rug = True, color = 'g', kde_kws = {"shade" = True})
```

![7](https://user-images.githubusercontent.com/59912557/79423774-22c35680-7ffa-11ea-9979-d6a47e943784.png)

이번엔 밀도 그래프와 히스토그램을 같이 그려보자
```python
plt.figure(figsize = (8,5))
sns.displot(iris['Sepal.Length'], color = 'm')
plt.show()
```
![8](https://user-images.githubusercontent.com/59912557/79420501-00c6d580-7ff4-11ea-8bc7-20f8f311fb3b.png)

그럼 위 네 개의 그래프를 한 화면에 담을 수 없을까? 당연히 할 수 있다. 
subplots 이라는 함수를 이용해보자

```python
# f = subplot들을 담는 전체 하나의 큰 그림
# axes = 각 subplot을 의미
# sharex = 각 subplot이 같은 x축을 공유
f, axes = plt.subplots(2,2 figsize = (7,7), sharex =True)
sns.despine(left = True) # 그래프 테두리 없애기

sns.distplot(iris['Sepal.Length'], kde = False, color = "b", ax = axes[0, 0])
sns.distplot(iris['Sepal.Length'], hist = False, rug = True, color="r", ax = axes[0, 1])
sns.distplot(iris['Sepal.Length'], hist = False, color = "g", kde_kws = {"shade": True}, ax = axes[1, 0])
sns.distplot(iris['Sepal.Length'], color = "m", ax = axes[1, 1])

plt.setp(axes, yticks = []) # y축 제거
plt.tight_layout() # x,y축 label 크기 자동조절
plt.show()
```
![9](https://user-images.githubusercontent.com/59912557/79420502-015f6c00-7ff4-11ea-8c02-1a26b1ce5e5e.png)

이번엔 각각 다른 열에 대해서 그림을 분석해보자
```python
f, axes = plt.subplots(2, 2, figsize = (7, 7), sharex = True)
sns.despine(left = True)

sns.distplot(iris['Sepal.Length'], color = "m", ax = axes[0, 0])
sns.distplot(iris['Sepal.Width'], color = "m", ax = axes[0, 1])
sns.distplot(iris['Petal.Length'], color = "m", ax = axes[1, 0])
sns.distplot(iris['Petal.Width'], color = "m", ax = axes[1, 1])

plt.setp(axes, yticks = [])
plt.tight_layout()
plt.show()
```
![10](https://user-images.githubusercontent.com/59912557/79420503-015f6c00-7ff4-11ea-8b02-c14fc7cb22a5.png)

위 그림은 전체 데이터에 관련된 그래프이기에 각 species 별 그래프 분포를 확인하여 어떤 species가 특정 범위에 몰려있는지 확인해보자

```python
plt.figure(figsize = (8,5))
sns.displot(iris[iris['Species'] == 'setosa']['Petal.Length'], color = 'b', label = 'setosa')
sns.displot(iris[iris['Species'] == 'versicolor']['Petal.Length'], color = 'r', label = 'versicolor')
sns.displot(iris[iris['Species'] == 'virginica']['Petal.Length'], color = 'm', label = 'virginica')

# 범례 만들기
plt.legend(title = 'Species')
plt.show() 
```
![11](https://user-images.githubusercontent.com/59912557/79728428-e27d1480-8328-11ea-8b86-7b069de2e5fa.png)
versicolor와 virginica는 비슷한 길이에 분포하지만, setosa는 비교적 작음을 확인할 수 있다.

좀 더 직관적으로 이해하기 위해 이번엔 상자 그림을 그려서 비교를 해보자.
```python
plt.figure(figsize = (8,5))
sns.boxplot(x = 'Petal.Length', y = 'Species', data = iris)
plt.show()
```
![12](https://user-images.githubusercontent.com/59912557/79728446-e741c880-8328-11ea-8d44-722b2e29e0f6.png)

##  Step 2. 산점도, 히스토그램

이번엔 산점도를 이용하여 length와 width간의 관계를 확인해보자

```python
plt.figure(figsize = (8, 8))
sns.scatterplot(x = 'Petal.Length', y = 'Petal.Width', data = iris)
plt.show()
```
![13](https://user-images.githubusercontent.com/59912557/79728450-e7da5f00-8328-11ea-871f-ed57ee793eff.png)

그림의 결과 length와 width간의 선형관계가 존재함을 확인 할 수 있다. 이번엔 각 종의 범주 별로 구분하여 그려보자

```python
plt.figure(figsize = (8,8))
sns.scatterplot(x = 'Petal.Length', y = 'Petal.Width', hue = 'Species', data = iris)
plt.show()
```
![14](https://user-images.githubusercontent.com/59912557/79728451-e7da5f00-8328-11ea-8d49-865bc9af8006.png)

setosa는 완전히 구별이 되나 versicolor와 virginica는 약간 겹쳐서 뚜렷한 구별이 안된다.

그렇담 각 범주 별로 회귀 추정선(regression line)을 그려서 같은 추세를 띄는지 확인을 해보자

```python
# lmplot : regression line 추가
# aspect = subplot 의 세로 대비 가로의 비율
g = sns.lmplot(x = 'Petal.Length', y = 'Petal.Width', hue = 'Species', height = 6, aspect = 8/6, data = iris)
# x,y축 이름 설정
g.set_axis_labels('Petal.Length (mm)', 'Petal.Width (mm)')
```
![15](https://user-images.githubusercontent.com/59912557/79728452-e872f580-8328-11ea-808d-60ff0805c6d4.png)

 주황과 초록이 구분이 잘 안갔지만 회귀선으로 확인하니 다른 기울기를 가졌음을 알 수 있었다.

이번엔 산점도와 히스토그램을 동시에 그릴 수 있는 joinplot() 함수를 사용해보자

```python
sns.jointplot(x = 'Petal.Length', y = 'Petal.Width', kind = 'scatter', 
              marginal_kws = dict(bins = 15, rug = True),  
              annot_kws = dict(stat = "r"), s = 40, 
              height = 8, data = iris)
```
![16](https://user-images.githubusercontent.com/59912557/79728453-e90b8c00-8328-11ea-8352-4fe560885a03.png)

산점도와 밀도함수 또한 한 그림에 그릴 수 있다.

```python
sns.pairplot(iris, hue = 'Species', palette = 'husl')
```
![17](https://user-images.githubusercontent.com/59912557/79728454-e90b8c00-8328-11ea-98f3-02a898bce556.png)

##  Step 3. Swarm, Strip, Correlation plot

지금부터는 전통적인 그림보단 조금 멋진 그림들을 그려보자. 

### swarm plot
```python
# 초기화
sns.set(style = 'whitegrid', palette = 'muted')

# pd.melt 범주 별로 데이터 순서를 나열
tidy_iris = pd.melt(iris, 'Species', var_name = 'Measurement') 
print(tidy_iris.head())
```
```python
  Species   Measurement  value
0  setosa  Sepal.Length    5.1
1  setosa  Sepal.Length    4.9
2  setosa  Sepal.Length    4.7
3  setosa  Sepal.Length    4.6
4  setosa  Sepal.Length    5.0
```

```python
plt.figure(figsize = (7, 7))
# Draw a categorical scatterplot to show each observation
sns.swarmplot(x = 'Measurement', y = 'value', hue = 'Species',
              palette = ['r', 'c', 'y'], data = tidy_iris)
plt.show()
```
![18](https://user-images.githubusercontent.com/59912557/79728455-e9a42280-8328-11ea-8e91-897485d739d6.png)

