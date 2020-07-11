---  
title:  "9. 회귀분석"  
  
categories:  
 - Python
tags:  
 - Study, Python
---

# 9. 회귀분석 

### 목차

-  Step 0. 데이터 전처리
-  Step 1. Steepest Descent Method
-  Step 2. Stochastic Gradient Descent



이번엔 연속형 변수를 예측하는 가장 기본이 되는 방법론인 회귀분석에 대해 공부를 해보려 한다.

선형회귀 모형은 종속(목표)변수가 존재하는 지도학습(Supervised learning)이다. 보통 통계학에서의 회귀분석 모형은 종속변수에 대해 가정이 필요하지만 데이터마이닝 영역에서는 그렇지 않다고 한다.

이렇듯 머신러닝에서 쓰이는 수치해석법으로는 Steepest Descent Method와 Stochastic Gradient Descent가 존재한다.

Steepest Descent Method는 최대경사도법으로 전체 자료에 대해 1차 미분한 탐색경로 설정하고 반복 갱신해가며 최적의 회귀계수 베타값을 찾아나가는 과정이다. 

그러나 데이터마이닝에선 관찰값의 데이터 수가 무수히 많기 떄문에 계산량이 기하급수적으로 증가하게 되는데, 그래서 나온 방법론인 Stochastic Gradient Descent 이다.

해당 방법론은 전체자료를 대상으로 gradient를 계산하지 않고 개별 관측값의 gradient를 반복 계산하며 갱신을 수행한다.

그래도 개별 관측값을 모두 계산하게되면 계산량이 많아지기에 Mini batch 라는 전체 자료의 일부 관측값을 이용함으로써 계산량을 줄인다.

그리고 해당 회귀모형이 잘 구축되었는지 확인하기 위해 검증 자료를 통해 훈련에 포함되지 않은 결과도 잘 설명하는지 확인하는 과정인 cross validation을 수행한다.

자 이제 예제 코딩을 통해 공부를 해보도록 하자

## Step 0. 데이터 탐색 및 전처리

회귀분석을 위한 데이터셋을 불러오자. 이번엔 url을 이용하여 여러 독립변수로 집값을 예측하는 california house price dataset을 이용할 것이다

Data description

목표변수 :
    
0. medianHouseValue: Median house value for households within a block (measured in US Dollars)

독립변수 :
    
1. longitude: A measure of how far west a house is; a higher value is farther west
2. latitude: A measure of how far north a house is; a higher value is farther north
3. housingMedianAge: Median age of a house within a block; a lower number is a newer building
4. totalRooms: Total number of rooms within a block
5. totalBedrooms: Total number of bedrooms within a block
6. population: Total number of people residing within a block
7. households: Total number of households, a group of people residing within a home unit, for a block
8. medianIncome: Median income for households within a block of houses (measured in tens of thousands of US Dollars)
10. oceanProximity: Location of the house w.r.t ocean/sea


```python
url = "https://raw.githubusercontent.com/johnwheeler/handson-ml/master/datasets/housing/housing.csv"
housing = pd.read_csv(url)

# 데이터 체크
print(housing.describe().transpose())
# describe() 대신 사용가능
print(housing.info())
```
```python
                      count           mean  ...           75%          max
longitude           20640.0    -119.569704  ...    -118.01000    -114.3100
latitude            20640.0      35.631861  ...      37.71000      41.9500
housing_median_age  20640.0      28.639486  ...      37.00000      52.0000
total_rooms         20640.0    2635.763081  ...    3148.00000   39320.0000
total_bedrooms      20433.0     537.870553  ...     647.00000    6445.0000
population          20640.0    1425.476744  ...    1725.00000   35682.0000
households          20640.0     499.539680  ...     605.00000    6082.0000
median_income       20640.0       3.870671  ...       4.74325      15.0001
median_house_value  20640.0  206855.816909  ...  264725.00000  500001.0000
[9 rows x 8 columns]

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 10 columns):
longitude             20640 non-null float64
latitude              20640 non-null float64
housing_median_age    20640 non-null float64
total_rooms           20640 non-null float64
total_bedrooms        20433 non-null float64
population            20640 non-null float64
households            20640 non-null float64
median_income         20640 non-null float64
median_house_value    20640 non-null float64
ocean_proximity       20640 non-null object
dtypes: float64(9), object(1)
memory usage: 1.6+ MB
None
```
확인 결과 total_bedrooms 변수가 다른 변수들에 비해 데이터 개수가 부족하다. 더 자세히 확인해보도록 하자

```python
print(housing.isnull().values.any())
print(housing.isnull().values.sum())
```
```python
# 결측값 존재함
True
# 총 207개의 결측값들이 존재
207
```

처음에 total_bedrooms에 결측값들이 존재함을 확인했기에 결측값이 존재하는 index들을 찾아보자

```python
print(housing[housing['total_bedrooms'].isnull())]
```
```python
       longitude  latitude  ...  median_house_value  ocean_proximity
290      -122.16     37.77  ...            161900.0         NEAR BAY
341      -122.17     37.75  ...             85100.0         NEAR BAY
538      -122.28     37.78  ...            173400.0         NEAR BAY
563      -122.24     37.75  ...            247100.0         NEAR BAY
696      -122.10     37.69  ...            178400.0         NEAR BAY
         ...       ...  ...                 ...              ...
20267    -119.19     34.20  ...            220500.0       NEAR OCEAN
20268    -119.18     34.19  ...            167400.0       NEAR OCEAN
20372    -118.88     34.17  ...            410700.0        <1H OCEAN
20460    -118.75     34.29  ...            258100.0        <1H OCEAN
20484    -118.72     34.28  ...            218600.0        <1H OCEAN
```
자 이제 결측값들이 존재하는 인덱스 번호에 imputation을 실행해주자

```python
# 연속형 변수기에 결측값을 해당 열의 평균값으로 imput 실행
housing['total_bedrooms'][housing['total_bedrooms'].isnull()] = np.mean(housing['total_bedrooms'])

print(housing.isnull().values.any())
print(housing.loc[290])
```
```python
False

# 평균값인 537.871로 채워짐
longitude              -122.16
latitude                 37.77
housing_median_age          47
total_rooms               1256
total_bedrooms         537.871
population                 570
households                 218
median_income            4.375
median_house_value      161900
ocean_proximity       NEAR BAY
Name: 290, dtype: obj
```
이번엔 시각화를 이용해서 데이터 탐색을 해보자

```python
# 기본 세팅
sns.set(style = "white", palette = "muted", color_codes = True)

sns.distplot(housing['median_hous_value'], kde = True, color = 'b') 
sns.distplot(housing['total_bedrooms'], kde = False, color = 'b') 
```

![1](https://user-images.githubusercontent.com/59912557/84215443-37573300-ab01-11ea-8dfb-a81c39890344.png)

![2](https://user-images.githubusercontent.com/59912557/84215451-3920f680-ab01-11ea-9fd5-a460f0557104.png)
 
데이터 분포를 확인하니 정규분포를 가정하기엔 어려운 분포 모양을 띄고있음. 회귀분석을 통해 예측하기엔 까다로운 dataset이다. 

그래서 최대한 분포모양을 적합시키기 위해 변환을 실행시켜준다.

```python
housing['avg_rooms'] = housing['total_rooms']/housing['households']
housing['avg_bedrooms'] = housing['total_bedrooms']/housing['households']
housing['pop_household'] = housing['population']/housing['households']
```
그리고 ocean_proximity는 범주형 자료여서 연속형 변수로 바꿔주기 위해 one-hot-encoding 작업을 수행한다.

```python
# 범주 값들 확인
print(housing['ocean_proximity'].unique()]
# ['NEAR BAY' '<1H OCEAN' 'INLAND' 'NEAR OCEAN' 'ISLAND']

# 각 범주값 마다 새로운 열을 생성
housing['NEAR BAY'] = 0
housing['<1H OCEAN'] = 0
housing['INLAND'] = 0
housing['NEAR OCEAN'] = 0
housing['ISLAND'] = 0

# 기존 열에서 각 범주에 속하는 인덱스 번호를 찾아, 위의 열에 해당하는 인덱스에 1 넣기

housing.loc[housing['ocean_proximity']=='NEAR BAY','NEAR BAY'] = 1
housing.loc[housing['ocean_proximity']=='INLAND','INLAND'] = 1
housing.loc[housing['ocean_proximity']=='<1H OCEAN','<1H OCEAN'] = 1
housing.loc[housing['ocean_proximity']=='ISLAND','ISLAND']=1
housing.loc[housing['ocean_proximity']=='NEAR OCEAN','NEAR OCEAN'] = 1

# 'NEAR BAY'에 존재하는 값별 개수 카운트
print(housing['NEAR BAY'].value_counts())
```
```python
0    18350
1     2290
Name: NEAR BAY, dtype: int64
```
전처리 과정을 수행했기에 회귀분석을 진행해보자.

## Step 1. Steepest Descent Method

처음은 전통적인 통계학 방법 이용해 회귀분석을 진행해보자

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 종속변수, 독립변수 나누기
train_x = housing.drop(['total_rooms','total_bedrooms','households',
	                    'ocean_proximity','median_house_value'], axis = 1)
train_y = housing['median_house_value']

# train, test data 나누기
X, test_x, Y, test_y = train_test_split(train_x, train_y, test_size = 0.2)
```
```python
mlr = LinearRegression()
mlr.fit(X,Y)

# 각 독립변수 별 회귀계수 확인
print(mlr.coef_)
# 회귀식의 설명력인 결정계수 확인
print(mlr.score(X,Y))
```
```python
[-2.63893385e+04 -2.50219286e+04  8.21503906e+02 -5.58585839e-01
  3.72685500e+04  1.11313010e+02  8.50837760e+03 -1.89296784e+04
 -6.80890154e+04 -2.46238116e+04  1.29016167e+05 -1.73736615e+04]

0.6100956006973763
```
약 0.61 의 설명력을 나타내는 회귀식임을 확인했다.

소숫점 자리가 매우 길기 때문에 소수점 두자리에서 절상하는 함수를 이용하자
```python
import math

def roundup(x):
	return int(math.ceil(x/100))*100

# map 함수를 이용하여 예측값에 roundup 함수 적용
pred = list(map(roundup, mlr.predict(test_x)))

# 예측값 확인
print(pred([:10])
```
```python
[276100, 81000, 145800, 252900, 285500, 167300, 261100, 259500, 266000, 202300]
```

## Step 2. Stochastic Gradient Descent

이제 계산량을 줄이기 위한 SGD regressor를 기존 데이터에 적용시켜보자

```python
from sklearn.preprocessing import StandardScaler

last_housing = housing.drop(['total_rooms','total_bedrooms','households',
                        'ocean_proximity'], axis = 1)

scaler = StandardScaler()
housing_scaled = scaler.fit_transform(last_housing)

housing_scaled = pd.DataFrame(housing_scaled, columns = last_housing.columns)
housing_scaled = housing_scaled.reset_index(drop = True)

train_x = housing_scaled.drop(['median_house_value'], axis = 1)
train_y = housing_scaled['median_house_value']

X, test_x, Y, test_y = train_test_split(train_x, train_y, test_size = 0.2)


sgd = SGDRegressor(learning_rate = 'constant', eta0 = 1E-8, max_iter = 200000, tol = 1E-3)
sgd.fit(np.array(X), Y)
```
