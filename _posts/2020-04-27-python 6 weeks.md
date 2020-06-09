---  
title:  "4. 데이터 클리닝"  
  
categories:  
 - Python
tags:  
 - Study, Python
---

# 4. 데이터 클리닝
### 목차
-  Step 1. 결측값 처리
-  Step 2. 정규화, 표준화
-  Step 3. 범주형 자료 처리


오늘은 결측치가 존재하는 데이터를 이용하여 데이터클리닝 이라는 공부를 해볼 것이다. 직관적으로 데이터를 청소한다, 즉 결측치나 이상치를 처리하여 데이터를 분석하기 쉽게 만드는 것이다.

우선 결측치가 존재하는 데이터를 이용하기 위해 기존 iris data에 결측치가 존재하는 dataset을 이용할 것이다

```python
import pandas as pd
iris = pd.read_csv("..data/iris_missing_values.csv")
# 인덱스 명 변경
iris.index.name = 'record'
# 컬럼명 변경
iris.rename(columns = {iris.columns[0] : 'Sepal.Length',
					   iris.columns[1] : 'Sepal.Width',
					   iris.columns[2] : 'Petal.Length',
					   iris.columns[3] : 'Petal.Width'
					   iris.columns[4] : 'Species'}, 
					   inplace = True)
print(iris.head())
```
```python
        Sepal.Length  Sepal.Width  Petal.Length  Petal.Width Species
record                                                              
0                NaN          3.5           1.4          0.2  setosa
1                4.9          3.0           1.4          0.2  setosa
2                NaN          3.2           1.3          0.2  setosa
3                4.6          3.1           1.5          0.2  setosa
4                5.0          3.6           1.4          0.2  setosa
```

##  Step 1. 결측값 처리

파이썬에서 결측값은 아래 세 가지로 표현된다.
- NA (Not Available) :  보정이 불가능하여 삭제해야 함
- NAN(Not A Number) :  보정이나 삭제 둘 다 가능
- NULL : 아직 정해지지 않은 값 
 
그럼 위 결측값들은 어떻게 처리를 해야할까? 그 전에 우선 결측값들이 존재하는지 확인을 해야한다.

```python
# isnull() : 열의 모든 값 중에 null 값이 있는지
# null 이라면 True 반환
print(iris['Sepal.Length'].isnull())
# null 값이 하나라도 있으면 True 반환
print(iris['Sepal.Length'].isnull().values.any())
# null 값의 개수를 반환
print(iris['Sepal.Length'].isnull().values.sum())
```
```python
record
0       True
1      False
2       True
3      False
4      False
 
145    False
146    False
147    False
148    False
149    False

True

6
```

위 과정을 통해 결측값이 존재함을 확인했다. 그럼 이제 결측값을 어떻게 처리를 해볼까

```python
# 결측값을 모두 0으로 imputation(대체)
# 그러나 모두 0으로 대체하는건 바람직하지 않다
# 데이터의 특징이 변질될 수 있기 때문.
print(iris['Sepal.Length'].fillna(0.0).head())
# 결측값이 존재하는 행을 모두 제거
iris.dropna(axis = 0)
# 결측값이 존재하는 열을 모두 제거
iris.dropna(axis = 1)
```
```python
# 0으로 바뀜을 확인 가능
record
0    0.0
1    4.9
2    0.0
3    4.6
4    5.0
```

그럼 어떤 수를 이용하여 대체해주면 좋을까? 보통은 연속형 변수의 경우 평균값을 이용한다.
결측값 생성이 가능한 numpy를 이용하여 예시를 들어보자.
```python
import numpy as np
sample = [1,2,np.NaN, 4, 5]
# 결측값이 포함되면 계산 불가
np.mean(sample) 
 # 결측값을 제외하고 계산
np.namean(sample) # 3.0
# 평균값인 3을 대체
revised_sample = [1,2,3.0,4,5]
```
그럼 위 개념을 이용하여 iris의 결측값 들을 각 변수의 평균으로 변경을 해주자

```python
# mean()은 default 값으로 na를 제외하고 계산을 해준다.
# 5.870139
Sepal_Length_mean = iris['Sepal.Length'].mean()

iris['Sepal.Length'] = iris['Sepal.Length'].fillna(Sepal_Length_mean)
print(iris['Sepal.Length'].head()
```
```python
#  평균값 5.870139 으로 대체됨을 확인이 가능하다
record
0    5.870139
1    4.900000
2    5.870139
3    4.600000
4    5.000000
```
그렇담 나머지 열들도 평균값으로 대체해주자. 일일이 입력하기 번거로우니 for 문을 이용하자.

```python
# 마지막 열은 범주형 자료이기에 제외함.
for i in range(0, len(iris.columns)-1):
    mean = iris.iloc[:,i].mean()
    iris.iloc[:,i] = iris.iloc[:,i].fillna(mean)
```

자 그럼 Species 를 제외한 모든 열은 결측치를 평균값으로 대체함을 확인했다. 그럼 species엔 결측치가 몇개가 존재할까?

```python
print(iris['Species'].isnull().values.sum())
3
```
3개가 존재함을 확인했다. 그러나 이는 특정종으로 대체하기엔 타당성이 존재하지 않기에, 결측값이 존재하는 행만 삭제할 것이다.

```python
cleaned_iris = iris.dropna(axis = 0)
print(cleaned_iris.isnull().any())
```
```python
# 결측값이 모두 존재하지 않음을 확인.
Sepal.Length    False
Sepal.Width     False
Petal.Length    False
Petal.Width     False
Species         False
dtype: bool
```
sklearn library를 사용하여 위 과정도 수행이 가능하다.

```python
from sklearn.impute import SimpleImputer

iris = pd.read_csv("./data/iris_missing_values.csv")
# Species 행 제외
iris_a = iris.iloc[:, :4]

#  평균값 imputer 생성
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
# array 형태로 변경
out_imp = imputer.fit_tranasform(iris_a)
print(out_imp[:5])
```
```python
[[5.87013889 3.5        1.4        0.2       ]
 [4.9        3.         1.4        0.2       ]
 [5.87013889 3.2        1.3        0.2       ]
 [4.6        3.1        1.5        0.2       ]
 [5.         3.6        1.4        0.2       ]]
```

arrary 형태로 변경되었기에, dataframe으로 다시 변경을 해주자
```python
iris_imp = pd.DataFrame(data = out_imp, columns = iris_a.columns)
print(iris_imp.head())

# 아까 제외했던 species 변수를 합쳐주자
iris_imp = pd.concat([iris_imp, iris['species']], axis = 1)
print(iris_imp.head())

# Species에 존재하는 na 행 제거
iris_imp = iris_imp.dropna(axis = 0)
print(iris_imp.isnull().values.any())
```
```python
   sepal length in cm  sepal width in cm  petal length in cm  petal width in cm
0            5.870139                3.5                 1.4                0.2
1            4.900000                3.0                 1.4                0.2
2            5.870139                3.2                 1.3                0.2
3            4.600000                3.1                 1.5                0.2
4            5.000000                3.6                 1.4                0.2

   sepal length in cm  sepal width in cm  ...  petal width in cm  species
0            5.870139                3.5  ...                0.2   setosa
1            4.900000                3.0  ...                0.2   setosa
2            5.870139                3.2  ...                0.2   setosa
3            4.600000                3.1  ...                0.2   setosa
4            5.000000                3.6  ...                0.2   setosa

[5 rows x 5 columns]
# 결측값이 없음을 확인
False
```

그러나 교수님 말씀으론 library보단 실제로 코딩을 통해 하나하나 확인하는 것이 바람직하다고 한다.

## Step 2. 정규화, 표준화

데이터 셋을 보면 변수마다 가지고 있는 구간이 모두 상이하며, 측정 단위도 다르기에 분석함에 어려움이 있다. 이럴 경우 사용하는 개념이 정규화와 표준화이다.

### 1. 정규화(Normalization)

정규화는 해당 구간을 0 ~ 1로 바꿔줌으로 범위를 통일 시켜주며  대표적 방식은 Min-Max Normalization 이다.

- x_new = ( x - min(x)) / ( max(x) - min(x))

```python
# 변수명 변경

iris_imp.rename(columns = {iris.columns[0] : 'Sepal.Length',  
                       iris.columns[1] : 'Sepal.Width',
                       iris.columns[2] : 'Petal.Length',
                       iris.columns[3] : 'Petal.Width',
                       iris.columns[4] : 'Species'}, 
                       inplace = True)

Sepal_Length_min = iris_map['Sepal.Length'].min()
Sepal_Length_max = iris_map['Sepal.Lenght'].max()

# 정규화된 새로운 column 생성
iris_imp['Sepal.Length.norm'] = (iris_imp['Sepal.Length'] - Sepal_Length_min)/(Sepal_Length_max - Sepal_Length_min)
print(iris_imp.head())
``` 
```python
   Sepal.Length  Sepal.Width  ...  Species  Sepal.Length.norm
0      5.870139          3.5  ...   setosa           0.436150
1      4.900000          3.0  ...   setosa           0.166667
2      5.870139          3.2  ...   setosa           0.436150
3      4.600000          3.1  ...   setosa           0.083333
4      5.000000          3.6  ...   setosa           0.194444
```
자 그럼 모든 변수에 따른 정규화 열을 생성해보자. 이것도 마찬가지로 for문을 이용해보자.

```python
columns = iris.colums[0:4]
for j in columns:

    max_x = iris_imp.loc[:,j].max()
    min_x = iris_imp.loc[:,j].min()
    iris_imp[j+'_norm'] = ((iris_imp.loc[:,j] - min_x) / (max_x - min_x))

print(iris_imp.describe().transpose()) 
```
```python
                   count      mean       std  ...       50%       75%  max
Sepal.Length       147.0  5.870139  0.823048  ...  5.800000  6.400000  7.9
Sepal.Width        147.0  3.042177  0.422472  ...  3.000000  3.300000  4.2
Petal.Length       147.0  3.803401  1.753916  ...  4.400000  5.100000  6.9
Petal.Width        147.0  1.218367  0.757973  ...  1.300000  1.800000  2.5
Sepal.Length_norm  147.0  0.436150  0.228624  ...  0.416667  0.583333  1.0
Sepal.Width_norm   147.0  0.473717  0.192033  ...  0.454545  0.590909  1.0
Petal.Length_norm  147.0  0.475153  0.297274  ...  0.576271  0.694915  1.0
Petal.Width_norm   147.0  0.465986  0.315822  ...  0.500000  0.708333  1.0
```

마찬가지로 sklearn의 정규화 함수를 이용해 위 과정이 가능하다.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler
out_scaled = scaler.fit_trasnform(iris[iris.columns[:-1]])
# 결과값이 마찬가지로 array 형태
# dataframe으로 변환 필요
df_out_scaled = pd.DataFrame(data = out_scaled, columns = iris.colums[:-1])
# species 컬럼 붙이기
df_out_scaled = pd.concat([df_out_scaled, iris['species']], axis = 1)
print(df_out_scaled.describe().transpose())
```
```python
                    count      mean       std  ...       50%       75%  max
sepal length in cm  144.0  0.436150  0.231010  ...  0.416667  0.583333  1.0
sepal width in cm   147.0  0.473717  0.192033  ...  0.454545  0.590909  1.0
petal length in cm  147.0  0.475153  0.297274  ...  0.576271  0.694915  1.0
petal width in cm   147.0  0.465986  0.315822  ...  0.500000  0.708333  1.0
```

### 표준화(Standardization)

표준화는 해당 데이터가 평균으로 부터 얼마나 떨어져 있는지 나타낼 수 있으며, 이를 통해 데이터가 특정 범위 안에 머물도록 한다. 정규화와 달리 범위는 음수부터 양수까지 존재한다.
- (x - mean(x)) / std(x)

```python
Sepal_Length_mean = iris_imp['Sepal.Length'].mean()
Sepal_Length_std = iris_imp['Sepal.Length'].std()
iris_imp['Sepal.Length.stand'] = (iris_imp['Sepal.Length'] - Sepal_Length_mean) / Sepal_Length_std
```
```python
columns = iris.colums[0:4]
for j in columns:

    mean_x = iris_imp.loc[:,j].mean()
    std_x = iris_imp.loc[:,j].std()
    iris_imp[j+'_std'] = ((iris_imp.loc[:,j] - mean_x) / (std_x))

print(iris_imp.describe().transpose()) 
```
```python
			       count          mean       std  ...       50%       75%       max
Sepal.Length_std   147.0 -1.804868e-15  1.000000  ... -0.085218  0.643779  2.466273
Sepal.Width_std    147.0 -1.896348e-15  1.000000  ... -0.099833  0.610272  2.740589
Petal.Length_std   147.0 -1.223511e-15  1.000000  ...  0.340152  0.739259  1.765534
Petal.Width_std    147.0 -6.563155e-16  1.000000  ...  0.107699  0.767353  1.690868
```

그럼 언제 정규화를 쓰고, 표준화를 써야하나?
- 정규화는 음수는 존재하지 않음.
- 표준화는 음수, 양수 모두 존재
- 
그 뒤에 사용 알고리즘이 양수를 쓰는지, 음/양수 둘 다 쓰는지 보고 판단

##   Step 3. 범주형 자료 처리

컴퓨터는 자료를 처리할 경우, 모두 숫자 형태로 처리하기 때문에 string으로 되어있는 범주형 변수를 숫자로 변경하는 과정을 해보자.

우선 이번엔 사람의 나이, 키, 몸무게 등 신체 사이즈로 점프한 길이를 나타내는 데이터셋을 이용해보자.

```python
jump = pd.read_csv('./data/long_jump.csv')
print(jump)
print(jump.describe().transpose())
print(jump['Shoe Size'].values.unique())
```
```python
    Person  Age  Height  Weight  ...  Jersey Color Jersey Size Shoe Size  Long Jump
0   Thomas   12    57.5    73.4  ...          blue       small         7       19.2
1     Jane   13    65.5    85.3  ...         green      medium        10       25.1
2   Vaughn   17    71.9   125.9  ...         green       large        12       14.3
3     Vera   14    65.3   100.5  ...           red      medium         9       18.3
4  Vincent   18    70.1   110.7  ...          blue       large        12       21.1
5  Lei-Ann   12    52.3    70.4  ...          blue       small         7       10.6

                     count       mean        std  ...    50%      75%    max
Age                    6.0  14.333333   2.581989  ...  13.50   16.250   18.0
Height                 6.0  63.766667   7.514963  ...  65.40   68.950   71.9
Weight                 6.0  94.366667  21.885855  ...  92.90  108.150  125.9
Training Hours/week    6.0   5.900000   4.164613  ...   7.20    8.650   10.5
Shoe Size              6.0   9.500000   2.258318  ...   9.50   11.500   12.0
Long Jump              6.0  18.100000   5.097843  ...  18.75   20.625   25.1
```
현재 확실한 범주형 변수는 'Jersey Color', 'Jersey Size'가 존재하며, 'Shoe Size'는 ordinal data로 생각할 수 있기에 연속형으로 판단하기 어렵다.

그럼 범주형 변수의 string 값을 숫자 형태로 변형해보자
```python
jump['Jersey_Size_trans'] = jump['Jersey Size']

for idx, val in enumerate(jump['Jersey Size']):
	if val == 'small':
		jump.loc[idx, 'Jersey_Size_trans'] = 0
	elif val == 'medium':
		jump.loc[idx, 'Jersey_Size_trans'] = 1
	elif val == 'large':
		jump.loc[idx, 'Jersey_Size_trans'] = 2

print(jump.loc[:,['Jersey Size', 'Jersey_Size_trans']])
```
```python
  Jersey Size  Jersey_Size_trans
0       small                 0
1      medium                 1
2       large                 2
3      medium                 1
4       large                 2
5       small                 0
```
마찬가지로 Jersey Color도 바꿔주면 된다.
```python
jump['Jersey_Color_trans'] = jump['Jersey Color']

for idx, val in enumerate(jump['Jersey Color']):
	if val == 'blue':
		jump.loc[idx,'Jersey_Color_trans'] = 0
	elif val == 'green':
		jump.loc[idx, 'Jersey_Color_trans'] = 1
	elif val == 'red':
		jump.loc[idx, 'Jersey_Color_trans'] = 2

print(jump.loc[:,['Jersey Color', 'Jersey_Color_trans']])
```
```python
  Jersey Color  Jersey_Color_trans
0         blue                   0
1        green                   1
2        green                   1
3          red                   2
4         blue                   0
5         blue                   0
```
