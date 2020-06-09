
---  
title:  "5. 인코딩 그리고 변수 선택"  
  
categories:  
 - Python
tags:  
 - Study, Python
---

# 5. 인코딩 그리고 변수 선택
### 목차
-  Step 1. One-hot Encoding
-  Step 2. Label-encoding
-  Step 3. Feature selection(filtering)

오늘은 long_jump.csv 데이터 셋을 사용하여 실습을 진행할 예정이다.

```python
import pandas as pd
jump = pd.read.csv("../data/long_jump.csv)"
jump.set_index('Person', inplace = True)
print(jump)
```
```python
         Age  Height  Weight  ...  Jersey Size Shoe Size Long Jump
Person                        ...                                 
Thomas    12    57.5    73.4  ...        small         7      19.2
Jane      13    65.5    85.3  ...       medium        10      25.1
Vaughn    17    71.9   125.9  ...        large        12      14.3
Vera      14    65.3   100.5  ...       medium         9      18.3
Vincent   18    70.1   110.7  ...        large        12      21.1
Lei-Ann   12    52.3    70.4  ...        small         7      10.6

[6 rows x 8 columns]
```

##  Step 1. One-hot Encoding

위 데이터를 살펴보면 Jersey Size 컬럼은 연속형 데이터가 아닌, 범주형 데이터로 이루어져있다. 

각 범주를 1,0 으로 표현하기 위해 범주형 변수의 해당 값을 클래스로 만들고 해당 되는 컬럼에 1을 부여하고 나머진 0인 벡터로 만드는 one-hot encoding을 공부해보자.

custom coding과 Scikit-learn library를 각각 이용해 one-hot encoding을 진행 해보자

### 1) custom coding

우선  Jersey Size 컬럼에 어떤 범주값들이 있는지 확인을 해보자

```python
print(jump['Jersey Size'].unique())
```
```python
['small' 'medium' 'large']
```
총 3 가지의 범주값을 가짐을 확인했다. 이제 해당 범주값들을 0,1 의 값만 가지는 벡터로 바꾸는 과정을 진행해보자.

우선 우리가 최종적으로 얻어야 하는 결과값은 아래와 같다.
```python
# 'small' ==> [1, 0, 0]
# 'medium' ==> [0, 1, 0]
# 'large' ==> [0, 0, 1]
```

위와 같이 표현하기 위해서는 비교연산자를 이용해 실제값과 비교를 통해 만드는 과정이 필요하다. 아래와 같이 말이다,

```python
js_cat = jump['Jersey Size'].unique()
print(js_cat == 'small')
print((js_cat == 'small') * 1)
```
```python
[ True False False]
[ 1 0 0 ]
```
boolean 값은 true, false 이지만 실제 숫자로 표현하면 1,0 값으로 이루어지기에 1을 곱하면 숫자로 표현이 가능하다.

위의 원리로 해당 컬럼을 바꿔주어 각 범주별 컬럼을 생성하여 one-hot-encoding을 완성시킨다.

```python
js_raw = list() # 변환된 인코딩을 담을 리스트
for idx, val in enumerate(jump['Jersey Size']):
	js_raw.append((js_cat == val) * 1)
print(js_raw)
```
```python
[array([1, 0, 0]), array([0, 1, 0]),
 array([0, 0, 1]), array([0, 1, 0]),
 array([0, 0, 1]), array([1, 0, 0])]
```

array 형태로 변환되기에 dataframe으로 변환하는 과정이 필요하다. 

```python
js_one_hot = pd.DateFrame(data = js_raw)
# 각 컬럼 이름 생성
js_one_hot.rename(columns = {0 : 'Jearsey Size_small', 
                             1 : 'Jearsey Size_medium', 
                             2 : 'Jearsey Size_large'}, inplace = True)
# 기존 데이터 프레임의 인덱스 이름 가져오기
js_one_hot.index = jump.index

# 기존 데이터 프레임과 one-hot-encoding한 데이터 프레임을 합치기
jump_js_one_hot = pd.concat([jump, js_one_hot], axis =1) # axis = 1 : 열기준

# Jersey Size 컬럼 버리기
jump_js_one_hot.drop('Jersey Size', axis = 1, inplace = True)
print(jump_js_one_hot.transpose())
```

```python
Person              Thomas   Jane Vaughn   Vera Vincent Lei-Ann
Age                     12     13     17     14      18      12
Height                57.5   65.5   71.9   65.3    70.1    52.3
Weight                73.4   85.3  125.9  100.5   110.7    70.4
Training Hours/week    6.5    8.9    1.1    7.9    10.5     0.5
Jersey Color          blue  green  green    red    blue    blue
Shoe Size                7     10     12      9      12       7
Long Jump             19.2   25.1   14.3   18.3    21.1    10.6
Jearsey Size_small       1      0      0      0       0       1
Jearsey Size_medium      0      1      0      1       0       0
Jearsey Size_large       0      0      1      0       1       0

```

### 2) Scikit-learn: OneHotEncoder

이번엔 library를 이용하여 one-hot-encoding을 진행해보자

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()

jump = pd.read_csv("./data/long_jump.csv")
cats = jump['Jersey Size']

out_encoder = encoder.fit_transform(cats)
print(out_encoder)
```
```python
[[0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 0. 1.]]
```
```python
# 각 컬럼명 지정
new_cols = encoder.get_feature_names(cats).tolist()
print(new_cols)
```
```python
['Jersey Size_large', 'Jersey Size_medium', 'Jersey Size_small']
```

이제 변환한 one-hot encoding 결과를 기존 데이터 프레임에 합치면 끝이다

```python
# array 형태 dataframe으로 변경하기
jump_enc = pd.DataFrame(data = out_encoder, colums = new_cols)
jump_enc.index = jump.index

jump.drop(cats, axis = 1, inplace =True)
jump = pd.concat([jump, jump_enc], axis =1)
print(jump.transpose())
```
```python
Person              Thomas    Jane Vaughn    Vera Vincent Lei-Ann
Age                     12      13     17      14      18      12
Height                57.5    65.5   71.9    65.3    70.1    52.3
Weight                73.4    85.3  125.9   100.5   110.7    70.4
Training Hours/week    6.5     8.9    1.1     7.9    10.5     0.5
Jersey Color          blue   green  green     red    blue    blue
Jersey Size          small  medium  large  medium   large   small
Shoe Size                7      10     12       9      12       7
Long Jump             19.2    25.1   14.3    18.3    21.1    10.6
```

## Step 2. Label encoding

이전에 0,1 으로만 변환하는 one hot encoding과 다른 label 번호로 encoding을 해주는 개념이다. 이것도 sklearn의 library를 사용할 것이다.

```python
from sklearn import preprocessing
enc = preprocessing.LabelEncoder()

out_enc = enc.fit_transform(1,2,5,2,4,2,5])
print(out_enc)
```
```python
# 이렇듯 숫자 종류에 따라 label이 부여됨을 확인할 수 있다
[0 1 3 1 2 1 3]
```
```python
out_enc = enc.fit_transform(["blue", "red", "blue", "green", "red", "red"])
print(out_enc)
```
```python
# 문자열 타입도 바뀜을 확인할 수 있다.
[0 2 0 1 2 2]
```

## Step 3. Feature selection(filtering)

데이터 분석을 할 때 변수가 너무 많으면 좋지 않은 결과가 나오는 경우가 많으며, 과적합 현상 또한 발생할 수 있다. 그래서 필요한 변수만 선택하여 분석을 하는 과정을 feature selection이라고 칭한다.

### 1) Variance Filtering

이번에 공부할 것은 Variance Filtering 이다. 
- 여러 개의 독립변수가 존재할 때, 각 변수의 분산의 크기가 다르다고 가정
- 그 변수 중에 분산이 작은 변수는 종속 변수에게 미치는 영향이 적을 것이라 판단
- 그래서 그 변수를 제거하는 filtering 수행
- VarianceThreshold : 특정 한 값(Threshold) 보다 작은 변수들은 제외

```python
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold()

iris = pd.read_csv("../data/iris.csv")
iris.index.name = 'record'
print(iris.head())

# define columns to filter
cols = ['sepal length in cm',
        'sepal width in cm',
        'petal length in cm',
        'petal width in cm']
 
selector.fit(iris[cols])
# 각 열의 분산 확인
print(selector.variances_)
```
```python
[0.68112222 0.18675067 3.09242489 0.57853156]
```

이제 분산의 기준을 설정하여 해당 기준에 미치지 못하면 제거를 한다.

```python
selector.set_parms(threshold = 0.6)

out_sel = selector.fit_transform(iris[cols])
print(selector.variances_)
# 0.6인 threshold를 넘긴 변수들 확인
print(selector.get_support())
print(out_sel[:10])
```
```python
[0.68112222 0.18675067 3.09242489 0.57853156]
[ True False  True False]
[[5.1 1.4]
 [4.9 1.4]
 [4.7 1.3]
 [4.6 1.5]
 [5.  1.4]
 [5.4 1.7]
 [4.6 1.4]
 [5.  1.5]
 [4.4 1.4]
 [4.9 1.5]]
```

0, 2번째 있는 컬럼이 해당 기준을 넘었음을 확인했으므로, dataframe을 다시 인덱싱 해준다

```python
iris_sel = iris.iloc[:, [0,2]]
# 기존에 있었던 species 열 concat
iris_sel = pd.concat([iris_sel, iris['Species']], axis =1)
iris_sel.set_index('record', inplace = True)
print(iris_sel.head())
```
```python
        sepal length in cm  petal length in cm species
record                                                
0                      5.1                 1.4  setosa
1                      4.9                 1.4  setosa
2                      4.7                 1.3  setosa
3                      4.6                 1.5  setosa
4                      5.0                 1.4  setosa
```

### 2) Correlation filtering

종속변수와 독립변수간의 상관계수로도 필터링이 가능하다.

보스턴 집 값과 관련 독립변수들로 이루어져있는 데이터셋으로 살펴보자

```python
# load boston dataset
boston = pd.read_csv("./data/boston.csv")
boston.index.name = 'record'

cor = boston.corr()
# 모든 변수간의 상관계수 확인
print(cor)
```
```python
        record      CRIM        ZN  ...         B     LSTAT      MEDV
record   1.000000  0.404600 -0.103393  ... -0.295041  0.258465 -0.226604
CRIM     0.404600  1.000000 -0.199458  ... -0.377365  0.452220 -0.385832
ZN      -0.103393 -0.199458  1.000000  ...  0.175520 -0.412995  0.360445
INDUS    0.399439  0.404471 -0.533828  ... -0.356977  0.603800 -0.483725
CHAS    -0.003759 -0.055295 -0.042697  ...  0.048788 -0.053929  0.175260
NOX      0.398736  0.417521 -0.516604  ... -0.380051  0.590879 -0.427321
RM      -0.079971 -0.219940  0.311991  ...  0.128069 -0.613808  0.695360
AGE      0.203784  0.350784 -0.569537  ... -0.273534  0.602339 -0.376955
DIS     -0.302211 -0.377904  0.664408  ...  0.291512 -0.496996  0.249929
RAD      0.686002  0.622029 -0.311948  ... -0.444413  0.488676 -0.381626
TAX      0.666626  0.579564 -0.314563  ... -0.441808  0.543993 -0.468536
PTRATIO  0.291074  0.288250 -0.391679  ... -0.177383  0.374044 -0.507787
B       -0.295041 -0.377365  0.175520  ...  1.000000 -0.366087  0.333461
LSTAT    0.258465  0.452220 -0.412995  ... -0.366087  1.000000 -0.737663
MEDV    -0.226604 -0.385832  0.360445  ...  0.333461 -0.737663  1.000000
```

우리가 필요한건 종속변수 MEDV와 다른 독립변수간의 상관계수이다.

```python
# abs 절대값으로 바꿔주어 음양 상관없이 크기를 따진다.
cor_target = abs(cor['MEDV'])
print(cor_target)
```
```python
record     0.226604
CRIM       0.385832
ZN         0.360445
INDUS      0.483725
CHAS       0.175260
NOX        0.427321
RM         0.695360
AGE        0.376955
DIS        0.249929
RAD        0.381626
TAX        0.468536
PTRATIO    0.507787
B          0.333461
LSTAT      0.737663
MEDV       1.000000
Name: MEDV, dtype: float64
```

마찬가지로 기준점을 설정하여 기준점에 못 미치는 변수는 제거하자

```python
selected_cols = cor_target[cor_target > 0.6]
print(selected_cols)
```
```python
RM       0.695360
LSTAT    0.737663
MEDV     1.000000
```

해당 변수만 남기기 위해 dataframe을 변경하자

```python
boston_sel = boston[selected.cols.index]
print(boston_sel.head())
```
```python
          RM  LSTAT  MEDV
record                    
0       6.575   4.98  24.0
1       6.421   9.14  21.6
2       7.185   4.03  34.7
3       6.998   2.94  33.4
4       7.147   5.33  36.2
```
