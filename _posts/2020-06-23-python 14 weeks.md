---  
title:  "12. 교차검증"  
  
categories:  
 - Python
tags:  
 - Study, Python
---

# 12. 교차검증

### 목차

-  Step 1. 교차검증
-  Step 2. Pipeline
-  Step 3. GridSearchCV
-  Step 4. Model deployment

![1](https://user-images.githubusercontent.com/59912557/84970468-b7ab1300-b155-11ea-9385-4f877aaf92ba.PNG)

 원래의 데이터 셋을 나누는 경우를 위 아래 두 가지로 생각해보자. 만약 training과 testing으로만 나눠 학습을 하게 되면 모델 검증을 위해 test set을 사용해야할 것이다. 그러나 고정된 test set을 가지고 모델의 성능을 확인하고 파라미터를 수정하는 과정을 반복하면 내가 만든 모델은 결국 test set에만 잘 동작하는 모델이 된다.

즉 test set에 과적합(overfitting)되어 다른 데이터를 가지고 예측을 수행하면 엉망인 결과가 나와버리게 된다. 이를 해결하고자 하는 거이 바로 교차 검증이다. 

test set에서의 과적합 문제는 test set이 데이터 일부분으로 고정되어 있기 때문에 발생한다. 교차 검증은 데이터의 모든 부분을 사용하여 모델을 검증하고 test set을 하나로 고정하지 않는다. 

![4](https://user-images.githubusercontent.com/59912557/84970795-5cc5eb80-b156-11ea-93ef-d319d1cc4fca.PNG)

위 그림처럼 전체 데이터 셋을 k개의 subset으로 나누고 k 번의 평가를 실행하는데, 이때 test set을 중복없이 바꾸어가면서 평가를 진행한다.

각 실행마다의 Accuracy 값을 구하고 모든 단계의 평균을 구하여 최종적으로 모델의 성능을 평가한다 . 대표적인 교차검증의 방법 중 하나인 k-fold cross validation 이다.

이제 코드 예제를 통해 공부해보자

```python
import pandas as pd
import seaborn as sns

sns.set_context('paper', font_scale = 1.5)
sns.set_style('white')

from sklearn.datatsets import load_iris

dataset = load_iris()
X, y = datatset.data, dataset.target
```
iris data는 총 4개의 독립변수로 이뤄진 데이터 셋이다. 주성분 분석을 통해 몇 개의 주성분으로 축소해야 성능이 더 좋은지 알아보자

```python
from sklearn.decomposition import PCA

# n = 2
pca = PCA(n_compnents = 2)
out_pca = pca.fit_transform(X)

# array -> dataframe
df_pca = pd.DataFrame(out_pca, columns = ['pca1', 'pca2'])
df_pca = pd.concat([df_pca, pd.DataFrame(Y, columns = ['species'])], axis = 1)
```
주성분 분석으로 차원을 축소한 데이터를 시각화하여 특징을 확인해보자

```python
sns.lmplot(x = 'pca1', y = 'pca2', data = df_pca, hue = 'species', fit_reg = False)

sns.violinplot(x = 'species', y = 'pca1', data = df_pca)

sns.violinplot(x = 'species', y = 'pca2', data = df_pca)
```
![5](https://user-images.githubusercontent.com/59912557/84971628-1d989a00-b158-11ea-9e05-98b035ee576d.png)

![6](https://user-images.githubusercontent.com/59912557/84971631-1e313080-b158-11ea-87ab-043cee6ee00e.png)
pca1은 각 species 가 0인 경우 특징을 매우 잘 보여주고 있다.

![7](https://user-images.githubusercontent.com/59912557/84971635-1ec9c700-b158-11ea-8311-f51289f50430.png)

이제 로지스틱 회귀분석을 이용해서 원자료를 사용할 경우와 주성분 분석을 한 경우를 나눠 성능을 빅해보자. 

성능 지표로는 F1-score를 사용할 것 이다. 우선 분류문제 성능 평가를 이해하기 위해선 confusion matrix부터 알아야 한다.

![2](https://user-images.githubusercontent.com/59912557/84970475-baa60380-b155-11ea-9119-5f538a1f697f.PNG)

총 결과는 TP, FN, FP, TN으로 나뉘어 진다. 

- TP : 예측값과 실제값이 모두 P 인 경우
- FN : 실제값은 P 이지만, 예측값은 N 인 경우
- FP :  실제값은 N 이지만, 예측값은 P 인 경우
- TN :  실제값과 예측값이 모두 N 인 경우

이 측도를 기반으로 F1 - score를 계산 할 수 있다.

![3](https://user-images.githubusercontent.com/59912557/84970478-bc6fc700-b155-11ea-8d31-d9f98a647210.PNG)

F1-score의 특징은 Recall 과 Precision 지표 두가지를 이용하여 조화평균을 적용한 score이다. 조화평균은 $[2 * ab/ a + b ]$의 공식을 가지고 있기 때문에 특정값이 크더라도 조화를 이루어서 작은 값에 가까운 어떠한 평균이 나타나게 된다.

그렇기 때문에 F1-score는 label이 불균형한 데이터셋에 대한 성능을 평가할 때, 매우 유용하다.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# 원자료를 이용하는 경우
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

clf = LogisticRegression()
# training
clf.fit(X_train, y_train)

# predict on test_set
y_pred = clf.predict(X_test)
# scoring by f1 score
fl = f1_score(y_test, y_pred, average = 'weighted')
print('f1 score is = ' + str(f1))
```
```python
0.955556
```
이번엔 위에서 구했던 주성분으로 성능을 평가해보자.

```python
X_train, X_test, y_train, y_test = train_test_split(df_pca[['pca1', 'pca2']], df_pca['species'], test_size = 0.3)

clf = LogisticRegression()
# training
clf.fit(X_train, y_train)

# predict on test_set
y_pred = clf.predict(X_test)
# scoring by f1 score
fl = f1_score(y_test, y_pred, average = 'weighted')
print('f1 score is = ' + str(f1))
```
```python
0.9111
```
원자료로 구했던 결과보단 좋지 않게 나왔다. 

## Step 2. Pipeline

그런데 일일이 pca, logistic을 적용해주는 과정을 해주기엔 번거롭게 느껴진다. 이럴 때 사용하는 개념이 pipeline 이다.

pipeline은 데이터 사전 처리 및 분류의 모든 단계를 포함하는 단일 개체를 만든다. 개체를 이용하여 단계를 통해 모형간 계산 순서를 정의를 해줄 수 있다. 즉 손수 입력할 필요 없이 순차적으로 계산을 해주는 개체이다.

```python
from sklearn.pipeline import Pipeline

pca = PCA()
logistic = LogisticRegressin()

# 알고리즘의 계산 순서를 정해준다.
pipe = Pipeline(steps = [('pca', pca), ('logistic', logistic)])

print(pipe.steps[0]) # pca
print(pipe.steps[1]) # logistic
```
```python
('pca', PCA(copy=True, iterated_power='auto', n_components=None, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False))

('logistic', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False))
```
이 처럼 순서대로 알고리즘을 진행함을 알 수 있다.

## Step 3. GridSearchCV

처음에 주성분 분석을 했을 때, 주성분을 2개로 축소를 시킨 후 로지스틱 분석을 실행했다. 그러나 과연 주성분 2개가 최적의 파라미터인지가 궁금하다.

이런 경우 모델의 하이퍼 파라미터를 찾아주는 library인 GridSearcCV를 사용한다.  
- 하이퍼 파라미터 매개 변수 설정
- 교차 검증을 실행
- 각 매개변수 조합의 성능을 평가하여 최적의 파라미터 산출

```python
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size = 0.30)
        
# girdsearch에서 찾을 파라미터 목록을 dictionary 형태로 저장
param_grid = {
    'pca__n_components': [2, 3, 4]
}

# 이전에 지정했던 pipeline 사용
# 5번의 교차 검증 실행
model = GridSearchCV(pipe, param_grid, iid= False, cv = 5, return_train_score = False)

# train model
model.fit(X_train, y_train)

print("Best parameter = "+ str(model.best_params_))
print('CV  score = %0.3f :'% model.best_score_)
```
```python
Best parameter = {'pca__n_components': 2}
CV  score = 0.952 
```
gridsearch 결과 파라미터가 2인 경우 가장 적합한 하이퍼 파라미터임을 보여줬다. 그럼 최적의 파라미터로 학습된 모델을 이용해 성능을 구해보자

```python
y_pred = model.predict(X_test)
f1 = f1.score(y_test, y_pred, average = 'weighted')
print('f1-score is = ' + str(f1))
```
```python
0.977724
```
## Step 4. Model Deployment

최종적으로 학습된 모델을 나중에도 사용하기 위해 저장해주는 작업이 필요하다. 이렇게 저장되는 형태를 **pickle** 이라고 한다. 

모델을 이진화 형태로 저장해주기 때문에 용량도 그렇게 크게 잡아먹지 않는다. 이제 위 모델을 pickle 형태로 저장해보자

```python
import pickle

# save model
# w = write, b = binary
pickle.dump(model, open('./model.pkl', 'wb'))

# load model
# r = read, b = binary
model_load = pickle.load(open('./model.pkl', 'rb'))
```
불러온 모델을 이용해서 다시 예측해보자

```python
y_pred = model_load.predict(X_test)
f1 = f1_socre(y_test, y_pred, average = 'weighted')
print('f1-score = '+str(f1))
```
```python
0.9777246
```
