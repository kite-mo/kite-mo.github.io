---  
title:  "11. 의사결정나무"  
  
categories:  
 - Python
tags:  
 - Study, Python
---

# 11. 의사결정나무 

### 목차

-  Step 0. 의사결정나무 개념
-  Step 1. 의사결정나무 적용

## Step 0 . 의사결정나무 개념

의사결정나무(Decision tree)는 데이터를 분석하여 이들 사이에 존재하는 패턴을 예측 가능한 규칙들의 조합으로 나타내며 그 모양이 나무와 같다고 해서 의사결정나무라고 불린다. 분류만 가능할 것 같지만 회귀도 마찬가지로 가능하다.  즉 범주형, 연속형 변수 모두 사용이 가능하다는 말이다.



![1](https://user-images.githubusercontent.com/59912557/84588546-50464800-ae63-11ea-9d92-04802ebfd378.PNG)

위 그림은 의사결정나무의 결과이다. 구조의 명칭은 
 - 뿌리 마디(root node) : 부모 마디 라고도 불리며 맨 위에 위치한 노드
 - 중간 마디(intermediate node) : 뿌리 마디와 끝 마디에 위치한 노드 
 - 끝 마디(terminal node = leaf node) : 자식 마디라고도 불리며 맨 아래에 위치한 노드
 - 깊이(depth)  : 노드 사이의 화살표 층의 개수

위 그림과 같이 하나의 노드에서 여러 노드로 분기하는 과정을 partioning이라고 하며 root node에서 leaf node를 생성하기 위해 recursive partitioning을 진행한다. 

의사결정나무는 root node의 종속변수를 기반으로 도수 분포표를 생성하여 leaf 노드 처럼 한가지 범주만 가지도록 분할수 있도록 분류를 수행하는 과정을 수행한다. 

분할 수행 과정에서 leaf node로 갈 수록 불순도(impurity)는 낮아지고 순도(purity)s는 높아진다. 
불순도를 나타내는 지표는 아래와 같다.

![2](https://user-images.githubusercontent.com/59912557/84588547-51777500-ae63-11ea-8d39-f9b73a4526d2.PNG)

아래 그림으로 예를 들면 두 번째 테이블인 play만 있는 불순도는 0 으로 분류가 잘 되었음을 나타낸다. 

![3](https://user-images.githubusercontent.com/59912557/84588549-51777500-ae63-11ea-9c97-9ff9ad02589b.PNG)

즉 의사결정나무는 원 데이터를 이용해 impurity meaure에 의해 partioning을 수행한다. 그 후  leaf 또는 branch를 삭제를 해나가는 과정인 가지치기(prunning)을 수행하며 impurity가 가장 적은 나무를 생성하는 것이 목적이다. 

이제 데이터를 이용해서 의사결정나무를 진행해보자.

## Step 1. 의사결정나무 적용

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# iris data load
iris = load_iris()

# divide X,Y
y = iris.target
x = iris.data[:, 2:]
feature_names = iris.feature_names[2:]

# decision tree
tree = DecisionTreeClassifier(criterion = 'entrophy', max_depth = 1, random_state = 0)
# training
tree.fit(x,y)
```
학습된 모델의 결과를 확인하기 위해 그림으로 표현할 함수를 구현하자
```python
import io
import pydot
from IPython.core.display import Image
from sklearn.tree import export_graphviz

def draw_decision_tree(model):
	dot_buf = io.StringIO()
    export_graphviz(model, out_file = dot_buf, feature_names = feature_names)
    graph = pydot.graph_from_dot_data(dot_buf.getvalue())[0]
    image = graph.create_png()
    return Image(image)

# 결과 확인
draw_decision_tree(tree)
```
![5](https://user-images.githubusercontent.com/59912557/84588551-52100b80-ae63-11ea-8aff-469bf76eea5a.png)

부모 마디를 기준으로 left node는 petal width가 0.8 이하이면 첫번째 범주(setosa)임을 알 수 있다. 이 정보만 알아도 setosa 인지 분류 가능하다. 반대로 right node는 petal width가 0.8 초과하면 나머지 두 개의 범주임을 알 수 있다.

그럼 depth를 2로 설정하면 어떻게 될까?
```python
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 2, random_state = 0)
tree.fit(X, y)

draw_decision_tree(tree)
```

![6](https://user-images.githubusercontent.com/59912557/84588886-0743c300-ae66-11ea-995b-5f9bab3e21f8.png)

두번째 가지에서 petal width가 1.75 이하인 경우 두 번째 범주인 versicolor 라고 분류 할 수 있음. 그러나 모두 분류된 경우는 아니다. 이렇듯 여러 depth를 해보며 적절한 branch의 길이를 확인해야 한다.

이번엔 cancer data를 이용해서 의사결정나무를 적용시켜보자.

```python
# 데이터 전처리
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

# scaling function
scaler = StandaradScaler()
# data load
cancer = load_breast_cancer()

cancer_data = scaler.fit_transform(cancer.data)

# split train, test set
# training, test datset 구분
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, 
                                                    test_size = 0.1,
                                                    stratify = cancer.target, 
                                                    random_state = 42)
feature_names = cancer.feature_names
```
```python
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 1, random_state = 0)

tree.fit(X_train, y_train)

# train, test data 별 결과 정확도 확인
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

draw_decision_tree(tree)
```
```python
Accuracy on training set: 0.920
Accuracy on test set: 0.912
```
![8](https://user-images.githubusercontent.com/59912557/84588890-0874f000-ae66-11ea-8b30-f7baaa71598b.png)

worst_perimeter이 -0.039 이하인 경우 암에 대해 음성인 경우로 더 많이 분류됨을 확인할 수 있다.

그렇담 depth가 2인 경우엔 어떨까?
```python
tree = DecisionTreeClassifier(criterion = 'entropy', max_depth = 2, random_state = 0)
tree.fit(X_train, y_train)

print("Accurary on training set : {:3.f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))

draw_decision_tree(tree)
```
```python
# training 결과 정확도는 높아졌으나, test data의 결과 정확도는 낮아짐
Accuracy on training set: 0.928
Accuracy on test set: 0.860
```
![9](https://user-images.githubusercontent.com/59912557/84588891-090d8680-ae66-11ea-9281-9d2990a986b0.png)

leaf node의 양 쪽 노드를 보면 entrophy가 0.1 근처로 낮아졌음을 확인할 수 있다. 

이번엔 불순도 기준을 entrophy가 아닌 gini 지수로 확인해보자

```python
tree = DecisionTreeClassifier(criterion = 'gini', max_depth = 1, random_state = 0)
tree.fit(X_train, y_train)

print('Accuracy on training set: {:3.f}'.format(tree.score(X_train, y_train)))
print('Accuracy on test set:{:3.f}'.format(tree.score(Y_test, y_test)))
draw_decision_tree(tree)
```
```python
# gini metric을 사용한 경우 더 좋은 성능을 보임
Accuracy on training set: 0.922
Accuracy on test set: 0.930
```
![10](https://user-images.githubusercontent.com/59912557/84588892-090d8680-ae66-11ea-8b54-af6ff549617e.png)


