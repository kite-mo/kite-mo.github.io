---
title:  "선형 회귀 모델 공부"

categories:
  - Machine Learning
tags:
  - Study, Machine Learning

---

# 선형회귀 모델이란?
### 목차

-  Step1. 선형 회귀란 무엇?
	* Step 1.1 단순 선형 회귀 분석
	* Step 1.2 다중 선형 회귀 분석
-  Step2. 가설(Hypothesis) 이란
-  Step3. 비용 함수( Cost function)
-  Step4.  Optimizer : 경사하강법(Gradient Descent)


## Step1. 선형 회귀란 무엇?

다른 변수의 값을 변하게 하는 입력변수를 $x$, 해당 입력변수에 의해서 값이 종속적으로 변하는 변수를 $y$라고 함

이때 변수 $x$는 독립적으로 변할 수 있으나, $y$는 해당 입력변수에 의해 종속적으로 결정됨으로 

- $x$ : 독립변수
- $y$ : 종속변수    

라고 명칭한다. **선형회귀**는 한 개 이상의 독립 변수와 종속 변수간의 선형 관계를 모델링 한다. 독립 변수 개수에 따라 명칭이 달라지는데

### step 1.1 단순 선형 회귀

$y = {Wx +b}$

위 수식은 단순 선형 회귀의 수식을 보여준다. 여기서 주목할 점은 $x$의 계수인 $W$이다. 해당 값을 머신 러닝에서는 가중치(weight)라고 칭하며 $b$는 편향(bias)라고 한다.
(직선 방정식에서 기울기와 절편을 의미한다고 보면 된다)

예를 들면 성적을 $y$로 가정하고 공부 시간을 $x$로 가정하면 공부 시간을 늘릴 수록 성적이 잘 나오거나, 적을 수록 성적이 잘 안 나온다면 해당 수식은 선형성을 잘 표현한다고 할 수 있다. 

### step 1.2 다중 선형 회귀

$y = {W_1x_1 + W_2x_2 + ... W_nx_n + b}$

그러나 현실적으로 성적은 공부 시간에만 영향을 받지 않고 다른 여러 요인들이 존재할 수 있다. TV, 친구, 게임 등등 이렇게 다수의 원인들을 가지고 성적 $y$를 예측하고 싶을때 위 수식과 같이 표현할 수 있다.

이를 **다중 선형 회귀**라고 한다.

##  Step2. 가설(Hypothesis) 이란

이번엔 실제 값들인 Height를 독립 변수, Weight를 종속 변수로 설정하여 좌표 평면위에 표현한 뒤, 만약 다른 Height 값을 넣었을 때  발생하는 Weight 값을 알고 싶다.

실제 값들을 좌표 평면 위에 표현하고 그것들을 가장 잘 설명할 수 있는 **선(line)**으로 그린다면 새로운 Height를 바탕으로 Weight 값을 예측할 수 있다. 아래의 그림과 같이 말이다.

![](https://i0.wp.com/hleecaster.com/wp-content/uploads/2019/12/linear01.jpg?w=1200)

이렇듯 $x$와 $y$으 의 관계를 유추하기 위해 수학적으로 식을 세우게 되는데 머신러닝에서는 이러한 식을 **가설(Hypothesis)** 라고 한다.

$H(x) = {Wx + b}$

그러나 위의 직선은 $W$값과 $b$의 값에 따라 천차만별로 그려질 수 있음으로 결국 선형 회귀는 주어진 데이터로 부터 독립 변수와 종속 변수간의 관계를 가장 잘 나타내는 직선을 그리는 일이 된다.

## Step 3.  비용 함수( Cost function )

앞서 말했듯 실제 데이터들의 규칙을 가장 잘 표현하는 $W$와 $b$를 찾는 것이 중요한데, 머신러닝에서는 실제값과 가설로 부터 얻은 예측값의 오차를 계산하는 식을 세우고, 이 식을 최소화하는 최적의   $W$와 $b$를 도출한다.

이때 실제값과 예측값에 대한 오차에 대한 식을 **비용 함수(Cost function)** 라고 한다. 

비용 함수는 단순히 실제값과 예측값에 대한 오차를  표현하는 것이 아닌, **예측값의 오차를 줄이는 일에 최적화 된 식** 이어야 한다.
그러므로 다양한 비용 함수들이 존재하며 회귀 문제에 경우에는 주로 **평균 제곱 오차(Mean Sqaured Error, MSE)** 가 사용된다.

평균 제곱 오차를 $W$와 $b$에 의한 비용 함수로 정의하여 수식을 표현하면

$cost(W, b) = \frac{1}{n} \sum_{i=1}^{n} \left[y^{(i)} - H(x^{(i)})\right]^2$

위와 같이 나타낼 수 있다.

$y$ = 실제값,  $H(x)$ = 예측값이며 두 값 사이의 오차가 작을 수록 MSE는 작아지게 된다.

즉, $cost(W, b)$를 최소가 되게 만드는  $W$와 $b$를 구하게 된다면 관계를 가장 잘 나타내는 직선을 그릴 수 있게 된다.

##  Step4.  Optimizer : 경사하강법(Gradient Descent)

선형 회귀 뿐아니라 머신러닝, 딥러닝 학습은 결국 모두 **Cost function을 최소화** 하는 parameter들을 찾는 것이 목적이다. 
이때 사용되는 알고리즘을 **Optimizer or 최적화 알고리즘**이라 칭한다. 

Optimizer를 통해서 해당 parameter들을 찾는 과정을 학습(training)이라고 한다. 가장 기본적인 optimizer인 **경사 하강법(Gradient Descent)** 에 대해 공부해보자.

경사 하강법을 이해하기 위해선 cost와 parameter인 기울기 $W$의 관계를 알아보자. 

편향 $b$가 없이 단순히 가중치 $W$만을 사용한 $y=Wx$라는 가설 $H(x)$를 가지고, 경사 하강법을 수행한다고 해보자. 

$cost(W)$= cost 

이에 따라 $W$와 cost의 관계를 그래프로 표현하면 아래와 같다.

![](https://wikidocs.net/images/page/21670/%EA%B8%B0%EC%9A%B8%EA%B8%B0%EC%99%80%EC%BD%94%EC%8A%A4%ED%8A%B8.PNG)

기울기가 무한대로 커지거나 작아지면 cost 값 또한 무한대로 커지게 된다. 위 그래프에서 cost가 가장 작은 값은 볼록한 부분의 $W$ 지점이므로 우리는 이 지점을 찾아야 한다.

머신 러닝에서는 랜덤의 $W$값을 설정한 뒤, 점차 맨 아래의 볼록한 부분을 향해 $W$값을 수정해 나가고 이를 가능하게 하는 것이 **경사 하강법** 이다. 경사 하강법은 순간 변화율 또는 접선에서의 기울기의 개념인 미분을 이용한다.

![](https://wikidocs.net/images/page/21670/%EC%A0%91%EC%84%A0%EC%9D%98%EA%B8%B0%EC%9A%B8%EA%B8%B01.PNG)

즉 위의 그림과 같이 접선의 기울기가 0이 되는 지점, 또한 미분값이 0이 되는 지점을 cost가 최소화 되는 지점을 말한다. 

경사 하강법의 원리는 비용 함수를 미분하여 현재 $W$에서의 접선의 기울기를 구하고, 접선의 기울기가 낮은 방향으로 $W$의 값을 변경하고 다시 미분하는 과정을 반복한다.

$W := W - α\frac{∂}{∂W}cost(W)$

이렇게 현재값과 접선의 기울기를 빼서 새로운 $W$를 갱신하며 최적의 $W$를 찾아간다. 
여기서 $a$는 learning rate를 의미하며 다음 포스팅때 공부하도록 하자


## 참고 자료
[https://wikidocs.net/21670](https://wikidocs.net/21670)
https://creativecommons.org/licenses/by-nc-sa/2.0/kr/

[http://hleecaster.com/ml-linear-regression-concept/](http://hleecaster.com/ml-linear-regression-concept/)

[https://dailyworker.github.io/Linear_regression/](https://dailyworker.github.io/Linear_regression/)

