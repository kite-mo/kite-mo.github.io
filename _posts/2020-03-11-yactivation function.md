﻿---  
title:  "활성화 함수와 옵티마이저 공부"  
  
categories:  
 - Deep learning  
tags:  
 - Study, Deep learning
 
---
# 활성화 함수(Activation function)
### 목차

-  Step 1. 활성화 함수(Activation function)
	 *  활성화 함수 특징
	 * 활성화 함수 종류 소개 


해당 게시물은 참고자료를 참고하거나 변형하여 작성하였습니다.

## Step 1. 활성화 함수(Activation function)

앞선 포스팅에서 여러번 언급되었던 활성화 함수에 대해서 공부해보자. 활성화 함수의 매커니즘은 실제 뇌를 구성하는 신경 세포 뉴런이 신호를 일정 수준을 받으면 서로 화학적으로 연결되는 모습을 모방한 것이다.

![](https://t1.daumcdn.net/cfile/tistory/99FC1C425BB6EB150A)

이렇듯 출력층의 뉴런(노드)에서 출력값을 결정하는 함수를 **활성화 함수** 라고 칭한다.  이전에 공부했던 계단 함수, 시그모이드 함수 등이 활성화 함수에 속한다.

 우선 활성화 함수에 대한 특징을 살펴보자.

### 1) 활성화 함수의 특징 - 비선형 함수(Nonlinear function)

딥러닝 신경망에서는 활성화 함수를 선형 함수가 아닌 비선형 함수를 이용한다. 그 이유는 선형 함수를 사용할 시 층을 깊게 하는 의미가 줄어들기 때문이다.

예를 들어 활성화 함수로 선형 함수($f(x) = Wx$)를 선택하고, 층을 계속 쌓는다고 가정해보자. 

은닉층 두 개를 추가한다고 하면 출력층을 포함해서 $y(x) = f(f(f(x)))$가 된다. 
이를 식으로 다시 표현하면  $W W*W*x$ 이다. 여기서 $W^3$ 을 $K$로 표현하면 결국 $y(x) = Kx$와 같은 선형 표현으로 머문다. 즉 선형 함수로는 은닉층을 여러번 추가하더라도 1회 추가한 것과 차이를 줄 수 없다.

그러므로 은닉층을 쌓는 이점을 원한다면 비선형 함수를 사용해야 한다.

그럼 이제 활성화 함수 종류에 대해서 공부해보자

### 2) 계단 함수(Step function)

![](https://wikidocs.net/images/page/24987/step_function.PNG)
 
 위 활성화 함수는 거의 이용하지 않지만, 앞선 퍼셉트론을 공부할 때 처음 접했던 활성화 함수다.

### 3) 시그모이드 함수(Sigomoid function) 그리고 기울기 손실(Vanishing Gradient )

이전 포스팅에서 인공 신경망에 쓰였던 활성화 함수인 시그모이드 함수이다. 

![](https://wikidocs.net/images/page/60683/simple-neural-network.png)

순전파, 역전파 과정을 반복해가며 가중치를 업데이트하여 최종 오차를 줄이는 결과를 확인했다. 그러나 이런 시그모이드 함수도 단점을 가지고있다. 그것은 미분을 하여 기울기(gradient)를 구하는 과정에서 발생한다. 

왜 그런 현상이 발생하는지 시그모이드 함수의 모양을 살펴보자.

![](https://wikidocs.net/images/page/60683/%EC%8B%9C%EA%B7%B8%EB%AA%A8%EC%9D%B4%EB%93%9C%ED%95%A8%EC%88%981.PNG)

시그모이드 함수에서는 출력값이 -1 또는 1에 가까워질 수록 그래프의 기울기가 완만해지는 모습을 볼 수 있다. 기울기가 완만해지는 구간을 주황색, 그렇지 않은 부분을 초록색으로 표현해보자.

![](https://wikidocs.net/images/page/60683/%EC%8B%9C%EA%B7%B8%EB%AA%A8%EC%9D%B4%EB%93%9C%ED%95%A8%EC%88%982.PNG)

주황색 부분의 기울기를 구하면 0에 아주 가까운 작은 값이 나온다. 그런데 앞서 공부했던 역전파 과정에서 0에 아주 가까운 기울기를 곱하게 된다면, 이전의 기울기가 전달이 되지 않는다. 이러한 현상을 **기울기 소실(vanishing gradient)** 라고 한다. 

이러한 현상은 은닉층의 개수가 점점 더 많아질 경우, 즉 네트워크의 깊이가 깊어질 수록 발생한다. 다시 말하면 매개변수 가중치($W$)가 업데이트 되지 않아 학습이 되지를 않는다.

![](https://wikidocs.net/images/page/60683/%EA%B8%B0%EC%9A%B8%EA%B8%B0_%EC%86%8C%EC%8B%A4.png)

위 그림은 은닉층의 개수가 많은 Deep Neural Network(DNN)이다.역전파를 수행하는 과정에서 출력층에 가까운 기울기는 전달이 원활하지만, 입력층으로 향할 수록 기울기가 제대로 전파되지 않는 현상을 보여준다.

결론적으로 은닉층의 개수가 많은 경우엔 활성화 함수로 시그모이드 함수를 사용되는 것은 지양된다.

그래서 이러한 문제를 해결하기 위한 여러 활성화 함수도 존재한다.

### 4) 하이퍼볼릭탄젠트 함수(Hyperbolic tangent function)

하이퍼볼릭탄젠트 함수는 입력값을 -1~1까지의 출력값으로 변환한다.  

![](https://wikidocs.net/images/page/60683/%ED%95%98%EC%9D%B4%ED%8D%BC%EB%B3%BC%EB%A6%AD%ED%83%84%EC%A0%A0%ED%8A%B8.PNG)

시그모이드 출력값 범위에서 -1 ~0 이 추가된 형태이다. 그래서 그나마 시그모이드 함수보다는 기울기 소실 증상이 적은 편이다.

### 5) 렐루 함수(ReLU)

인공 신경망에서 엄청난 인기를 얻고 있는 함수라고 한다. 수식은 $f(x) = max(0,x)$로 매우 심플하다.

![](https://wikidocs.net/images/page/60683/%EB%A0%90%EB%A3%A8%ED%95%A8%EC%88%98.PNG)

렐루 함수는 입력값이 0보다 작으면 0을 출력하고, 양수를 입력하면 입력값을 그대로 출력값으로 반환한다. 렐루 함수는 입력값이 양수인 경우 특정 양수값에 수렴하지 않고 계속 발산하는 형태이기에 시그모이드보다 훨씬 성능이 좋으며, 어떤 연산이 필요하지 않고 그저 단순 임계값이므로 연산 속도도 빠르다. 

하지만 단점은 입력값이 음수인 경우엔 기울기가 0으로 수렴한다. 그리고 이 문제로 소실된 노드(뉴런)은 다시 살리기는 매우 어렵다. 이 문제를 **dying ReLU** 라고 한다.

### 6) 리키 렐루(Leaky ReLU)

dying ReLU의 문제를 보완하기 위해 ReLU의 변형 함수들이 등장했다. 여기서는 Leaky ReLU를 공부해보자. 해당 활성화 함수는 입력값이 음수인 경우 0이 아닌 0.1, 0.01과 같은 매우 작은 값으로 출력값을 반환을 한다.

여기서 수식은 $f(x) = max(ax, x)$로 아주 간단쓰하다. $a$는 하이퍼파라미터로 Leaky(새는) 정도를 결정하며 일반적으로 0.01 값을 가진다. 여기서 Leaky 정도는 입력값이 음수일 때의 기울기를 비유하고 있다.

![](https://wikidocs.net/images/page/60683/%EB%A6%AC%ED%82%A4%EB%A0%90%EB%A3%A8.PNG)

위와 같이 입력값이 음수라도 기울기가 0이 되지 않으면 dying ReLU 현상이 나타나지 않는다.

### 7) 소프트맥스 함수(Softmax function)

은닉층에서 ReLU, Leaky ReLU 함수를 사용하는 것이 보통이지만 그렇다해서 시그모이드 함수, 소프트맥스 함수를 사용하지 않는 것은 아니다.

보통 분류 문제를 위해 로지스틱 회귀와 소프트맥스 회귀를 출력층에 적용하여 사용한다.

![](https://wikidocs.net/images/page/60683/%EC%86%8C%ED%94%84%ED%8A%B8%EB%A7%A5%EC%8A%A4.PNG)

시그모이드 함수는 이진 분류(Binary Classification) 문제에서 사용된다면, 세 가지 이상의 선택지를 고르는 다중 클래스 분류(Multi Classification) 문제에 주로 사용된다.

다음은 옵티마이저에 대해서 공부해보자!

## 참고자료

[https://wikidocs.net/24987](https://wikidocs.net/24987)
[https://creativecommons.org/licenses/by-nc-sa/2.0/kr/](https://creativecommons.org/licenses/by-nc-sa/2.0/kr/)
