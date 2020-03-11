---  
title:  "퍼셉트론 공부"  
  
categories:  
 - Deep learning  
tags:  
 - Study, Deep learning
 
---
# 퍼셉트론(Perceptron)
### 목차

-  Step 1. 퍼셉트론 정의
-  Step 2. 단층 퍼셉트론
-  Step 3. 퍼셉트론 한계
-   Step 4. 다층 퍼셉트론

해당 게시물은 참고자료를 참고하거나 변형하여 작성하였습니다.

## Step 1. 퍼셉트론 정의

퍼셉트론은 신경망(딥러닝)의 근본이 되는 알고리즘으로 다수의 신호를 입력으로 받아 한의 신호를 출력한다. 
퍼셉트론은 이 신호를 입력으로 받아 임계치를 기준으로 1 or 0(흐른다 or 안 흐른다)라는 정보를 전달한다.

![](https://t1.daumcdn.net/cfile/tistory/99BDCE4D5B98A1022C)

위 그림에서 
- $x_1, x_2$를 입력 신호, $y$를 출력 신호 그리고 $w_1, w_2$를 가중치(weight)라고 의미한다.
- 동그라미를 뉴런 or 노드라고 부른다.
- 입력 신호가 노드로 전달될 때, 입력 신호와 가중치가 곱해진다.
- 전달 된 신호의 총합은 임계값($\theta$)를 넘어야만 1을 출력한다.

수식으로 표현하면 아래와 같다

$$if  \sum_i^{n} W_{i}x_{i}\ ≥ \theta → y=1$$

$$if  \sum_i^{n} W_{i}x_{i}\ < \theta → y=0$$

이러한 함수를 step function이라고 부르며 아래 그래프 모양과 같다.

![](https://wikidocs.net/images/page/24987/step_function.PNG)

위 함수를 활성화  함수(Activation function)이라 지칭하며, 매우 여러가지 종류가 존재한다. 이것은 나중 포스팅에서 공부해보자.

퍼셉트론은 복수의 입력 신호 각각에 고유한 가중치를 부여하며, 가중치는 각 신호가 결과에 주는 영향력을 조절하는 요소이며, 
가중치가 클수록 결과에 영향을 많이 끼침을 뜻한다.

위 식에서 $\theta$를 $-b$로 치환을 하면은 아래와 같이 변형이 된다.

$$if  \sum_i^{n} W_{i}x_{i}\  + b≥ 0→ y=1$$

$$if  \sum_i^{n} W_{i}x_{i}\ + b< 0 → y=0$$

여기서 $b$를 **편향(bias)** 라고 한다. 실제로 $b$또한 딥러닝이 최적의 값을 찾아야 할 변수 중 하나이다.

## Step 2. 단층 퍼셉트론

위에서 배운 퍼셉트론을 단층 퍼셉트론이라고 한다. 말 그대로 층이 하나 존재하는 구조인데 퍼셉트론은 단층 퍼셉트론과 
다층 퍼셉트론으로 이루어진다.

단층 퍼셉트론은 입력 신호를 보내는 단계와 받아서 출력하는 두 단계로만 이루어진다. 이때 각 단계를 층(layer)라고 부르며, 
이 두 개의 층을 입력층(input layer) 그리고 출력층(output layer)라고 한다.

![](https://wikidocs.net/images/page/24958/perceptron3_final.PNG)

그럼 단층 퍼셉트론이 어떤 일을 하며, 한계는 무엇인지 알아보자.

대표적으로 단층 퍼셉트론을 이용하면 AND, OR, NAND 게이트는 쉽게 구현이 가능하다. 
 - AND gate : 두 개의 입력값이 모두 1인 경우에만 출력값이 1
 - OR gate : 두 개의 입력값이 모두 0인 경우에만 출력값이 0, 나머지 경우는 1
 - NAND gate : 두 개의 입력값이 모두 1인 경우에만 출력값이 1, 나머지 경우는 0

AND 게이트 경우를 python 코드로 구현해보자. 
```python
def AND_gate(x1, x2):
	w1 = 0.5
	w2 = 0.5
	b = -0.7
	result = x1*w1 + x2*w2 + b
	
	if result > 0:
		return 1
	else:
		return 0
```
위 함수에 각각의 입력값을 넣어보면 오직 두 개의 입력값이 1인 경우에만 출력이 된다.

```python
AND_gate(0,0), AND_gate(1,0), AND_gate(0,1), AND_gate(1,1) 
```
```python
(0,0,0,1)
```

나머지 gate도 이와 같은 방법으로 구현이 가능하다. 

그러나 구현이 불가능한 gate가 존재하는데, 그것은 **XOR gate** 이다. 이 경우는 입력값 두 개가 서로 다른 값을 갖고 있을때만 출력값이 1이 된다. 
위의 알고리즘으로 풀어도 XOR은 구현이 불가능하다. 그 이유는 단층 퍼셉트론은 직선 하나로 두 영역을 나눌 수 있는 문제에 대해서만 구현이 가능하기 때문이다. 

시각화하여 이해를 해보자.

그림에서 출력값을 0인 경우 하얀색, 1인 경우 검은색 원으로 표현했다. 

- AND gate 
![](https://wikidocs.net/images/page/24958/andgraphgate.PNG)

- OR, NAND gate
![](https://wikidocs.net/images/page/24958/oragateandnandgate.PNG)

위 경우는 직선 하나로 검은색과 하얀색 점들의 구분이 가능하다. 

그렇담 XOR gate를 경우는 어떨까.

![](https://wikidocs.net/images/page/24958/xorgraphandxorgate.PNG)

해당 경우는 직선을 이리저리 옮겨봐도 절대로 각각의 점들이 구분이 되지 않는다. 

이를 단층 퍼셉트론은 **선형 영역** 에 대해서만 분리가 가능하다고 한다.

다시 말해 XOR 게이트는 직선이 아닌 곡선, 비선형 영역으로 분리하면 구현이 가능하다.
아래와 같이 말이다.

![](https://wikidocs.net/images/page/24958/xorgate_nonlinearity.PNG)

위 경우가 **다층 퍼셉트론** 으로  발전된 된 계기이다.

## Step 4.  다층 퍼셉트론

말 그대로 다층 퍼셉트론은 단층 퍼셉트론에서 층을 추가한 구조이다. 단층 퍼셉트론은 입력층과 출력층만 존재하지만, 다층 퍼셉트론은 중간에 **은닉층(hidden layer)** 을 추가한 구조이다. 

![](https://wikidocs.net/images/page/24958/perceptron_4image.jpg)

XOR gate를 구현하기 위해선 입력값을 각각 NAND와 OR gate에 보낸 다음, 해당 결과값을 AND gate에 보내면 XOR gate를 구현이 가능하다.

다층 퍼셉트론은 XOR 문제 뿐 아니라 더욱 복잡한 문제를 해결하기 위해 은닉층을 더욱 더 추가할 수 있고 아래와 같이 은닉층이 두 개 이상인 신경망을 **심층 신경망(Deep Neural Network) DNN** 이라고 칭한다.

![](https://wikidocs.net/images/page/24958/%EC%9E%85%EC%9D%80%EC%B8%B5.PNG)

앞선 코딩으로 구현한 AND gate 경우에는 가중치를 수동으로 입력했다. 이제는 기계가 스스로 가중치를 찾아내도록 자동화 해야하는데, 이것이 머신 러닝에서 말하는 **학습(training)** 단계에 해당된다. 

앞선 포스팅에서 공부했던 **cost function(비용 함수)와 옵티마이저(optimizer)**를 사용한다. 

그리고 학습시키는 대상이 심층 신경망일 경우에는 **딥 러닝(Deep learning)** 이라고 한다.

## 참고자료

[https://wikidocs.net/24958](https://wikidocs.net/24958) 
[https://creativecommons.org/licenses/by-nc-sa/2.0/kr/](https://creativecommons.org/licenses/by-nc-sa/2.0/kr/)
[https://excelsior-cjh.tistory.com/169](https://excelsior-cjh.tistory.com/169)





