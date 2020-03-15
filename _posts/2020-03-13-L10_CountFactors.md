---  
title:  "L10_CountFactors"  
  
categories:  
 - Codility
tags:  
 - Study, Codility
 
---

# L10_CountFactors
### 목차

-  Step 1. 문제 설명
-  Step 2. 풀이
-  Step 3. 결과

## Step 1. 문제 설명
---
말 그대로 N이 주어졌을때, N의 약수의 개수를 구하는 문제이다.

문제 : 
[https://app.codility.com/programmers/lessons/10-prime_and_composite_numbers/count_factors/](https://app.codility.com/programmers/lessons/10-prime_and_composite_numbers/count_factors/)


## step 2. 풀이
---

### 첫번째 풀이

```python
def solution(N):   
  factor_list = []  
  
    for i in range(1, N + 1):  
        if N % i == 0:  
            factor_list.append(i)  
        else:  
            pass  
 return len(factor_list)
```
1. 약수를 담을 리스트를 생성

2. 1~N까지 N을 나눠서 0이 나머지가 0인 경우를 factor_list에 넣기

3. factor_list의 길이 반환

정확도는 100점이지만, N이 무지하게 큰 경우 오래 걸리기에 performance를 0점 맞아서 57 점 맞았다.


### 두번째 풀이

```python
def solution(N):  
  
    num_factors = 0  
	factors = 1  
  
    while factors * factors < N:  
	    if N % factors == 0:  
	       num_factors = num_factors + 2  
        factors = factors + 1
  
	if factors*factors == N:  
	    num_factors = num_factors + 1 #num_factors += 1  
  
    return num_factors
```

하나의 약수 x 다른 약수 = N 의 성질을 이용했다.

1. 초기값을 설정

2. 약수 x 약수가 N보다 작은 경우, 나머지가 0이면 약수의 개수 2개를 추가

3. 약수 x 약수가 N인 경우, 약수의 개수를 1개 추가

4. 최종 개수 리턴

## Step 3. 결과
---

![CountFactors](https://user-images.githubusercontent.com/59912557/76590733-74774d80-6531-11ea-8595-ea0d8f52deb9.PNG)


