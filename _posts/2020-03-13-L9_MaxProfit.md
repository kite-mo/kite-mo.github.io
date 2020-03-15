---  
title:  "L9_MaxProfit"  
  
categories:  
 - Codility
tags:  
 - Study, Codility
 
---

# L9_MaxProfit
### 목차

-  Step 1. 문제 설명
-  Step 2. 풀이
-  Step 3. 결과

## Step 1. 문제 설명
---
주식을 처음 구입한 날과 주식을 파는 날의 최대값을 구하는 문제이다. list A는 해당 주식의 가치가 원소로 이루어져있다. 물론 파는날은 사는날보다 이후거나 같아야 한다. 그리고 0 이하일 경우 0을 리턴한다

A의 범위는 0 ~ 200,000 

문제 : 
[https://app.codility.com/programmers/lessons/9-maximum_slice_problem/max_profit/](https://app.codility.com/programmers/lessons/9-maximum_slice_problem/max_profit/)

## step 2. 풀이
---

### 첫번째 풀이

```python
def solution(A):  
    profit = []  
    if len(A) == 0:  
        return 0  
  
    for i in range(len(A)):  
        for j in range(len(A)):  
            if j >= i:  
                profit.append(A[j] - A[i])  
            else:  
                continue  
  
    if max(profit) <= 0:  
        return 0  
    else:  
        return max(B)

```
1. 수익을 담을 빈 리스트를 생성한다.

2. 이중 반복문을 사용하여 파는날이 사는날보다 이후인 경우에 이익을 profit list에 담는다.

3. 최종적으로 profit list의 max 값을 반환하는데 0보다 작으면 0을 리턴하고, 아니면 max 값을 반환한다.

정확도는 100점이였지만, performance에서 0점을 맞아서 총 55점을 맞았다. 아마 이중 반복문에서 list의 길이가 길어질 경우 소비되는 시간이 많은 듯 하다.

### 두번째 풀이

```python
def solution(A):
	 # Initialization
	 max_profit = 0
	 min_element = 200000
 
	 for element in A:
		 min_element = min(min_element, element)
		 max_profit = max(max_profit, element-min_element)
		 
	 return max_profit
```
아예 다른 방향으로 접근했다.

1. 어차피 element의 최대값은 200000이므로 최소값을 찾아 비교하기 위해, 초기값을 위와 같이 설정했다.

2. A의 원소값으로 순차적으로 반복문을 사용하여 최소값을 가지는 원소를 구한다

3. 원소값과 최솟값의 차이를 반복문으로 갱신되는 max_profit과 비교하여 최종적으로 가장 큰 max_profit을 리턴한다.

## Step 3. 결과
---

![MaxProfit](https://user-images.githubusercontent.com/59912557/76589670-30367e00-652e-11ea-8635-38a2fa056728.PNG)

알고리즘을 간단하고 효율적으로 생각하는 것이 참 쉽지 않은 것 같다.
