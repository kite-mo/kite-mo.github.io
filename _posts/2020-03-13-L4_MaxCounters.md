---  
title:  "L4_MaxCounters"  
  
categories:  
 - Codility
tags:  
 - Study, Codility
 
---

# L4_MaxCounters
### 목차

-  Step 1. 문제 설명
-  Step 2. 풀이
-  Step 3. 결과

## Step 1. 문제 설명
---
양수로 이루어져있는 list A와 N 개의 0 값을 가지는 list가 있다.

list A의 원소 값이 N보다 작다면 그 해당 값을 N의 인덱스 위치라고 여기고 +1을 한다. 
만약 list A의 원소값이 N보다 크다면 N의 list의 원소 값에서 최대값으로 모두 갱신한다.

그 과정을 모두 거친 list N을 리턴하는 문제다.

문제 : 
[https://app.codility.com/programmers/lessons/4-counting_elements/max_counters/](https://app.codility.com/programmers/lessons/4-counting_elements/max_counters/)

## step 2. 풀이
---
```python

def solution(N, A):
 list_N = [0]*N
 
 for i in A:
	 if 1 <= i <= N:
		 list_N[i-1] = list_N[i-1] + 1
	 elif i > N:
		 max_value = max(list_N)
		 list_N = [max_value]*N
		 
 return list_N
```
1. N 길이 만큼의 0으로 구성된 list 생성

2. A의 원소로 for 문을 도는데 N 보다 작으면 해당 위치에 +1을 해주고 N보다 크면 최대값으로 모든 원소값을 갱신한다.


## Step 3. 결과
---

![MaxCounters](https://user-images.githubusercontent.com/59912557/76537027-810f8d80-64c0-11ea-9256-fd43c0dd2c15.PNG)

정확도는 만점이지만, performance가 너무 부족하다.

N의 길이가 매우 길 경우 매번 최대값으로 해당 리스트 길이만큼 갱신해 줄 때, 시간 소비가 매우 큰 듯 하다.

즈흠...다시 고민해보자
