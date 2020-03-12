---  
title:  "L3_TapeEquilibrium"  
  
categories:  
 - Codility
tags:  
 - Study, Codility
 
---

# L3_TapeEquilibrium
### 목차

-  Step 1. 문제 설명
-  Step 2. 풀이
-  Step 3. 결과

## Step 1. 문제 설명
---
리스트 A[0] = 3 , A[1] = 1,  A[2] = 2,  A[3] = 4,  A[4] = 3 가 존재할 때, 
1~4까지의 숫자 기준을 이용하여 순차적으로 자른 왼쪽 리스트와 오른쪽 리스트의 차이의 절대값이 가장 적은 값을 리턴하는 것이다

P = 1, difference = |3 − 10| = 7       
P = 2, difference = |4 − 9| = 5    
P = 3, difference = |6 − 7| = 1  
P = 4, difference = |10 − 3| = 7  

문제 : 
[https://app.codility.com/programmers/lessons/3-time_complexity/tape_equilibrium/](https://app.codility.com/programmers/lessons/3-time_complexity/tape_equilibrium/)

## Step 2. 풀이

```python
def solution(A):  
  
 left_list = [0]*len(A)  
 right_list = [0]*len(A)  
 last_list = [0]*(len(A)-1)  
  
     for i in range(1, len(A)):  
	     left_list[0] = A[0]  
	     left_list[i] = left_list[i - 1] + A[i]
       
     for j in range(-2, -len(A) - 1, -1):  
	     right_list[-1] = A[-1]  
	     right_list[j] = right_list[j + 1] + A[j]
       
     for k in range(0, len(A) - 1):  
         last_list[k] = abs(left_list[k] - right_list[k + 1])  
  
     return min(last_list)
```
1. 어떠한 숫자를 기준으로 자르고 생성된 list틀을 만들기 위해 리스트 길이만큼 0으로 구성된 right list, left list, 그리고 마지막으로 계산할 last list 또한 만듦.

2. A 리스트의 값을 왼쪽에서부터 순차적으로 더한 값을 left list에 갱신

3. A 리스트의 값을 오른쪽에서부터 순차적으로 더한 값을 right list에 갱신

4. k를 기준으로 A 리스트를 잘랐을때 왼쪽 리스트의 총합과 오른쪽 리스트의 총합의 차이의 절댓값을 앞서 구했던 right, left list를 이용하여 계산한 뒤 last list에 갱신

5. 최소값 반환


## Step 3. 결과
----

![TapeEquilibrium](https://user-images.githubusercontent.com/59912557/76530460-1a39a680-64b7-11ea-9d19-e44a7c49c7ff.PNG)

ㅇ...어...어렵다...

