---  
title:  "L8_Dominator"  
  
categories:  
 - Codility
tags:  
 - Study, Codility
 
---

# L8_Dominator
### 목차

-  Step 1. 문제 설명
-  Step 2. 풀이
-  Step 3. 결과

갑자기 Lesson을 건너 뛴 이유는 같이 진행하는 스터디에 진도를 맞추기 위해서 미리 먼저 풀고있다.

## Step 1. 문제 설명
---
리스트 A가 주어졌을 때, 구성 원소 값들 중에서 리스트의 길이의 절반 개수보다 해당 원소 값의 개수가 많은 경우를 dominator라 하며, dominator의 인덱스 위치를 리턴하는 문제다.

문제 : 
[https://app.codility.com/programmers/lessons/8-leader/dominator/](https://app.codility.com/programmers/lessons/8-leader/dominator/)

## step 2. 풀이
---
```python
def solution(A):  
  
    length = len(A) #allocate length of array in length variable  
	only_A = list(set(A))  

    h_length = length // 2 # half length of array  
    dom_num = 0  
  
    for i in only_A: 
    
        a_count = A.count(i) # count number of i in list A  
        
	    if a_count > h_length: # find case of the dominator
           dom_num = i 
           return A.index(dom_num) 

    return -1 # if dominator is not exist, return -1
```

1. 우선 for 문에서 반복 횟수를 줄이기 위해, 집합의 성질을 이용하여 list A의 중복 원소를 없애준다.

2. 집합의 원소 값으로 반복문을 돌리는데, 해당 원소 값의 A에서의 개수가 list A 길이의 절반보다 크다면 dom_num에 저장한다

3. dom_num을 가지는 A에서의 인덱스 위치를 반환한다.


## Step 3. 결과
---

![dominator](https://user-images.githubusercontent.com/59912557/76587208-7982cf80-6526-11ea-97bc-7ac3812e8d1c.PNG)

타임 아웃 에러로 performance가 50점을 맞았다 ㅠ
반복 횟수를 줄이려고 중복 값들을 제거했는데도 문제가 있는건 for 문에서 시간 소비가 많은 것 같다.
