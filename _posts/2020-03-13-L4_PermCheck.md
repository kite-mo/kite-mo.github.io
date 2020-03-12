---  
title:  "L4_PermCheck"  
  
categories:  
 - Codility
tags:  
 - Study, Codility
 
---

# L4_PermCheck
### 목차

-  Step 1. 문제 설명
-  Step 2. 풀이
-  Step 3. 결과

## Step 1. 문제 설명
---
리스트가 주어졌을때, 해당 리스트가 완전한 순열의 형태이면 1을 리턴, 그렇지 않으면 0을 리턴하는 문제다

문제 : 
[https://app.codility.com/programmers/lessons/4-counting_elements/perm_check/](https://app.codility.com/programmers/lessons/4-counting_elements/perm_check/)

## step 2. 풀이
---

```python

def solution(A):
  
    len_A = len(A)  
    set_A = set(A)  
  
    if len_A != len(set_A):  
        return 0  
  
    list_com = [0] * len_A  
    for i in range(0, len_A):  
        list_com[i] = i+1  
  
    set_com = set(list_com)  
  
    if len(set_A - set_com) == 0  
        return 1  
	else:  
        return 0

```

1. 리스트 A와 집합 A의 길이가 같지 않다면 0을 리턴한다

2. 1부터 순차적으로 A 리스트의 길이만큼 이루어져있는 집합을 생성
한다.

3.  A의 집합과 방금 생성한 집합간의 차집합 길이가 0 이라면 순열이므로 1을 리턴, 그렇지 않으면 0을 리턴

## Step 3. 결과
---

![PermCheck](https://user-images.githubusercontent.com/59912557/76541061-56c0ce80-64c6-11ea-83d8-b4b2b79264db.PNG)


