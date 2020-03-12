---  
title:  "L3_PermMissingElem"  
  
categories:  
 - Codility
tags:  
 - Study, Codility
 
---

# L3_PermMissingElem
### 목차

-  Step 1. 문제 설명
-  Step 2. 풀이
-  Step 3. 결과

## Step 1. 문제 설명
---
매우 간단하다. 연속적인 양수로 구성되어 있는 리스트에서 빠진 값을 리턴하면 된다.

e.g. [1,2,4,5] 일 경우 , 3을 리턴

문제 : 
[https://app.codility.com/programmers/lessons/3-time_complexity/perm_missing_elem/](https://app.codility.com/programmers/lessons/3-time_complexity/perm_missing_elem/)


## step 2. 풀이

```python
def solution(A):  
  
        set_A = set(A)  
        set_B = set(range(1, len(A) + 2))  
        num = list(set_B - set_A)  
  
        return num[0]
```

1. 리스트를 집합으로 변환
2. 1부터 A의 길이 + 1 만큼 존재하는 집합 생성 즉, 순차적으로 숫자가 존재하는 집합 생성
3. 둘 간의 차집합을 결과를 리스트로 변환 후 해당 값 리턴

## Step 3. 결과


![PermMissingElem](https://user-images.githubusercontent.com/59912557/76526983-bf518080-64b1-11ea-8299-ec755d4880c4.PNG)






