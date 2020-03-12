---  
title:  "L2_OddOccurrencesInArray"  
  
categories:  
 - Codility
tags:  
 - Study, Codility
 
---

# L2_OddOccurrencesInArray
### 목차

-  Step 1. 문제 설명
-  Step 2. 풀이
-  Step 3. 결과

## Step 1. 문제 설명
---
A [0] = 9 
A [1] = 3 
A [2] = 9 
A [3] = 3 
A [4] = 9 
A [5] = 7 
A [6] = 9

이와 같이 홀수로만 이루어진 list가 존재한다. 리스트안의 값은 항상 짝을 이루는 같은 수의 홀수가 존재한다. 여기서 짝을 이루지 못하는 홀수를 반환하는 문제이다.

문제 : 
[https://app.codility.com/programmers/lessons/2-arrays/odd_occurrences_in_array/](https://app.codility.com/programmers/lessons/2-arrays/odd_occurrences_in_array/)

## Step 2. 풀이
---

### 첫번째 풀이

```python
def solution(A):  
    if len(A) == 1:  
         return A[0] 
          
    A = sorted(A)
      
    for i in range(0 , len (A) , 2):  
         if i+1 == len(A):  
             return A[i]  
         if A[i] != A[i+1]:  
             return A[i]
```

1. A 리스트 길이가 1인 경우 첫번째 값 리턴
2. A 리스트를 오름차순으로 정렬
3. 짝꿍인 홀수끼리 붙어있는 형태가 아닌 홀수를 찾기
-- 가운데에 홀수 혼자 껴있는 경우
-- 마지막에 혼자 있는 경우

### 두번째 풀이

```python

def solution(A):  
    for idx, num in enumerate(A):  
        if A.count(num) % 2 != 0:  
            return num
            
``` 
너무 복잡하게 생각한 것 같아서 좀 더 심플한 방법으로 풀어봄

리스트를 dictionary 형태로 연산을 해 줄 수 있게 해주는 enumerate() 를 이용하여 단순히 리스트 해당 값의 개수가 짝수가 아닌 것을 리턴함

## Step.2 결과 
---


![OddOccruenceInArray](https://user-images.githubusercontent.com/59912557/76523438-ca091700-64ab-11ea-9ce6-d9900ab038fb.PNG)

위의 두 풀이 모두 100점이지만, 두번째 풀이가 훨씬 직관적으로 다가온다.
