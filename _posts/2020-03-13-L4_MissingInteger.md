---  
title:  "L4_MissingInteger"  
  
categories:  
 - Codility
tags:  
 - Study, Codility
 
---

# L4_MissingInteger

-  Step 1. 문제 설명
-  Step 2. 풀이
-  Step 3. 결과

## Step 1. 문제 설명
---
순차적으로 정수로 구성된 list A가 주어졌을 때, 해당 리스트에서 빠져있는 정수값을 return하는 문제이다. 단, 모든 원소가 음수인 경우에는 +1 값을 return 하면 된다.   

문제 : 
[https://app.codility.com/programmers/lessons/4-counting_elements/missing_integer/](https://app.codility.com/programmers/lessons/4-counting_elements/missing_integer/)

## step 2. 풀이
---
```python
def solution(A):  
  
    A.sort()  
    list_A = list(set(A))  
    min = 1  
  
  for val in list_A:  
  
        if val > min:  
            return min  
        if val > 0:  
            min = val + 1  
  
  return min
```
1. 오름차순으로 나열, 집합을 취하여 중복값을 제거한 뒤 다시 리스트로 변환했다.

2.  어차피 음수이면 +1 리턴이므로, min의 초기값을 1 로 설정

3. A 리스트의 원소로 for 문을 돌면서 min 값을 +1 만큼 갱신시켜준다.

4. 만약 min 값보다 해당 list 원소값이 크다면 그 순서에는 연속적으로 이어져야할 정수가 비어있다는 것을 의미하므로, 그 값을 return 한다.


## Step 3. 결과
---

![Missing_Integer](https://user-images.githubusercontent.com/59912557/76538608-e9f80500-64c2-11ea-800b-7e78decb3296.PNG)

최솟값을 1로 시작하는 것이 key point 문제인듯 하다.
