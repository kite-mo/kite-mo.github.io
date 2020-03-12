---  
title:  "L2_CyclicRotation"  
  
categories:  
 - Codility
tags:  
 - Study, Codility
 
---

# L2_CyclicRotation
### 목차

-  Step 1. 문제 설명
-  Step 2. 풀이
-  Step 3. 결과

## Step 1. 문제 설명
---
배열 A가 존재하는데 하나의 인덱스 값을 다음 인덱스로 하나씩 넘겨서 최종적으로 K만큼 움직인 배열을 반환하는 문제

A  = [1,2,3] 이고, K = 2 라면, 최종 A = [2, 3, 1]

문제 : 
[https://app.codility.com/programmers/lessons/2-arrays/cyclic_rotation/](https://app.codility.com/programmers/lessons/2-arrays/cyclic_rotation/)


## Step 2. 풀이
---

```python
def solution(A, K):  
    # if list A is empty, return A  
  if len(A) == 0:  
        return A  
    # if K is bigger then len(A), use mods of K/len(A)  
  elif K > len(A):  
        K1 = K % len(A)  
	    return A[-K1:] + A[:-K1]  
  else:  
        return A[-K:] + A[:-K]
```

1. A 리스트가 아무것도 없는 경우 

2. K가 A 리스트 길이보다 더 큰 경우는 최소 한 바퀴를 돌았다는 의미이므로 A의 길이만큼 K를 나눠 나머지를 구함
 
3. 리스트의 성질을 이용하여 [-K, -K + 1, -K + 2 ... ,-1] + [ -리스트 길이, -리스트 길이 + 1, -리스트 길이 + 2 .. , -K -1] 을 합쳐 줌으로써 한칸씩 움직이는 것을 구현 

4. 나머지 경우도 마찬가지로 구현

## Step. 3 결과
---

![CyclicRotation](https://user-images.githubusercontent.com/59912557/76520456-4698f700-64a6-11ea-8c4a-ae929d049ce1.PNG)

물론 시간은 1분이 걸린건 절대 아니다 ㅋㅋㅋ
여러번 시도 끝에 얻은 결과이다
