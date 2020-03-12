---  
title:  "L4_FrogRiverOne"  
  
categories:  
 - Codility
tags:  
 - Study, Codility
 
---

# L4_FrogRiverOne
### 목차

-  Step 1. 문제 설명
-  Step 2. 풀이
-  Step 3. 결과

## Step 1. 문제 설명
---
강 건너편으로 잎사귀를 밟아 건너고 싶은 개구리가 있다.
이 개구리는 X 지점까지 가고싶은데, X 지점까지 가기 위해서는 1부터 x까지의 지점을 모두 거쳐야한다. 1부터 X 까지의 지점은 잎사귀가 존재하는 지점이며, 이를 모두 거쳐서 X에 도달했다면 해당 인덱스를 리턴하면 된다.
그 지점까지 가지 못하면 -1 를 리턴한다.

문제 : 
[https://app.codility.com/programmers/lessons/4-counting_elements/frog_river_one/](https://app.codility.com/programmers/lessons/4-counting_elements/frog_river_one/)

## step 2. 풀이
---

```python
def solution(X,A):  
    leaf_set = set()  
    
    for idx, val in enumerate(A):  
        leaf_set.add(val)  
        if len(leaf_set) == X:  
            return idx  
    return -1
```
1. 잎사귀 집합 생성

2. A의 값들을 순차적으로 집합에 갱신

3. 집합의 성질을 이용하여 중복되는 값들은 하나만 남을 것이고 결국 해당 집합의 길이가 X가 됐을 때, A의 idx를 리턴


## Step 3. 결과
---

![FrogRiverOne](https://user-images.githubusercontent.com/59912557/76534058-5a4f5800-64bc-11ea-8fd8-f6f646570c58.PNG)

enumerate() 가 서서히 익숙해진다. 
