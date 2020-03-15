---  
title:  "L9_MaxDoubleSliceSum"  
  
categories:  
 - Codility
tags:  
 - Study, Codility
 
---

# L9_MaxDoubleSliceSum
### 목차

-  Step 1. 문제 설명
-  Step 2. 풀이
-  Step 3. 결과

## Step 1. 문제 설명
---
범위가 0 ≤ X < Y < Z 인 세 가지 수가 주어지며, 리스트 A를 X,Y,Z를 기준으로 slice를 실행한다. 두 개로 나눠진 list의 사이의 값들의 최대값을 반환하는 문제다.

e.g. A[0] = 3, A[1] = 2, A[2] = 6, A[3] = -1, A[4] = 4, A[5] = 5, A[6] = -1, A[7] = 2 인 경우, X,Y,Z가  (0, 3, 6)일 때, 총합은 2 + 6 + 4 + 5 = 17 이다

문제 : 
[https://app.codility.com/programmers/lessons/9-maximum_slice_problem/max_double_slice_sum/](https://app.codility.com/programmers/lessons/9-maximum_slice_problem/max_double_slice_sum/)

## step 2. 풀이
---
```python
def solution(A):  
    A_len = len(A)  
  
    leftmax = [0] * A_len  
    rightmax = [0] * A_len  
  
    for i in range(1, A_len-1):  
        leftmax[i] = max(0, leftmax[i-1] + A[i])  
  
    for j in range(-2, -A_len, -1):  
        rightmax[j] = max(0, rightmax[j+1] + A[j])  
  
    maximum = 0  
  
  for k in range(1, A_len-1):  
        sub_max = leftmax[k-1] + rightmax[k+1]  
        if maximum < sub_max:  
           maximum = sub_max  
    return maximum
```
앞서 풀었던 TapeEquilibrium 문제와 비슷한 방향으로 접근했다.

1. A list의 길이 만큼에 0으로 구성되어 있는 right, left list 생성한다.

2. right, left list를 오른쪽, 왼쪽부터 순차적으로 더해주는데 최대값을 찾기 때문에, 음수인 경우는 제외해주기 위해 0과 비교하여 max를 취해준다.

3. 순차적으로 leftmax와 rightmax를 더한 값을 sub_max에 저장하고 더 큰 값을 찾아 계속하여 갱신한 뒤, 최종적으로 가장 큰 maximum 값을 리턴한다.

## Step 3. 결과
---
![MaxDoubleSliceSum](https://user-images.githubusercontent.com/59912557/76588117-87861f80-6529-11ea-9eb7-52ef98623df7.PNG)


