---  
title:  "L1_BinaryGap"  
  
categories:  
 - Codility
tags:  
 - Study, Codility
 
---

# L1_BinaryGap
### 목차

-  Step 1. 문제 설명
-  Step 2. 풀이
-  Step 3. 결과

##  Step 1. 문제 설명
---

주어진 N의 정수를 이진법으로 변환하여 양 끝에 1로 둘러 쌓인 0의 sequence 중에서 최대 길이를 반환하는 문제이다.

e.g. 10001 이면 3을 반환

단, 100000 처럼 1로 둘러 쌓이지 않으면 0을 반환해야한다.

문제 : 
[https://app.codility.com/programmers/lessons/1-iterations/binary_gap/](https://app.codility.com/programmers/lessons/1-iterations/binary_gap/)

## Step 2. 풀이
---

```python
def solution(N):   
  a = bin(N)[2:].split('1') #convert decimal to binary, slicing except to '0b' and dividing string as '1'  
  if a.count('') == len(a): # if all index values of list is '', then return 0  
	  return 0  
	  
  elif len(a) == 2: # if length of list is 2, then return 0  
	  return 0  
	  
  elif a[0] == '' and a[len(a)-1] != '': # if last index of list is not '',  
	  return max(map(len, a[0:len(a)-1])) # then use function except last index of list  
	  
  else:  
        return max(map(len, a))  

 # map(len, a) : apply len() function to indexes of list  
 # max(map(len, a)) : find max index length of list

```
1. N을 이진수로 바꾼 값에서 0b를 없애기 위해 2번째 인덱스부터 뽑고 1로 split() 실행 
2. 결과가 문자형으로 바뀌기에 ''로 바뀐 1을 카운트해 모두 1인 경우 return 0
3. 10, 11, 00 인 경우 return 0
4. 첫번째 인덱스는 1인데, 마지막 인덱스가 1이 아닌 경우, 그 사이에 존재하는 0 sequence 길이에서 가장 큰값 return
5. 나머지 경우에서 return


## Step 3. 결과
---
![binary gap](https://user-images.githubusercontent.com/59912557/76501071-f6f60380-6484-11ea-9ddd-72939123d400.PNG)

물론 시간은 2분이 걸린건 절대 아니다 ㅋㅋㅋ
여러번 시도 끝에 얻은 결과이다
