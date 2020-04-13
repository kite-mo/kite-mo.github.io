---  
title:  "Cleaning and Nomalization"  
  
categories:  
 - NLP
tags:  
 - Study, NLP, Deep learning
 
---

# Cleaning(정제) and Nomalization(정규화)
### 목차

-  Step 1. 개요
-  Step 2. 규칙에 기반한 표기가 다른 단어 통합
-  Step 3. 대, 소문자 통합
-  Step 4. 불필요한 단어의 제거
-  Step 5. 정규 표현식(Regular Expression)

해당 게시물은 참고자료를 참고하거나 변형하여 작성하였습니다.

## Step 1. 개요
---
tokenization 작업 전, 후에는 corpus를 용도에 맞게 정제 및 정규화하는 일이 필수적이다.

- Cleaning(정제) : corpus로부터 노이즈 데이터 제거
- Normalization(정규화) : 표현 방법이 다른 단어들을 통합시켜 같은 단어로 만듦

## step 2. 규칙에 기반한 표기가 다른 단어 통합 
---

표기가 다른 단어들을 하나의 단어로 정규화하는 방법들이 존재한다. 예를들면 USA, US는 같은 의미를 가지므로 하나의 단어로 정규화할 수 있다. 이러한 정규화 과정을 거치면 US를 찾아도 USA를 함께 찾을 수 있을 것이다. 

이러한 표기가 다른 단어들을 통합하는 방법은 어간 추출(stemming)과 표제어 추출(lemmatization)이 존재한다.

## Step 3. 대, 소문자 통합
---

영어에서는 대, 소문자를 통합하는 것은 단어의 개수를 줄일 수 있는 또 다른 정규화 방법이다. 영어는 대부분은 문장의 맨 앞 등과 같은 특정 상황에서만 쓰이고, 대부분은 글은 소문자로 작성되기 때문이다. 그래서 대문자를 소문자로 변환하는 작업이 이루어지게 된다.

```python
corpus = "I am a boy"
corpus.lower()
print(corpus)
"i am a boy"
```
그러나 모든 토큰을 소문자로 무작정 통합해서는 안된다. 대명사같은 경우에는 대문자로 유지되어야 하기 때문이다. 

이러한 작업은 다 많은 변수를 사용하여 소문자 변환을 언제 사용할지 결정하는 머신 러닝 시퀸스 모델로 더 정확하게 진행이 가능하다고 한다.

## Step 4. 불필요한 단어의 제거
---

정제 작업에서 제거해야하는 noise data는 자연어가 아니면서 아무 의미도 갖지 않는 글자를 의미하기도 하지만, 분석하고자 하는 목적에 맞지 않는 불필요한 단어들을 의미한다.

불필요한 단어들을 제거하는 방법으로는 stopwords 제거와 등장 빈도가 적은 단어, 길이가 짧은 단어들을 제거하는 방법들이 존재한다. 

### (1) Stopwords(불용어) 제거
갖고있는 corpus에서 유의미한 단어 토큰만 선별하기 위해서는 큰 의미가 없는 단어 토큰을 제거해야한다. 즉, 자주 등장하지만 분석을 하는 것에 있어서는 큰 도움이 되지 않는 단어들을 말한다.

예를들면 I, my, me, he, that, 조사, 접미사, 관계대명사 같은 단어들은 문장에서는 자주 등장하지만 실제 의미 분석을 하는데는 거의 기여하는 바가 없다. 이러한 단어들을 **stopwords** 라고 하며,  여러 불용어 리스트를 제공하는 라이브러리도 존재하고 사용자가 직접 정의도 가능하다.
```python
from nltk.corpus import stopwords   
from nltk.tokenize import word_tokenize   
  
example = "Family is not an important thing. It's everything."  
stop_words = set(stopwords.words('english'))   
  
word_tokens = word_tokenize(example)  
  
result = []  
for w in word_tokens:   
    if w not in stop_words:   
        result.append(w)
```

### (2) 길이가 짧은 단어 제거

영어권에서 길이가 짧은 단어들을 삭제하는 것은 즉, 불용어를 제거하는 것과 어느정도 중복이 된다. 또한 길이가 짧은 단어를 제거하는 2차 이유는 길이를 조건으로 텍스트를 삭제하면 단어가 아닌 구두점들까지도 한꺼번에 제거하기 위함도 있다.

그러나 한국어는 단어가 한자어도 존재하고 한 글자 만으로도 이미 의미를 가진 경우가 많기 때문에 주의리를 기울여야 한다.

##  Step 5. 정규 표현식(Regular Expression)
---

정규 표현식은 컴퓨터 싸이언스의 정규 언어로부터 유래한 것으로 **특정한 규칙을 가진 문자열의 집합을 표현하기 위해 쓰이는 형식언어** 이다. 즉, 어떤 corpus 내에서 특정한 형태나 규칙을 가진 문자열을 찾기 위해 그 형태나 규칙을 나타내는 패턴을 정규 표현식이라고 이해하면 된다.

파이썬에서는 정규 표현식 모듈 re을 지원하므로, 이를 이용하면 특정 규칙이 있는 corpus를 빠르게 정제할 수 있다. 

### (1) re.sub() : 정규 표현식 패턴과 일치하는 문자열을 찾아 다른 문자열로 대체
```python
import re
text = "I was wondering if he could sky diving"
# 소문자를 제외한 나머지는 '' 로 변환
change_text = re.sub('[^a-z]', '', text)
print(change_text)

was wondering if he could sky diving
```

### (2) re.split() : 입력된 정규 표현식을 기준으로 문자열들을 분리하여 리스트로 리턴
```python
import re  

text = "블로그 글쓰기는 참 재밌다."
text2 = """블로그
글쓰기는
참
재밌다."""
text3 = "블로그+글쓰기는+참+재밌다"
  
print(re.split(" ", text))
print(re.split("\n", text2))
print(re.split("\+", text3))

['블로그', '글쓰기는', '참', '재밌다.']
['블로그', '글쓰기는', '참', '재밌다.']
['블로그', '글쓰기는', '참', '재밌다.']
```

### (3) re.findall() : 정규 표현식과 매치되는 모든 문자열들을 리스트로 리턴한다. 단, 매치되는 문자열이 없다면 빈 리스트를 리턴

```python
# 숫자만 찾도록
re.findall("\d+", "저는 1996년에 태어났습니다.")
['1996']
```

추가적인 정규 표현식의 모듈 및 함수는 아래의 링크가 매우 친절하게 설명이 되어있다.
[https://wikidocs.net/21703](https://wikidocs.net/21703)



## 참고 자료
___

[https://wikidocs.net/21698](https://wikidocs.net/21698)
https://creativecommons.org/licenses/by-nc-sa/2.0/kr/

