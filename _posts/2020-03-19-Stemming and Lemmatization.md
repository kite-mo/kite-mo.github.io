---  
title:  "Stemming(어간 추출) and Lemmatization(표제어 추출)"  
  
categories:  
 - NLP
tags:  
 - Study, NLP, Deep learning
 
---

# Stemming(어간 추출) and Lemmatization(표제어 추출)
### 목차

-  Step 1. 개요
-  Step 2. Lemmatization(표제어 추출)
-  Step 3. Stemming(어간 추출)

해당 게시물은 참고자료를 참고하거나 변형하여 작성하였습니다.


## Step 1. 개요
---
정규화 기법 중 corpus의 단어 개수를 줄일 수 있는 기법인 stemming과 lemmatization의 개념에 대해 공부하고 차이점을 알아보자.

위 두 작업의 의미는 표기상으론 서로 다른 단어이지만, 하나의 단어로 일반화 시킬 수 있다면 하나의 단어로 일반화시켜서 문서 내의 단어 수를 줄이겠다는 것이다. 이러한 방법들은 단어의 빈도수를 기반으로 활용 할 수 있는 BoW(Bag of Words) 표현을 사용하는 NLP 문제에서 주로 사용된다 

NLP에서의 전처리, 더 정확히는 정규화의 지향점은 언제나 갖고 있는 corpus의 복잡성을 줄이는 일이다.


## step 2. Lemmatization(표제어 추출)
---
표제어란 '기본 사전형 단어' 정도의 의미를 갖는다. 표제어 추출은 단어들로부터 표제어를 찾아가는 과정이다. 예를 들면 are, am, is는 서로 다른 표현법이지만 그 뿌리는 be 라는 단어로 볼 수 있다. 그럴 때 이 단어들의 표제어를 be라고 한다.

표제어 추출을 하는 가장 섬세한 방법은 형태학적 파싱을 먼저 하는 것이다. 즉, 형태로소부터 단어들을 만들어 가는 것이다.

형태소는 두 가지 종류가 존재하다.

- 어간(stem) : 단어의 의미를 갖고 있는 핵심 부분
- 접사(affix) : 단어에 추가적인 의미를 주는 부분

형태학적 파싱은 이 두 가지 구성 요소를 분리하는 작업을 말한다. NLTK에서는 표제어 추출을 위한 도구인 WordNetLemmaizer를 지원한다. 

```python
from nltk.stem import WordNetLemmatizer
n=WordNetLemmatizer() 
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 
'lives', 'fly', 'dies', 'watched', 'has', 'starting'] 
print([n.lemmatize(w) for w in words])
```
```python
['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 
'fly', 'dy', 'watched', 'ha', 'starting']
```
표제어 추출은 단어의 형태가 적절히 보존되는 특징이 있다. 그럼에도 위의 결과에서 'dy', 'ha'와 같이 의미를 알 수 없는 적절하지 못한 단어를 출력하고 있다. 즉 **lemmatizer은 본래 단어의 품사 정보를 알아야지만 정확한 결과를 알 수 있다.**

```python
n.lemmatize('dies', 'v')
'die'
```
표제어 추출은 문맥을 고려하며, 수행했을 때 의 결과는 해당 단어의 품사 정보를 보존(POS 태그를 보존)한다.

## Step 3. Step 3. Stemming(어간 추출)
---

어간(stem)을 추출하느 작업을 stemming이라고 한다. 어간 추출은 형태학적 분석의 단순화 버전이라고 할 수 있고, 정해진 규칙만 보고 단어의 어미를 자르는 어림짐작의 작업이라 할 수 있다. 

즉, 이 작업은 섬세한 작업이 아니기 때문에 어가 추출 후에 나오는 결과 단어는 사전에 존재하지 않는 단어일 수도 있다.

어간 추출기인 포터 알고리즘을 사용해보자.

```python
from nltk.stem import PorterStemmer  
from nltk.tokenize import word_tokenize  
s = PorterStemmer()  
text="This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."  
words=word_tokenize(text)  
print([s.stem(w) for w in words])
```
```python
['thi', 'wa', 'not', 'the', 'map', 'we', 'found', 'in', 'billi', 'bone', "'s", 'chest', ',', 'but', 'an', 'accur', 'copi',
 ',', 'complet', 'in', 'all', 'thing', '--', 'name', 'and', 'height', 'and', 'sound', '--'
  'with', 'the', 'singl', 'except', 'of', 'the', 'red', 'cross', 'and', 'the', 'written', 'note', '.']
```
결과를 살펴보면 사전에 없는 단어들도 포함되어 있다. 그 이유는 단순 규칙에 기반하여 이루어지기 때문이다. Poter 알고리즘의 상세 규칙은 마틴 포터의 홈페이지에서 확인이 가능하다.

이렇듯 stemming은 단어의 형태가 완전하지 않음으로 단어의 정확한 정보가 필요하다면 Lemmatization, 단순 단어의 개수를 카운팅하여 결과를 얻고자 하는 목표면 stemming을 사용 할 수 있다. 

**Stemming**  
am → am  
the going → the go  
having → hav

**Lemmatization**  
am → be  
the going → the going  
having → have


## 참고 자료
___

[https://wikidocs.net/21698](https://wikidocs.net/21698)
https://creativecommons.org/licenses/by-nc-sa/2.0/kr/

