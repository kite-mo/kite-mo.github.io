---  
title:  "Tokenization"  
  
categories:  
 - NLP
tags:  
 - Study, NLP, Deep learning
 
---
# Tokenization
### 목차

- Step 1. Tokenization
- Step 2. Word Tokenization
- Step 3. Tokenization 고려사항
- Step 4. Sentence Tokenization
- Step 5. Binary Classifier
- Step 6. 한국어에서 tokenization의 어려움
- Step 7. Part-of-speech tagging

해당 게시물은 참고자료를 참고하거나 변형하여 작성하였습니다.

## Step 1. Tokenization
---
주어진 corpus에서 token이라 불리는 단위로 나누는 작업을 tokenization이라고 한다. 토큰의 단위는 상황에 따라 다르지만, 보통 의미있는 단위로 토큰을 정의한다.

## Step 2. Word Tokenization
---

토큰의 기준을 word로 하는 경우, word tokenization이라한다. 다만, 여기서  word 외에도 단어구, 의미를 갖는 문자열로도 간주되기도 한다.

### input : Time is an illusion. Lunchtime double so!

해당 입력으로부터 구두점(punctuation)과 같은 문자를 제외시키는 간단한 토큰화 작업을 한다. 구두점은 온점(.), 콤마(,), 물음표(?) 등과 같은 기호를 말한다.

### output : "Time", "is", "an", "illustion", "Lunchtime", "double", "so"

해당 예제는 매우 간단한 작업이다. 그러나 보통 tokenization에선 단순히 구두점이나 특수문자를 전부 제거하는 cleaning 작업을 수행하는 것만으로 해결되지 않는다. 모두 제거할 경우 토큰이 의미를 잃어버리는 경우도 발생하기 때문이다.

### 1) tokenization 중 생기는 선택의 순간
tokenization 과정 중에서 예외의 경우가 존재하여 해당 기준을 생각해봐야 하는 경우가 발생한다.  

예를들어 영어 언어에서 apostrophe( ' )가 들어있는 단어는 어떻게 토큰으로 분류할까라는 문제를 보여주면

### input : **Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop.**

Don't와 Jone's는 어떻게 tokenization 할 수 있을까?

- Don't, Dont, Don t, Do n't
- Jone's, Jone s, Jone, Jones

토큰화 도구를 직접 설계도 가능하겠지만, 기존에 공개된 도구를 사용을 하게 되면 아래와 같은 결과가 나온다.

### NLTK - word_tokenize
```python
from nltk.tokenize import word_tokenize 
print(word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
```
```python
['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.',
 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
```
### NLTK - wordPunctTokenizer
```python
from nltk.tokenize import WordPunctTokenizer
print(WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
```
```python
['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 
'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
```
해당 tokenization 도구는 구두점을 별도로 분류하는 특징을 갖고 있다.

### 케라스 토큰화 도구 - text_to_word_sequence

```python
from tensorflow.keras.preprocessing.text import text_to_word_sequence 
print(text_to_word_sequence("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
```
```python
["don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'mr', "jone's", 'orphanage',
 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
```

기본적으로 모든 알파벳을 소문자로 바꾸면서 ( ' )를 제외한 나머지의 구두점을 제거한다.

## Step 3. Tokenization 고려사항
---

### 1) 구두점이나 특수 문자를 단순 제외해서는 안 된다.

코퍼스 cleaning 작업을 하다보면, 구두점조차도 하나의 토큰으로 분류하기도 한다. 대표적인 예시로보면 온점( . )과 같은 경우는 문장의 경계를 알 수 있으므로 온점을 제외하지 않을 수 있다.

### 2) 줄임말과 단어 내에 띄어쓰기가 있는 경우

tokenization에서 ' 로 압축된 단어를 다시 펼치는 역할도 한다. 예를 들면 what're은 what are의 줄임말이며, 위의 예에서 re를 접어(clitic)이라고 한다.

New York, rock 'n' roll 과 같은 단어는 하나의 단어 사이에 띄어쓰기가 있는 경우에도 하나의 토큰으로 봐야하는 경우도 있을 수 있으니 tokenization 작업은 저러한 단어를 하나로 인식할 수 있는 기능도 가져야한다.

 ### 3) Standard tokenization example

표준으로 쓰이는 **Penn Treebank Tokenization** 의 규칙에 대해서 알아보고 토큰화 결과를 확인해보자

- 하이푼( - )으로 구성된 단어는 하나로 유지
- clitic이 함께하는 단어는 분리

### - TreebankWordTokenizer
```python
from nltk.tokenize import TreebankWordTokenizer 

tokenizer=TreebankWordTokenizer() 
text="Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own." 
print(tokenizer.tokenize(text))
```
```python
['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does',
"n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']
```
결과를 보면 규칙에 따라 home-based를 하나의 토큰으로 취급하고, doesn't인 경우 does와 n't가 분리됨을 알 수 있다.

##  Step 4. Sentence Tokenization

해당 작업은 코퍼스 내에 문장 단위로 구분하는 작업으로 문장 분류(sentence segmentation)라고도 부른다. 그러나 sentence tokenization은 주의할 점은 무조건 온점( . )으로 자를 경우 예외 사항이 발생하기 때문이다.

### e.g.  Since I'm actively looking for Ph.D. students, I get the same question a dozen times every year.
 
 위 문장을 온점( . )으로 구분할 시에 Ph.D.가 하나의 단어이지만, 문장의 끝이 나오기 전에 위 단어는 두개로 쪼개진다.

그래서 코퍼스가 어떤 국적의 언어인지, 특수문자들이 어떻게 사용되고 있는지에 따라 규칙을 주의할 필요가 있다.

NLTK에서는 영어 문장의 토큰화를 수행하는 **sent_tokenize**를 지원하고 있다.
### sent_tokenize
```python
from nltk.tokenize import sent_tokenize 
text="I am actively looking for Ph.D. students. and you are a Ph.D student." 
print(sent_tokenize(text))
```
```python
['I am actively looking for Ph.D. students.', 
'and you are a Ph.D student.']
```

해당 도구는 온점을 구분자로 하여 문장을 구분하지 않았기 떄문에, Ph.D를 문장 내의 단어로 인식하여 성공적으로 sentence tokenization 됨을 볼 수 있다.

## Step 5. Binary Classifier
---
sentence tokenization 과정에서 예외 사항을 발생시키는 온점( . ) 처리를 위해 입력에 따라 두개의 클래스로 분류하는 binary classifier를 사용하기도 한다.

- 온점( . ) 이 단어의 일부분인 경우
- 온점( . ) 이 문장의 구분자인 경우

이러한 sentence tokenization을 수행하는 오픈 소스로는 NLTK, OpenNLP, standford CoreNLP, splitta, LingPipe 등이 있다. 

##  Step 6. 한국어에서 tokenization의 어려움
---

한국어는 영어와 달리 띄어쓰기만으로는 토큰화를 하기엔 역부족이다. 띄어쓰기 단위가 되는 단위를 '어절' 이라고 하는데 한국어에서는 어절 tokenization이 지양되고 있다. 그 이유는 어절 tokenization과 단어 tokenization이 같지 않기 때문이다.

근본적인 이유는 한국어는 조사, 어미 등이 붙어 만드는 언어인 교착어이기 때문이다. 그런 이유로 한국어는 조사 등의 무언가인 교착어를 분리해줘야한다.

그래서 한국어 tokenization에선 **형태소(morpheme)** 란 개념이 중요하다. 형태소란 뜻을 가진 가장 작은 말의 단위를 말하며 두 가지 형태소가 존재한다.

- 자립 형태소 : 체언(명사, 대명사, 수사), 수식언(관형사, 부사), 감탄사 등이 있다.
- 의존 형태소 : 접사, 어미, 조사, 어간이 있다.

이를 통해 유추할 수 있는 것은 한국어는 어절 tokenization이 아닌 형태소 tokenization이 수행되어야 한다.

## Step 7. Part-of-speech tagging
---

단어의 표기가 같을 지라도 품사에 따라 단어의 의미가 달라지기도 한다. 
 - 	e.g. 못(명사), 못(부사)

결국 단어의 의미를 제대로 파악하기 위핸 해당 단어가 어떤 품사로 쓰였는지 보는 것이 매우 중요하다. 

그렇기에 단어 tokenization 과정에서 각 단어가 어떤 품사로 쓰였는지를 구분해 놓는 작업인 **품사 태깅(part-of-speech-tagging)** 이라 한다.

### 1) NLTK를 이용한 영어 tokenization

```python
from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import pos_tag

tokenizer = TreebankWordTokenizer()  
text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."  
x = tokenizer.tokenize(text)  
print(pos_tag(x))
```
```python
[('Starting', 'VBG'), ('a', 'DT'), ('home-based', 'JJ'), ('restaurant', 'NN'), ('may', 'MD'), ('be', 'VB'), ('an', 'DT'), ('ideal.', 'NN'), ('it', 'PRP'), ('does', 'VBZ'),
 ("n't", 'RB'), ('have', 'VB'), ('a', 'DT'), ('food', 'NN'), ('chain', 'NN'), ('or', 'CC'), ('restaurant', 'NN'), ('of', 'IN'), ('their', 'PRP$'), ('own', 'JJ'), ('.', '.')]
```

### 2) KoNLPy를 이용한 한국어 tokenization

한국어 NLP를 위해서는 KoNLPy를 통해 사용할 수 있는 형태소 분석기로는 Okt, Mecab, Komoran, Hannanum, Kkmo가 있다.

```python
from nltk.tokenize import TreebankWordTokenizer
from konlpy.tag import Kkma

  
kkma=Kkma()  
print(kkma.pos("자연어 처리를 제대로 공부를 해볼까?"))  
print(kkma.morphs("자연어 처리를 제대로 공부를 해볼까?"))  
print(kkma.nouns("자연어 처리를 제대로 공부를 해볼까?"))
```
```python
[('자연어', 'NNG'), ('처리', 'NNG'), ('를', 'JKO'), ('제대로', 'MAG'), ('공부', 'NNG'), ('를', 'JKO'), ('해보', 'VV'), ('ㄹ까', 'EFQ'), ('?', 'SF')]
['자연어', '처리', '를', '제대로', '공부', '를', '해보', 'ㄹ까', '?']
['자연어', '처리', '공부']
```

## 참고 자료
___

[https://wikidocs.net/21698](https://wikidocs.net/21698)
https://creativecommons.org/licenses/by-nc-sa/2.0/kr/

