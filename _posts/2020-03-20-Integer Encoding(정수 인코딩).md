---  
title:  "정수 인코딩(Integer Encoding)"  
  
categories:  
 - NLP
tags:  
 - Study, NLP, Deep learning
 
---

# 정수 인코딩(Integer Encoding)
### 목차

-  Step 1. 개요
-  Step 2. 정수 인코딩(Integer Encoding)
-  Step 3. 케라스(Keras)의 텍스트 전처리


해당 게시물은 참고자료를 참고하거나 변형하여 작성하였습니다.


## Step 1. 개요
---
컴퓨터는 텍스트보다는 숫자를 더 잘 처리 할 수 있다(연산 속도 측면). 이를 위해 자연어 처리에서 텍스트를 숫자로 바꾸는 여러 기법들이 있다. 그러한 기법들은 본격적으로 적용시키기 위해 첫 단계로 각 단어에 고유한 정수에 mapping 시키는 전처리 작업이 필요할 때가 있다. 

예를 들어 텍스트 단어가 500개 있다면, 각각 단어들에게 1~500번까지의 단어와 맵핑되는 고유한 정수, 즉 **인덱스**를 부여한다. 보통은 빈도수 높은 단어들을 이용하기 위해 단어에 대한 빈도수를 기준으로 정렬한 뒤 부여한다.

## step 2. 정수 인코딩(Integer Encoding)
---
해당 작업이 왜 필요한 지에 대해서는 one-hot encoding, word embedding 챕터에서 더  공부해도록 하자.

```python
from nltk.tokenize import sent_tokenize  
from nltk.tokenize import word_tokenize  
from nltk.corpus import stopwords  
  
text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! " \  
       "The Secret He Kept is huge secret. Huge secret. His barber kept his word. " \  
       "a barber kept his word. His barber kept his secret. " \  
       "But keeping and keeping such a huge secret to himself was driving the barber crazy. " \  
       "the barber went up a huge mountain."
```
문장 토큰화 진행

```python
text = sent_tokenize(text)
print(text)

['A barber is a person.', 'a barber is good person.', 'a barber is huge person.', 
'he Knew A Secret!', 'The Secret He Kept is huge secret.', 'Huge secret.', 'His barber kept his word.', 
'a barber kept his word.', 'His barber kept his secret.', 
'But keeping and keeping such a huge secret to himself was driving the barber crazy.', 
'the barber went up a huge mountain.']

```
corpus cleaning 및 word_tokenize 진행
```python 
sentences = []  
stop_words = set(stopwords.words('english'))  
  
for i in text:  
    sentence = word_tokenize(i) # 단어 토큰화를 수행합니다.  
    result = []  
  
    for word in sentence:  
        word = word.lower() # 모든 단어를 소문자로  
		  if word not in stop_words: # 불용어를 제거  
			  if len(word) > 2: # 단어 길이가 2이하인 경우 단어를 제거  
			  result.append(word)  
			  
    sentences.append(result)  
print(sentences)
```
```python
[['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'],
 ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'],
 ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'],
 ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
```

### (1) Counter 사용하기
단어 집합(vocabulary)를 만들기 위해 sentences에서 문장의 경계인 [, ] 를 제거하고 하나의 리스트로 변형
```python
words = sum(sentences, [])
```
**Counter()** 를 이용하여 중복을 제거하고 단어의 빈도수를 기록(dictionary 형태로 변형)
```python
from collections import Counter
vocab = Counter(words)
print(vocab)
```
```python
Counter({'barber': 8, 'secret': 6, 'huge': 5, 'kept': 4, 'person': 3, 'word': 2, 'keeping': 2,
 'good': 1, 'knew': 1, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1})
```
**most_common()** 함수를 이용하여 상위 빈도수를 가진 사용자가 설정한 단어 수 만큼을 리턴한다. 즉, 등장 빈도수가 높은 단어들을 원하는 개수만큼 얻을 수 있다.
```python
vocab_size = 5
vocab = vocab.most_common(vocab_size)
print(vocab)
```
```python
[('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3)]
```
이제 높은 빈도수를 가진 단어일수록 낮은 정수 인덱스를 부여한다.
```python
word_to_index = {}
i = 0
for (word, freq) in vocab:
	i = i + 1
	word_to_index[word] = i
print(word_to_index)
```
```python
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
```
### (2) NLTK의 FreqDist 사용하기

FreqDist() 함수는 Counter() 함수와 비슷한 방법으로 쓰인다.
```python
from nltk import FreqDist  
import numpy as np  
  
#np.hstack으로 문장의 경계인 [, ]을 제거하여 입력으로 사용 가능  
vocab = FreqDist(np.hstack(sentences))  
vocab_size = 5  
vocab = vocab.most_common(vocab_size)  
print(vocab)
```
```python
[('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3)]
```
이번엔 enumerate() 함수를 이용하여 빈도수가 높은 단어일 수록 낮은 정수 인덱스를 부여한다.

```python
word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)}
print(word_to_index)
```
```python
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
```

위와 같이 인덱스를 부여할 때는 enumerate()를 사용하는 것이 편리하다.

### (3) enumerate 이해하기

enumerate()는 순서가 있는 자료형을 입력받아 인덱스를 순차적으로 함께 리턴한다는 특징을 가짐.

```python
test=['a', 'b', 'c', 'd', 'e']  
for index, value in enumerate(test): # 입력의 순서대로 0부터 인덱스를 부여함.  
  print("value : {}, index: {}".format(value, index))
```
```python
value : a, index: 0
value : b, index: 1
value : c, index: 2
value : d, index: 3
value : e, index: 4
```

## Step 3. Step 3. 케라스(Keras)의 텍스트 전처리
---

케라스에서는 정수 인코딩을 위한 전처리 도구인 토크나이저를 이용할 수 있다.

```python
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
  
# input인 corpus로부터 단어 빈도수가 높은 순으로 낮은 정수 인덱스를 부여  
tokenizer.fit_on_texts(sentences)
# 각 단어의 인덱스 확인  
print(tokenizer.word_index)
# 각 단어의 카운트 확인  
print(tokenizer.word_counts)

```
```python
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7,
 'good': 8, 'knew': 9, 'driving': 10, 'crazy': 11, 'went': 12, 'mountain': 13}

OrderedDict([('barber', 8), ('person', 3), ('good', 1), ('huge', 5), ('knew', 1), ('secret', 6), ('kept', 4),
 ('word', 2), ('keeping', 2), ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)])
```
앞서 빈도수가 가장 높은 단어 n개만을 사용하기 위해 most_common()을 사용했는데, 이번엔 케라스에서도 빈도수가 높은 상위 몇개 단어만 사용하겠다고 지정할 수 있다.

```python
vocab_size = 5  
tokenizer = Tokenizer(num_words= vocab_size + 1))
tokenizer.fit_on_texts(sentences)  
```

+1 을 해주는 이유는 num_words는 숫자를 0부터 시작하기에 5까지 카운트하기 위해서 1을 더하는 것이다.

text_to_sequences() 는 입력으로 들어온 corpus에 대해서 각 단어를 이미 정해진 인덱스로 반환하는데, 앞서 설정한 상위 5개 단어의 인덱스로만 구성되는지 확인해보자.

```python
print(tokenizer.texts_to_sequences(sentences))
```
```python
[[1, 5], [1, 5], [1, 3, 5], [2], [2, 4, 3, 2],
 [3, 2], [1, 4], [1, 4], [1, 4, 2], [3, 2, 1], [1, 3]]
```
총 1부터 5까지의 인덱스, 즉 상위 빈도를 가진 단어들로만 구성되있음을 확인이 가능하다.

## 참고자료
---

[https://wikidocs.net/21698](https://wikidocs.net/21698)
https://creativecommons.org/licenses/by-nc-sa/2.0/kr/

