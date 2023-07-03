---
title: ""
permalink: /labwork/bookreview_machine_learning
layout: single
---
## Book Review:『Machine Learning』, 오일석  

### Contents
1. 소개  
    1. 기계 학습이란
    2. 특징 공간에 대한 이해
    3. 데이터에 대한 이해
    4. 간단한 기계 학습의 예
    5. 모델 선택
    6. 규제
    7. 기계 학습 유형
    8. 기계 학습의 과거와 현재, 미래

***

### Chapter1.1 기계 학습이란
- 학습의 정의  

![학습이란](/_pages/images/chap01_02/image1.png)  
> ###### 키워드는 '경험'

- 현대적 정의의 기계 학습  

![기계 학습의 정의_1](/_pages/images/chap01_02/image2.png)
![기계 학습의 정의_2](/_pages/images/chap01_02/image3.png)  
> ###### 키워드는 경험(= 데이터)과 성능 개선  

- 지식 기반 접근 방식에서 데이터 중심 접근 방식으로의 변화  

![지식 기반의 한계](/_pages/images/chap01_02/image5.png)  
> ###### <center>단추를 '가운데 구멍이 몇 개 있는 물체'라고 규정하면 많은 문제가 발생한다. <br>즉 단추가 아닌 물건이 단추가 되고 단추가 단추가 아닌 물건이 된다. <br> *사람은 변화가 심한 장면을 아주 쉽게 인식하지만, 왜 그렇게 인식하는지 서술하지 못함*</center>  

- 목표치의 데이터 유형에 따른 기계 학습 - 회귀와 분류(Regression and Classification)
> ##### 회귀는 목표치가 실수형(또는 연속형), 분류는 범주형(또는 이산형)  
![회귀의 예시](/_pages/images/chap01_02/image7.png)  
> ##### 위 예제는 회귀 문제  

기계 학습이란  
- 가장 정확하게 예측할 수 있는 최적의 매개변수 값을 찾는 작업
- 처음에는 임의의 값에서 시작한 후 점점 성능을 개선하여 최적의 값에 도달하는 것  

기계 학습의 궁극적인 목표  
- 훈련 집합에 없는 새로운 샘플(테스트 집합)에 대한 오류를 최소화
- 일반화(Generalization) 능력: 테스트 집합에 대한 높은 성능  

![사람의 학습과 기계 학습의 비교](/_pages/images/chap01_02/image11.png)<br><br><br>

### Chapter1.2 특징 공간에 대한 이해  

![1차원 특징 공간과 2차원 특징 공간](/_pages/images/chap01_02/image12.png)
> ##### 특징 벡터 표기: x = (x<sub>1</sub>, x<sub>2</sub>)<sup>T</sup><br><br>


![다차원 특징 공간](/_pages/images/chap01_02/image13.png)  
> 다양한 데이터 셋<br><br>

- d차원 데이터
> ##### 특징 벡터 표기: x = (x<sub>1</sub>, x<sub>2</sub>, … , x<sub>d</sub>)<sup>T</sup>  
- d차원 데이터를 위한 학습 모델  
> 직선 모델: 매개변수 수 = d + 1<br>
y = w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + ... + w<sub>d</sub>x<sub>d</sub> + b  

> d차 곡선 모델: 매개변수 수 = d<sup>2</sup> + d + 1  

- 초기 기계 학습 알고리즘은 XOR 문제를 해결하지 못함
![선형분리 불가능](/_pages/images/chap01_02/image16.png)
> 선형 분리 불가능한 원래 특징 공간(왼쪽)과 변환된 새로운 특징 공간(오른쪽)  
- 차원의 저주
> 차원이 높아짐에 따라 데이터 연산 속도가 기하 급수적으로 증가함

<br><br>

## 1.3 데이터에 대한 이해  
- 기계 학습
> 기계 학습이 푸는 문제는 훨씬 복잡함  
단순한 수학 공식으로는 표현 불가능함  
자동으로 모델을 찾아내는 과정이 필수  
- 실제 기계 학습 문제에서는 데이터 생성 과정을 알 수 없음
> 단지 주어진 훈련 집합 X, Y로 예측 모델 또는 생성 모델을 근사 추정할 수 있을 뿐  
- 데이터베이스의 왜소한 크기
> MNIST 데이터는 이론적으로 2<sup>784</sup>가지의 샘플 수를 가지지만 실제로는 약 6만 개 정도이다  

![왜소한 데이터베이스](/_pages/images/chap01_02/image24.png)  
> 방대한 공간에서 실제 데이터가 발생하는 곳은 매우 작지만 ![확률0](/_pages/images/chap01_02/image25.png)와 같은 샘플의 발생 확률은 거의 0이다  

- 4차원 이상의 초공간은 한 번에 가시화 불가능함
> 2개 씩 조합하여 여러 개의 그래프를 그림  

![가시화방법](/_pages/images/chap01_02/image30.png)<br><br><br>

### Chapter1.4 간단한 기계 학습의 예  
- 선형 회귀 문제
> 두 개의 매개변수 Θ = (𝑤, 𝑏)<sup>T</sup>: y = wx + b  

![선형 회귀](/_pages/images/chap01_02/image7.png)  
- 목적 함수(objective function) 또는 비용 함수(cost function)  
![MSE](/_pages/images/chap01_02/image31.png)
> 위 식을 평균 제곱 오차(MSE)라고 부름  
- 기계 학습이 할 일을 공식화하면   
![공식화](/_pages/images/chap01_02/image38.png)  

- 알고리즘 형식으로 쓰면  
![알고리즘](/_pages/images/chap01_02/image39.png)  

- 조금 더 현실적인 상황  
> 실제 세계는 선형이 아니며 잡음이 섞임, 즉 비선형 모델이 필요
![비선형모델](/_pages/images/chap01_02/image40.png)<br><br><br>

### Chapter1.5 모델 선택  
- 과소 적합과 과대 적합(과잉 적합)  
![과소적합 과대적합](/_pages/images/chap01_02/image41.png)  
> 과소 적합: 훈련 집합과 테스트 집합 모두 낮은 성능  
과대 적합: 훈련 집합에선 거의 완벽한 예측, 테스트 집합에선 낮은 성능  

- 훈련 집합을 여러 번 수집하자  
![2차와 12차 비교](/_pages/images/chap01_02/image43.png)  
> 2차는 편향이 크지만(매번 큰 오차) / 분산이 낮음(비슷한 모델)  
12차는 편향이 작지만(매번 작은 오차) / 분산이 높음(크게 다른 모델)  

**<center>분산과 편향은 트레이드 오프 관계</center>**  

- 기계 학습의 목표  
![기계학습의 목표](/_pages/images/chap01_02/image44.png)  
> 낮은 편향과 낮은 분산을 가진 모델 제작이 목표, 즉 왼쪽 아래의 상황

**<center>하지만 분산과 편향은 트레이드 오프 관계이기에 편향의 희생을 최소로 하며 분산을 최대로 낮추는 전략이 필요</center>**  


- 검증 집합을 이용한 모델 선택  
![검증작업](/_pages/images/chap01_02/image45.png)