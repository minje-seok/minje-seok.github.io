---
layout: article
title: Chapter 2. Multi-arm Bandits (2)
aside:
  toc: true
sidebar:
  nav: layouts
---

> 강화학습의 바이블이라고 불리는 Richard S. Sutton의 Reinforcement Learning: An Introduction을 공부하고 필자가 이해한 내용과 추가 정보들을 예제와 함께 간략하고 쉽게(?) 설명하는 방식으로 정리해봅니다. 용어 같은 경우, 원문 그대로 사용하겠지만 혹시 모를 번역 오류 및 잘못된 설명에 대한 지적과 보충 환영합니다. 

# 2. Multi-arm Bandits (2)

## 2.5 Optimistic Initial Values

우리는 그동안 initial action-value estimates $Q_1(a)$ 값에 의존해왔고 이는 곧 biased을 의미한다. sample average 방법의 경우에는 모든 action이 한번 이상 선택되면 bias이 사라지지만, constant $\alpha$를 사용하는 경우 시간이 지남에 따라 감소하더라도 bias은 사라지지않는다. initial action-value의 bias은 예상되는 reward 수준에 대한 사전 지식을 제공하기도 하지만, 이 또한 사용자가 설정하는 파라미터로 취급될 수도 있다는 단점이 존재한다. 

<br/>

### 2.5.1 Using Initial Action Value to Exploration 

bias를 지닌 initial action-value는 exploration은 장려하는 방식으로 사용될 수 있다. 10-armed testbed에서 initial action-value를 0으로 설정하는 대신 +5로 설정했다고 가정한다. 이 문제의 $q(a)$는 본래 $\mu=0, \sigma=1$인 starnard distribution었으므로, initial estimate는 optimistic하다고 여겨지고 action-value를 explore하도록 권장한다. 어떤 action을 선택해도 reward는 initial action value보다 적기 때문에 agent는 더 나은 action을 선택하려고 한다. 결과적으로 estimate가 수렴하기 전에 모든 action이 여러번 시도되며 greedy action이 매번 선택되더라도 agent는 상당한 exploration을 수행하게 된다. 

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/224050454-b7e7ac79-3d78-479b-aca5-b37fe80582ca.png" width="70%" height="70%"></center>

위 그래프는 모든 $a$에 대해 $Q_1(a) = +5$를 사용하는 greedy 방법과 $Q_1(a) = 0$인 $\epsilon$-greedy를 비교한 10-armed bandit testbed의 성능을 보여준다. 초기 optimistic 방법이 더 많이 explore 하기 때문에 성능이 좋지 않지만 시간이 지남에 따라 explore이 어 $\epsilon$-greedy보다도 성능이 더 좋아진다. 

한눈에 보기에는 매우 좋은 방법처럼 보일 수 있으나 이는 stationary 문제에만 적용이 가능하다. 즉, action-value가 변경되는 일반적인 강화학습 문제가 적용되는 non-stationary env에서는 적합하지 않다. 그럼에도 불구하고 이를 기반으로 한 방법들은 매우 단순하면서도 종종 sample average와 같은 방법들에서 적합한 경우도 존재한다.  

<br/>

## 2.6 Upper-Confidence-Bound Action Selection

estimated action value의 부정확함으로 인해 exploration이 더 필요하다. $\epsilon$-greedy를 수행할수도 있지만 이는 불확실한 action에 대한 preference 없이 무차별적으로 시도된다. 우리는 estimate가 maximal과 얼마나 가까운지와 estimate의 불확실성을 모두 고려하여 실제로 optimal일 가능성에 따라 non-greedy action 중에서 선택하는 것이 바람직하다. 


이를 효과적으로 수행하기 위해서는 action을 다음과 같이 선택한다. 여기서 $\ln t$는 $t$($e \approx 2.71828$)의 natural logarithm를 나타내며 $c > 0$는 exploration 정도를 제어한다. 만약 $N_t(a) =0$이면, $a$는 maximizing action이라고 간주된다. 


$$ A_t = \arg\max_a \left[ {Q_t(a) + c\sqrt{\cfrac{\ln t}{N_t(a)}}} \, \right ] \tag{8}$$

<br/>

upper confidence bound(UCB)의 아이디어는 불확실성 또는 variance의 척도를 의미하는 square-root 항을 사용하여 $a$의 estimated action-value를 표현하자는 것이다. 따라서 해당 action value가 최대가 되는 값은 신뢰 수준을 결정하는 $c$와 함께 action $a$의 가능한 true value에 대한 upper bound이다. 

특정 action $a$가 선택됨에 따라, $a$가 선택된 횟수인 $N_t(a)$는 증가하고 이는 불확실성 항의 분모에 나타나므로 항은 감소한다. 반면에 다른 $a$가 선택될 때마다 $t$는 증가하고 이는 분자에 나타나므로 불확실성 추정치가 증가한다. 결과적으로 $a$가 선택될 때마다 불확실성은 감소할 것이다. natural logrithm의 사용은 증가폭이 시간이 지남에 따라 작아지지만 제한이 없음을 의미한다. 결국 모든 action이 선택되지만 시간이 지남에 따라 estimate가 낮거나 이미 더 많이 선택된 action의 경우 대기 시간이 길어지고 선택 빈도가 낮아지게 된다.

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/224050461-b89bf748-346b-46c5-960e-abacadcbabcd.png" width="70%" height="70%"></center>

위 그래프는 10-armed testbed에서 UCB를 사용한 결과이다. UCB는 종종 잘 수행되지만 bandit 문제 이외에서는 강화학습에서 일반적인 설정이 아닌 stationary env 성질로 인해 다른 문제들로의 확장은 어렵다. 또한 나중에 배우게 될 large state space, 특히 function approximation에서의 적용도 어렵다. 이러한 고급 설정에서는 UCB 아이디어를 활용하는 실용적인 방법은 존재하지 않는다고 한다. 


<br/>

## 2.7 Gradient Bandits

여태까지는 action-value를 추정하고 해당 estimate를 사용하여 action을 선택하는 방법을 고려했으나, 각 action $a$에 대한 numerical preference $H_t(a)$ 학습을 고려해본다. preference가 클수록 해당 action이 더 자주 수행되지만 preference는 reward 측면에서 해석되지 않고 오직 한 action이 다른 action보다 상대적으로 preference 되는 것 만을 고려한다. 

모든 preference에 1000을 추가하면 *(1000이 어느정도의 수치인지는 모르겠다)* softmax distribution(i.e. Gibbs or Boltzmann distribution)에 따라 결정되는 action probability에 영향을 미치지 않는다고 하는데 수식은 $(9)$와 같다. 여기서 $\pi_t(a)$는 time $t$에 action $a$를 수행할 확률을 의미한다. 초기 모든 preferences는 같다(e.g. $H_1(a) = 0, \forall a$). 


$$ \Pr \{ {A_t = a} \} = \cfrac{e^{H_t(a)}}{\sum^n_{b=1}e^{H_t(b)}} = \pi_t(a) \tag{9}$$

<br/>

### 2.7.1 How to Update Preference $H_t(a)$

이에 대한 stochastic gradient ascent에 기반한 알고리즘이 존재한다. 매 step에서 action  $A_t$를 선택한 후, reward $R_t$를 받은 뒤에 preference는 $(10)$과 같이 update된다. 이 때 $\alpha > 0$는 step-size, $\bar R_t\in \mathbb R$은 incrementally implementation하게 계산 가능한 time $t$까지의 rewards의 평균이다. 

$$ \begin{align*} H_{t+1}(A_t) &= H_t(A_t) + \alpha(R_t-\bar R_t)(1-\pi_t(A_t)), \, and\\
H_{t+1}(a) &= H_t(a) - \alpha(R_t-\bar R_t)\pi_t(a), \quad\quad\quad \forall a \ne A_t \tag{10}\end{align*} $$

<br/>

$\bar R_t$는 reward가 비교되는 baseline 역할을 수행하고, 만약 reward $R_t$가 더 높은 경우에 미래에 $A_t$를 취하는 probability $\pi_t$가 증가하고, 낮은 경우에는 감소한다. 선택되지 않은 action들은 반대쪽으로 이동한다. 이전과는 다르게 action-value estimate가 아닌 preference라는 수치를 통해 action을 취할 확률이 결과로 나오게 되어 이를 update 한다. 

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/224055619-8935787a-c513-4d28-856a-0cf8ce94a358.PNG" width="70%" height="70%"></center>

위 그래프는 $\mu=4$인 normal distribution에서의 10-armed testbed 결과를 보여준다. reward의 전반적인 상승이 있었지만 reward baseline의 사용으로 인해 gradient-bandit 알고리즘에는 큰 영향을 주지 못했다. 그러나 만약 baseline을 생략하면($\bar R_t = 0$), 성능이 크게 저하된다. 


<br/>

### 2.7.2 Stochastic Approximation in Graient Ascent 

해당 알고리즘을 gradient ascent에 대한 stochastic approximation의 측면에서 이해해보자. gradient ascent에서 각 preference $H_t(a)$는 성능에 대한 증가의 효과에 비례하여 증가한다. 이 때, 성능의 평가는 expected reward $\mathbb{E}[R_t] = \sum_b\pi_t(b)q(b)$로 이루어진다. 

$$ H_{t+1}(a) = H_t(a)+\alpha \cfrac{\delta \mathbb E[R_t]}{\delta H_t(a)} \tag{11} $$

<br/>

물론 우리는 정확한 true $q(b)$를 알 수 없기 때문에, 정확한 gradient ascent 구현은 불가능하지만 알고리즘의 expected update와 gradient of expected reward는 거의 유사함을 보인다.

따라 gradient bandit 알고리즘은 stochastic gradient ascent의 instance이며 강력한 수렴 성질을 지닌다. reward baseline는 선택된 action이 아닌 다른 action에 의존하므로 update시 딱히 필요하지 않아 이는 어떤 값이든 상관없다. 따라 알고리즘의 expected update에 영향을 미치지는 않지만 update의 분산과 수렴 속도에 영향을 미친다. reward 평균으로 동작하는 것은 suboptimal일지도 모르나 잘 작동한다고 한다. 

<br/>

## 2.8 Associative Search (Contextual Bandits)

지금까지는 서로 다른 action을 서로 다른 situation와 연결할 필요 없는 non-associative task만 고려했다. 즉, 항상 같은 situation에서 매번 action을 선택했으나 일반적인 강화학습에는 하나 이상의 situation에서 policy의 학습을 원한다. 즉 전체 문제에 대한 단계 설정을 위해서는 각 situation에서 가장 optimal action으로의 매핑 즉, associative task로 전환해야 한다. 

associative search는 optimal action을 search하는 형태에서의 trial-and-error 학습과 situation에 optimal하게 action을 association하는 형태를 모두 포함한다. associative search는 n-armed bandit 문제와 전체 강화학습 문제 사이의 중간이다. 그들은 situation에 따른 policy 학습(depend on situation)을 한다는 점에서 전체 강화학습과 비슷하지만, 각 action이 immediate reward에만 영향을 미친다는 점(stationary)에서 n-armed bandit 문제와 비슷하다. 이제부터는 action이 long term reward에 영향을 미치는 non-stationary env에서, 매 situation에 의존한 policy를 학습하는 강화학습 문제를 제시하고 결과를 고려하게 된다. 

<br/>

## 2.9 Summary

이번 챕터에서는 exploration과 exploitation의 균형을 맞추는 몇가지 방법을 제안했다. $\epsilon$-greedy 방법은 시간의 매우 작은 부분을 무작위로 선택하는 반면, UCB 방법은 deterministic하게 선택하지만 더 적은 수의 sample을 받은 action을 선호하여 선택한다. Gradient-Bandit 알고리즘은 action-value가 아닌  action preference를 추정하고 softmax distribution을 사용하여 확률적 방식으로 선호하는 action을 파악한다. 별개로, 특정 값으로의 estimate value로 초기화한 상태로의 greedy 방법 또한 살펴보았다. 

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/224055717-b414ca86-e7da-4cbc-86f9-cb1e110c72c3.PNG" width="60%" height="60%"></center>

최종적으로 위 그래프는 10-armed testbed에서 매개변수 값에 따른 성능 비교를 보여준다. 전체적으로 역 U자 모양을 그리며 매개변수의 중간 값에서 잘 작동하며 UCB가 가장 성능이 좋았다. $n$-armed bandit 문제를 푸는 이외에도 많은 알고리즘들이 있지만 고려하는 강화학습 문제 정의에는 포함되지 못했고, 살펴본 방법들은 아직 exploration과 exploitation의 균형 문제에 대한 완전히 만족스러운 해결책은 아니었다.  

<br/>

