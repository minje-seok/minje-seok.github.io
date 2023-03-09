---
layout: article
title: Chapter 2. Multi-arm Bandits (1)
aside:
  toc: true
sidebar:
  nav: layouts
---

> 강화학습의 바이블이라고 불리는 Richard S. Sutton의 Reinforcement Learning: An Introduction을 공부하고 필자가 이해한 내용과 추가 정보들을 예제와 함께 간략하고 쉽게(?) 설명하는 방식으로 정리해봅니다. 용어 같은 경우, 원문 그대로 사용하겠지만 혹시 모를 번역 오류 및 잘못된 설명에 대한 지적과 보충 환영합니다. 

# 2. Multi-arm Bandits (1)

강화학습은 올바른 action을 지시하는 것 대신 취한 action을 평가하는 교육 정보를 사용한다. 좋은 action에 대한 explicit trial-and-error을 위해서 적극적인 explore이 필요한 이유이다. action을 평가하는 방식인 feedback에는 evaluative feedback과 instructive feedback이 존재한다. 이번 챕터에서는 오직 하나의 situation에서만 action하는 단순화된 stationary 환경에서의 강화학습의 evaluative 측면을 연구하고 이러한 설정은 evaluative feedback이 수행된 설정이며 전체 강화학습 문제의 복잡성을 방지한다. 


**- Evaluative feedback**

전적으로 취해진 action에 의존하여, 해당 action이 얼마나 좋은지를 나타내지만 가능한 최선 또는 최악인지 나타내지 않는 feedback.


**- Instructive feedback**

취해진 action과 관계없이, 궁극적으로 취해야 할 올바른 action을 나타내는 feedback.

<br/>

## 2.1 An n-Armed Bandit Problem

해당 문제에서는 $n$개의 슬롯머신이 있고, 한번의 action 선택으로 잭팟을 터뜨리면 지급되는 reward를 위해 슬롯머신 중 하나의 레버를 선택해 플레이하는 것이다. 당신은 반복되는 action 선택을 통해 최상의 레버에 action을 집중하여 상금을 최대화할 수 있다. 어떤 action을 선택하는 것이 최고의 기대되는 reward을 가질 수 있는지 알아보자. 


이러한 문제는 다음과 같이 일반화가 가능하다. $n$개의 다른 action 중 하나를 선택해야 하는 상황에 반복적으로 노출되며, 하나의 action 선택 후 그에 따라 stationary probability distribution에서 선택한 수치의 reward을 받는다. 사용자는 일정 기간 혹은 일정 횟수의 action 선택 동안 예상되는 총 reward을 최대화 하는 것이 목표이다. 각 action은 기대되거나 평균적인 reward를 가지게 되는데 이는 action의 value라고 할 수 있고 지난 챕터에서 언급한 강화학습 요소인 value function과 연결된다. 

<center><img src="https://user-images.githubusercontent.com/127359789/224034422-6d5ce4e2-d930-4924-a84a-34e578010d41.jpg" width="60%" height="60%"></center>
<center><span style="font-size:80%">http://sanghyukchun.github.io/96/</span> </center>

만약 확실하지는 않아도 각 action의 value estimate를 알고 있다면, 우리는 value가 가장 높은 action을 선택할 것이다. 이러한 행위를 'greedy action'이라고 부른다. 이는 action의 value에 대한 현재 지식을 exploit하고 있다고 말할 수 있다. 대신 action의 value estimate를 향상 시키기 위해 다른 action을 선택한다면 explore하고 있다고 말할 수 있다. 

exploit는 해당 step에서 기대되는 보상을 최대화할 수 있지만 explore는 불확실성 속에서 장기적으로 더 큰 총 보상을 만들어낼 수 있다. 그렇다고 explore을 많이하면 총 보상이 무조건적으로 늘어난다고 단언할 수도 없고 일반적인 문제에서 시간 혹은 선택이 제한되어있다는 것 또한 고려해야 한다. 단일 action 선택으로는 결국 exploration과 exploitation을 동시에 수행하는 것은 불가능해 충돌이 발생하게 된다. 이번 챕터에서는 exploration과 exploitation의 균형을 맞추는 방법과 그 적절성을 보여준다. 

<br/>

## 2.2 Action-Value methods
### 2.2.1 Action-Value Estimation: sample average

먼저 action에 대한 value를 estimate로 나타내본다. action $a$의 true value를 $q(a)$, $t$th time step의 estimated value를 $Q_t(a)$로 표시한다. 여기서 true value는 해당 action이 선택되었을 때 받는 평균 reward를 의미한다. 예를 들어, $t$th time step에서의 action $a$가 t 이전 $N_t(a)$번 선택되어 reward가 $R_1$, $R_2$, $\ldots$ , $R_{N_t(a)}$라면 estimated value는 다음과 같다. 


$$ Q_t(a) = \cfrac{R_1 + R_2 + \cdots + R_{N_t(a)}}{N_t(a)} \tag{1} $$ 

<br/>

만약 $N_t(a) = 0$이면, $Q_t(a)$는 $Q_1(a) = 0$으로 정의한다. 대수의 법칙에 따라 $N_t(a)$ $\to$ $\infty$이므로 $Q_t(a)$는 $q(a)$로 수렴한다. 각 estimate가 해당 action에 대한 reward sample들의 단순 평균이기 때문에 이를 action-value 추정을 위한 'sample average' 방법이라 부른다. 해당 방식 외에도 value를 측정하는 방식은 다양하지만 설명을 위해서 이번 챕터에서는 estimate 계산을 위해 사용한다. 

<br/>

### 2.2.2 Action-Value Selection: greedy

가장 단순하게 action을 선택하는 방법은 가장 높은 estimated action-value $A^{*}_t$를 선택하는 것이다. $Q_t(A^{*}_t) = \max_a Q_t(a)$. 이러한 greedy action 선택은 $(2)$와 같이 쓸 수 있다. 여기서 $\arg\max_a$는 다음 표현식이 최대화 되는 $a$의 값을 의미한다. 여기서 중요한 점은 greedy 방식은 현재 지식을 활용하여 즉각적인 보상을 극대화하며, 실제로 더 나은지 확인하기 위해 열등한 action을 샘플링하는데 시간을 소비하지 않는다. 즉, exploit로만 action을 선택하고 explore를 진행하지 않는다. 


$$ A_t = \arg\max_a Q_t(a) \tag{2} $$

<br/>

### 2.2.3 Action-Value Selection: $\epsilon$-greedy

exploit만 진행하는 greedy 방식에서 explore 하기 위한 간단한 대안으로는 대부분의 시간에 greedy하게 행동하되 가끔 정말 작은 확률인 $\epsilon$으로 estimate는 신경쓰지 않고 동일한 확률로 모든 action중에서 랜덤으로 선택하는 것이다. 이는 $\epsilon$-greedy 방식이라고 불리며 추후 배울 알고리즘들에서 다시 언급될 것이다. 


이러한 방법은 재생 횟수가 증가하는 한도 내에서 모든 action이 무한 횟수로 샘플링되어 모든 $a$에 대해 $N_t(a)$ $\to$ $\infty$를 보장하고 따라 모든 $Q_t(a)$가 $q(a)$로 수렴하게 된다. 이는 물론 optimal action을 선택할 확률이 $1-\epsilon$보다 크게, 즉 거의 확실하게 수렴한다는 것을 의미하지만 이는 점근적인 보장임을 상기해야 한다. 

<br/>

### 2.2.4 Comparision between greedy and $\epsilon$-greedy

무작위로 생성된 2000번의 10-Armed Bandit 문제; 10-armed testbed를 통해 greedy와 $\epsilon$-greedy의 실용적 관점에서의 상대적 효과를 평가해보자. 각 bandit에 대해,  $a = 1 , \ldots , 10$인 action-values $q(a)$는 standard normal(gaussian) distribution($\mu = 0, \sigma = 1$)에 의해 선택되었다. $t$th time step에서 true reward $R_t$는 선택된 action인 $A_t$에 대한 $q(A_t)$에 gaussian noise를 더한 것이다. 


아래 그래프는 greedy($\epsilon = 0$), $\epsilon$-greedy($\epsilon = 0.01, \epsilon = 0.1$)의 sample average로 측정한 action-value estimate를 보여준다. 위 그래프는 experience를 통한 expected reward를, 아래 그래프는 optimal action을 선택한 비율을 의미한다. greedy 방식보다 $\epsilon$-greedy$(\epsilon$값이 클수록) 좋은 성능을 보이고 있다. 이는 $\epsilon$ 값의 존재로 인해 agent가 exploit하게만 동작하는 것이 아니라, 적은 확률로 랜덤하게 action을 선택하는 explore를 진행하여 suboptimal action-value에 머물지 않고 optimal action-value를 찾아가기 때문이다. 그래프에는 step이 1000까지만 표기되었지만 계속 진행해보면, $\epsilon = 0.01$ 방식이 느리게 향상되지만 두 성능 측정 모두에서 $\epsilon = 0.1$보다 더 나은 성능을 보인다. 결과적으로, $\epsilon$ 값은 조절이 성능에 영향을 미치고, 나중에는 constant 값이 아닌 시간이 지남에 따라 $\epsilon$값을 줄이는 방식을 택하기도 한다.

<center><img src="https://user-images.githubusercontent.com/127359789/224034888-80066f9d-d4a3-481f-aaa2-41e82a2489d3.png" width="60%" height="60%"></center>


<br/>

### 2.2.5 Pros and Cons of $\epsilon$-greedy and Usefulness

구체적으로 살펴보자면, $\epsilon$-greedy 방식의 장점은 task에 따라 다르다. 예를 들어, reward distrbution의 variance가 1이 아니라 10인 경우와 같이 노이즈가 많은 reward에서는 optimal action-value를 찾기위해 더 많은 explore가 필요하며  $\epsilon$-greedy는 greedy 방식보다 훨씬 더 잘 작동한다. 그러나 만약 reward distribution이 0이라면 greedy 방식은 한번만 시도한 후에 true action-value를 알 수 있기 때문에 가장 잘 작동할수 있다. 

<br/>

deterministic case에서도 다른 가정을 약화시키면 exploration이 큰 이점이 있다. bandit task가 non-stationray; true-action value가 시간에 따라 변경되었다고 가정해보자. 이 경우 suboptimal action-value를 가진 action이 현재 greedy action인 optimal action-value보다 낫도록 변경되지 않았는지 확인하기 위해서는 determinisitc한 경우에도 exploration이 필요하다. 쉽게 말하면, true-action value가 현재 action을 진행하고 변하기 때문에, action을 value에 따라 고정적으로 선택되어오던 case에서도 action-value에 대한 새로운 측정이 요구된다.

non-stationary는 강화학습에서 일반적으로 발생하는 현상이며 action이 stationary 및 deterministic한 경우라도 agent는 학습 프로세스로 인해 시간이 지남에 따라 action-value가 변경될 수 있다. 다시한번 강조하지만, 강화학습에서는 explore과 exploit의 균형이 매우 중요하다. 

<br/>

## 2.3 Incremental Implementation

지금까지 사용했던 sample average 방식의 action-value 측정 $(1)$은 구현 상, 각 action 선택에 따른 모든 reward 기록을 유지하고 계산하는 것이 일차원적인 방식이나 메모리와 계산 요구 사항이 제한없이 증가하게 되는 문제가 발생한다. 따라서 새로운 reward 처리를 위한 작고 지속적인 계산 방식을 고안한다. 

<br/>

### 2.3.1 Increment Update Rule at stationary env: sample average

어떤 action에 대한 $Q_k$가 $k$th reward estimate, 즉 $k-1$ reward의 평균을 나타낸다고 하자. 이 평균과 action에 대한 $k$th reward $R_k$가 주어지면 $k$ reward의 평균은 $(3)$과 같이 계산할 수 있다. $k=1$인 경우에도 유지되며, 임의의 $Q_1$에 대해서는  $Q_2=R_1$을 얻을 수 있다. 해당 구현에는 $Q_k$와 $k$에 대한 메모리만 필요하고 새로운 reward에 대한 계산만 필요로 한다. 


$$  \begin{align*} 
  Q_{k+1} &= \cfrac{1}{k} \sum^k_{i=1}R_i \\
 &= \cfrac{1}{k} (R_k + \sum^{k-1}_{i=1}R_i)  \\
 &= \cfrac{1}{k}(R_k + (k-1)Q_k + Q_k - Q_k) \\
 &= \cfrac{1}{k}(R_k + kQ_k - Q_k) = Q_k + \cfrac{1}{k}[R_k - Q_k]  \end{align*} \tag{3} $$

<br/>

이를 조금 쉽게 표현하자면, $(4)$와 같다. 여기서 $[Target - OldEstimate]$는 traget과 estimate의 error를 의미한다. 이는 올바른 방향으로 이끄는 $Target$을 향해 $StepSize$만큼 error를 감소시킨다. $(3)$에서의 target은 $k$th reward이고 step-size는 time step마다  변경되는 $\frac{1}{k}$의 step-size를 사용한다. 해당 교재에서는 일반적으로 step-size를 $\alpha$ 혹은 일반적으로 $\alpha_t(a)$로 표기한다. 


$$ NewEstimate \leftarrow OldEstimate + StepSize[Target - OldEstimate] \tag{4} $$

<br/>

## 2.4 Tracking a Non-stationary Problem

sample average 방법은 stationary env에서는 적절하지만, non-stationary env에서는 적합하지 않다. 이러한 경우, 오래 전 reward보다는 최근 reward에 더 큰 가중치를 부여하는 것이 좋다. 여기서 말하는 가중치는 곧 step-size를 의미하고 incremental update rule은 $(5)$와 같이 수정된다. 

<br/>

### 2.4.1 Increment Update Rule at non-stationary env: recency weighted average

결과적으로, $Q_{k+1}$는 아래와 같이 과거 rewards와 initial estimate $Q_1$의 'weighted average'가 된다. weighted average라고 부르는 이유는 모든 항의 summation이 1이기 때문이다. 


$$  \begin{align*} 
  Q_{k+1} &= Q_k + \alpha[R_k - Q_k] \tag{5} \\
  &= \alpha R_k + (1-\alpha)Q_k \\
  &= \alpha R_k + (1-\alpha)[\alpha R_{k-1}+(1-\alpha)Q_{k-1}]\\
  &= \alpha R_k + (1-\alpha)\alpha R_{k-1}+(1-\alpha)^2 Q_{k-1}\\
  &= \alpha R_k + (1-\alpha)\alpha R_{k-1}+(1-\alpha)^2 R_{k-2} + \cdots + (1-\alpha)^{k-1} \alpha R_1 + (1-\alpha)^k Q_1 \\
  &= (1-\alpha)^k Q_1 + \sum^k_{i=1} \alpha(1-\alpha)^{k-i} R_i \tag{6}
  \end{align*} $$

<br/>

$(6)$을 보면 reward $R_i$에게 적용되는 weight인 $\alpha(1-\alpha)^{k-i}$는 rewards가 얼마나 오래 되었는지인 $k-i$ 즉, time step에 종속된다. $1-\alpha$는 1보다 작은 값이며, $R_i$에 적용되는 weight는 rewards의 수가 증가할수록 exponential하게 값이 작아진다. 만약 $1-\alpha=0$이면 모든 weight는 $0^0=1$이라는 관례 때문에 맨 마지막 reward인 $R_k$만 계산된다. 이는 'exponential' 혹은 'recency weighted average'라고 불린다. 

<br/>

### 2.4.2 Guarantee True Value by changing step-size

$\alpha_k(a)$가 action $a$의 $k$th 선택 이후의 step-size라고 할 때, 경우에 따라 step-size를 변경하는 것이 더 좋다. 이전의 sample average 방법에서 사용했던 step-size인 $\alpha_k(a) = \frac{1}{k}$ 같은 경우, 대수의 법칙에 의해 true action-values로 수렴하도록 보장된다. 그러나 모든 sequence $\alpha_k(a)$의 모든 선택에 의해 수렴이 보장되는 것은 아니다. 

아래는 stochastic approximation theory에서 근거한 확률 1로 수렴하는데 필요한 조건이다. 전자는 step-size가 초기 조건이나 무작위 변동을 극복할 수 있을 만큼 충분히 크다는 것을, 후자는 결국 수렴을 보장할 수 있을만큼 step-size가 작아지는 것을 보장한다. 


$$ \sum^\infty_{k=1}\alpha_k(a) = \infty \quad and\quad \sum^\infty_{k=1}\alpha^2_k(a) < \infty \tag{7}$$

<br/>

sample average case에서는 이를 만족하나 step-size가 constant한 case에서는 후자의 조건을 충족하지 못한다. 이로인해 constant case의 경우, estimate가 완전히 수렴되지는 않고 가장 최근에 받은 reward에 의해 달라진다. 결국, 이와 같은 방법들이 등장한 이유는 사실상 강화학습의 표준인 non-stationary env를 고려하기 위함이고 $(7)$를 충족하는 step-size sequence는 매우 느린 수렴 혹은 상당한 조정이 요구된다. 참고로, step-size sequence는 응용 및 실제 연구보다는 이론 작업에서 주로 사용된다. 

<br/>

