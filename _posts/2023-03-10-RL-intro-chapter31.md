---
layout: article
title: Chapter 3. Finite Markov Decision Processes
aside:
  toc: true
sidebar:
  nav: layouts
---

> 강화학습의 바이블이라고 불리는 Richard S. Sutton의 Reinforcement Learning: An Introduction을 공부하고 필자가 이해한 내용과 추가 정보들을 예제와 함께 간략하고 쉽게(?) 설명하는 방식으로 정리해봅니다. 용어 같은 경우, 원문 그대로 사용하겠지만 혹시 모를 번역 오류 및 잘못된 설명에 대한 지적과 보충 환영합니다. 

# 3. Finite Markov Decision Processes

이제 본격적으로 말하는 강화학습 문제를 정의하고 이 문제를 해결하는데 적합한 모든 방법을 강화학습 방법으로 간주한다. 이 챕터에서는 강화학습 문제를 표현하고 수학적인 형태를 알아본다. 

<br/>

## 3.1 The Agent-Environment Interface

강화학습 문제는 목표를 달성하기 위해 상호작용으로부터 학습하는 문제의 간단한 프레임워크를 의미한다. agent는 학습자 및 의사 결정자를, env는 agent의 외부 모든 것을 포함하여 상호작용하는 것으로 reward를 제공하는 일종의 시스템을 의미한다. 

<br/>

### 3.1.1 Agent-Env Interaction 
agent와 env는 discrete time step($t = 0, 1, 2, \ldots$)의 sequence 각각에서 상호작용한다. 아래 그림에서 이를 표현한다.

<center><img src="" width="60%" height="60%"></center>

1. agent는 매 time step $t$마다 env's state $S_t \in \mathcal S$에 대한 표현을 받는다. 이 때 $\mathcal S$ 는 가능한 모든 state 집합을 의미한다. 
2. 받은 state에 근거하여 action $A_t \in \mathcal A(S_t)$를 수행한다. 이 때, $\mathcal A(S_t)$는 state $S_t$에서 가능한 action 집합을 의미한다. 
3. 1 time step 이후, agent는 action에 대한 결과인 numerical reward $R_{t+1} \in \mathcal R \in \mathbb R$과 새로운 state $S_{t+1}$를 받게된다. 

각 time step에서 agent는 state에서 가능한 action을 확률에 기반하여 매핑을 구현한다. 여기서 state는 결정을 내리는데 사용가능한 모든 정보들이, action은 원하고자 하는 모든 행동이 될 수 있다. 매핑은 이전에 언급했듯이 policy라고하며 $\pi_t$로 표기된다. $\pi_t(a|s)$라면 $S_t=s$에서 $A_t =a$일 때의 확률을 의미한다. 강화학습은 경험의 결과로 policy를 변경하게 되며, 궁극적으로 장기적인 총 reward를 최대화하는 것을 목적한다. 

<br/>

### 3.1.2 Difference between Agent and Env

일반적으로 env란 agent에 의해 임의로 변경할 수 없는 모든 것을 의미한다. 따라서 agent와 env는 실제 물리적 경계와 일치하지 않는 경우가 많다. 예를 들어, 로봇이 agent라고 했을 때, 로봇의 모터와 하드웨어 또한 env로 간주된다. 물론 agent가 env 및 심지어 reward가 계산되는 방식까지 절대 알 수 없다는 것은 아니지만, 강화학습은 agent의 사전 지식을 통한 학습이 아닌 주어진 정보로부터의 완벽한 제어를 추구한다.

결국 강화학습 프레임워크는 목표 지향적 문제를 추상화하여 세부 정보가 무엇이든지 간에 agent와 env 간에 오가는 세가지 신호로 축소 가능하다고 제안한다. 선택이 이루어지는 기준(state), agent의 선택(action), agent의 목표를 정의(reward)로 완벽히 모든 의사결정 문제를 표현하기는 어려울 수 있지만 이는 널리 유용하고 적용가능하다. 

<br/>

## 3.2 Goals and Rewards
agent의 목표는 env에서 전달되는 특별한 reward signal로 표현된다. 각 time step에서  reward는 간단한 number $R_t \in \mathbb R$이다. 비공식적으로, agent는 받는 reward의 총량을 최대화 하는 것이다. 즉, 수신된 scalar signal(reward)의 누적 합계의 expected value를 극대화하는 것과 같다. 

따라서 성취하고자 하는 바를 reward로 나타내는 것이 중요하다. reward signal은 원하는 것을 달성하는 방법에 대한 사전지식이 아닌 진정 학습하기 원하는 것을 의미해야한다. 

<br/>

## 3.3 Returns

지금까지 말한 강화학습의 목적을 수식적으로 정의해본다. time step $t$ 이후에 받는 reward를 $R_{t+1}, R_{t+2}, R_{t+3}, \ldots$라고 할 때, expected return을 최대화 하기 위해서는 간단하게 모든 reward를 더하는 것으로 생각해볼 수 있다. $T$는 마지막 time step을 의미한다.

$$ G_t= R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_T \tag{1}$$

<br/>

### 3.3.1 Episodic Tasks
이러한 approach는 episode라고하는 final time step이 존재하는 application에서 의미가 있다. episode는 terminal state라고 하는 특수 state에서 끝나고 standard state 혹은 standard state의 standard distribution에서의 sample로 재설정된다. 이러한 episode가 있는 task를 episodic task라고 부르기도 한다. episodic task에서 non-terminal state 집합을 $\mathcal S, terminal state 집합을 $\mathcal S^+$라고 표기한다. 

<br/>

### 3.3.2 Process-Control Task
반면에 episodic 하지 않고, 제한없이 agent와 env 간 상호작용이 계속되는 경우도 존재한다. 이는 process-control task라고 부르고, final time step $T=\infty$이기 때문에 return 또한 무한으로 발산하게 되는 문제가 존재한다. 

<br/>

### 3.3.3 Returns considering discount concept
따라서 이 책에서는 discounting 개념을 추가하여 return을 재정의한다. agent는 future discounted rewards의 합이 최대가 되도록 action을 선택하도록 하고, 특히 expected discounted return을 최대화 하기 위해 $A_t$를 선택한다. $0 \le \gamma \le 1$는 discount rate라고 불리는 파라미터이다. 

$$ G_t= R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum^\infty_{k=0} \gamma^k R_{t+k+1} \tag{2} $$

<br/>

discount rate는 reward의 현재 value를 결정한다. $\gamma$의 존재로 인해 $k$ time step 뒤 future reward는 immediate reward의  $\gamma^{k-1}$ 밖에 되지 않는다. 만약 $\gamma < 1$라면, reward sequence ${R_k}$의 infinite sum은 finite한 값을 가진다. 

- $\gamma = 0$에 가까워질수록, agent는 즉각적인 reward 최대화에만 관심있는 'short-sighted'하게 동작한다. 이 경우에는 $R_{t+1}$만 최대화하는 방법을 배우는 것과 같다. 그러나 근시안적인 reward에만 초점을 맞출 경우, future reward에 대한 접근성이 줄어들어 결과적인 보상이 줄어들 수 있다. 
- $\gamma= 1$에 가까워질수록, agent는 future reward를 최대화 하는 것을 고려하게되고 이는 'far-sighted'하게 동작한다. 뒤 모든 time step을 고려하여, 최대화하는 방법을 배우게 된다. 

<br/>

## 3.4 Unified Notation for Episodic and Continuing Tasks
episodic 그리고 process-control task를 모두 고려하기 위해 모두를 표현할 수 있는 하나의 표기법으로 나타내야 한다.

episodic tasks는 하나의 긴 time step sequence가 아닌 finite time step sequence로 구성된다. 0부터 시작하는 각 episode $i$의 time step에 번호를 매기고, time step $t$에서의 state인 $S_{t,i}$를 참조한다. 그러나 서로 다른 episode를 구분할 필요가 없다는 것이 밝혀진 바, 앞으로는 특정 single episode를 고려하거나 모든 episode에 대해 사실인 것을 전제로 한다. 따라 위의 명시적 참조 $i$는 삭제하고 $S_t$와 같이 사용한다.  

<br/>

episode의 종료는 자신에게만 전환되고 reward가 0인 특별한 absorbing state로 들어가는 것으로 간주하여 통합할 수 있다. 아래 state transition diagram에서 속 사각형은 episode의 마지막을 의미하는 표현을 통해 두 task 모두 정의가 가능하다. 

<center><img src="" width="60%" height="60%"></center>

이러한 방식으로 우리는 $T=\infty \,\, or \,\, \gamma=1$의 가능성을 포함하여 return을 $(3)$과 같이 쓸 수 있다. 

$$ G_t = \sum^{T-t-1}_{k=0}\gamma^k R_{t+k+1} \tag{3} $$

<br/>

## 3.5 The Markov Property
우선 우리는 state signal의 설계가 아닌 state를 통해 어떤 action을 수행할지에 대한 결정에 초점을 둔다는 것을 짚고 넘어가자. state란 agent가 사용할 수 있는 모든 정보를 의미지만, state가 env에 대한 모든 정보 혹은 결정에 필요한 모든 것을 알려줄 것으로 기대하면 안된다. 

이상적으로 우리가 원하는 것은 모든 관련 정보가 유지되는 방식으로 과거의정보를 간결하게 요약하는 state signal이다. 이는 즉각적인 정보를 의미하지만 모든 과거 정보의 완전한 history를 요구하지는 않는다. 추후 정의하겠지만, 이러한 정보를 성공적으로 유지한 state signal은 Markov 또는 Markov property을 갖는다고 한다. 중요한 모든 정보가 현재 state signal에 있기 때문에 이는 'independence of path'라고도 한다. 

<br/>

### 3.5.1 General Environment
Markov property을 정의하기 전, infinite 수의 state와 reward가 있다고 가정한다. 

일반적으로 time step $t$와 $t+1$ 간 인과 관계가 있는 경우 이전에 발생한 모든 것을 고려해야 한다. 이와 같은 경우 과거 events에 대한 모든 가능한 values에 대해 완전한 probability distrubution을 $(4)$와 같이 지정해야만 dynamics 정의된다.  

$$ \Pr\{R_{t+1} = r, S_{t+1}=s'|S_0, A_0, R_1, \ldots, S_{t-1}, A_{t-1}, R_t, S_t, A_t\} \tag{4} $$

<br/>

### 3.5.2 Markov Environment
그러나 state signal이 Markov property을 가진 경우 $t+1$에서의 env의 응답은 $t$에서의 state 및 action에만 의존하며, 이 경우 env의 dynamics는 모든 $r, s', S_t, A_t$에 대해 (5)와 같이 정의된다. 

$$ p(s',r|s,a) = \Pr \{R_{t+1} = r, S_{t+1} = s' | S_t, A_t \} \tag{5} $$

<br/>

즉, state signal은 makrov property를 가지는 Markov state이며, 모든 $s', r$ 및 history $S_0, A_0, R_1, \ldots, S_{t-1}, A_{t-1}, R_t, S_t, A_t$에 대해 $(5)$가 $(4)$와 같은 경우에 이를 충족한다. 이 경우, env와 task 전체도 Markov property를 가진다고 한다. 

<br/>

env에 Markov property가 있는 경우, $(5)$의 dynamics를 통해 현재 state 및 action이 주어지면 next state 및 next reward를 예측할 수 있다. 이 방정식의 반복을 통해 현재 state에 대한 지식만으로도 모든 future state와 expected reward를 예측할 수 있다. Markov state는  action 선택을 위한 최상의 기반을 제공한다. 즉, Markov state의 function으로 action을 선택하는 policy는 완전한 hisotry의 function으로 action을 선택하는 optimal policy만큼 좋다.

만약 state signal이 Markov가 아니여도 강화학습의 state는 Markov state의 근사치로 생각하는 것이 적절하다. 특히, 우리는 state가 항상 future reward를 예측하고 action을 선택하는 좋은 기준이 되기를 원하기 때문에 Makrov state는 이러한 task에 최적의 기반을 제공한다. 이 책의 모든 이론은 Markov state signal를 가정한다. Markov가 엄격하게 적용되지 않는 경우에까지 Markov property에 기반한 이론들이 엄격하게 적용되는 것은 아니지만, 이에 대한 완전한 이해는 non-Markov task까지의 확장까지도 적용될 수 있다. 

<br/>

## 3.6 Markov Decision Process
Markov property를 만족하는 강화학습 task를 Markov decision process라 한다. state space와 action space가 finite한 경우 이를 finite Markov decision process (finite MDP)라 한다. 

### 3.6.1 Finite MDP
state와 action 집합과 env의 1 step dynamics으로 정의된다. state $s$와 action $a$가 주어지면 next state $s'$와 next reward $r$은 $(6)$과 같이 표시된다. 

$$ p(s',r|s,a) = \Pr \{S_{t+1} = s', R_{t+1} = r | S_t = s, A_t = a \} \tag{5} $$

<br/>

$(5)$은 finite MDP의 dynamics를 완전히 지정한다. 책의 나머지 부분에서 제시하는 대부분의 이론은 env가 finite MDP라고 암시적으로 가정한다. $(6)$에 지정된 dynamics가 주어지면 state-action pair에 대한 env에 대해 알고 싶은 다른 모든 것들 계산이 가능하다. 


**\- State-Transition Probabilities**

state $s$에서 action $a$를 수행했을 때, next state가 $s'$, reward 가 $r$일 전이 확률 

$$ p(s', r|s,a) = \Pr\{S_{t+1} = s', R_t = r | S_t=s, A_t = a\} \tag{7}$$

<br/>

state $s$에서 action $a$를 수행했을 때, next state가 $s'$일 전이 확률 

$$ p(s'|s,a) = \Pr\{S_{t+1} = s' | S_t=s, A_t = a\} = \sum_{r \in \mathcal{R}}p(s',r|s,a) \tag{8}$$

<br/>


추가로 모든 state $s$에서 action $a$를 수행했을 때, next state가 $s'$, reward가 $r$이 나올 수 있는 확률은 1이다. $s$, $a$일 때, $s', r$이 나올 수 있는 가능성은 1이 되어야 한다. 

$$ \sum_{r \in \mathcal{R}} \sum_{s' \in \mathcal{S}}p(s',r|s,a) = 1, \, \forall s\in \mathcal{S}, a \in \mathcal{A(s)} \tag{8}$$

<br/>

**\- Expected Rewards**

state $s$에서 action $a$를 수행했을 때, reward(*return 아님*)

$$ r(s,a) = \mathbb E[R_{t+1} | S_t=s, A_t=a] = \sum_{r \in \mathcal{R}}r \sum_{s' \in \mathcal{S}}p(s',r|s,a) \tag{6}$$

<br/>


**\- Expected Rewards for state-action-next state**
$$ r(s, a, s') = \mathbb E[R_{t+1} | S_t=s, A_t=a, S_{t+1} = s'] = \cfrac{\sum_{r \in \mathcal R}rp(s',r|s,a)}{p(s'|s,a)} \tag{9} $$ 

<br/>


## 3.7 Value Function
value function은 given state-action pair를 수행하는 것이 얼마나 좋은지 추정하는 것으로 expected future reward 즉, 정확히는 expected return의 개념으로 정의된다. agent의 expected future return은 수행하는 action에 따라 달라지게 되므로 value function은 policy과 관련하여 정의된다. 이 때, policy $\pi$는 각 state $s \in \mathcal S$와 $a \in \mathcal{A(s)}$에서 state $s$에 있을 때 action $a$를 취할 확률인 $\pi(a|s)$로의 매핑이다. 

<br/>

### State-Value Function
policy $\pi$ 하의 state $s의 value는 $s$에서 시작하여 이후 $\pi$를 따를 때의 expected return이다. MDP의 경우 $v_\pi(s)$는 $v_\pi(s)$는 $(10)$과 같이 정의된다. $\mathbb{E}[\cdot]$는 policy $\pi$를 따르는 agent의 random variable의 expected value를 나타낸다. terminal state의 value는 항상 0이다. 이러한 성질의 함수  $v_\pi$를 policy $\pi$에 대한 state-value function이라고 한다. 

$$ v_\pi(s) = \mathbb{E}[G_t|S_t=s] = \mathbb{E}_\pi \left[ {\sum^\infty_{k=0} \gamma^k R_{t+k+1} }| {S_t=s}\right] \tag{10} $$

<br/>

### Action-Value Function
유사하게, policy $\pi$ 하에서 state $s$에서 action $a$를 취했을 때의 value는 $s$에서 시작하여 이후 $\pi$를 따라 $a$를 취하는 expected return을 $q_\pi(s,a)$로 정의한다. 
이러한 성질의 함수 $q_\pi$ policy $\pi$에 대한 action-value function이라고 한다. 

$$ q_\pi(s,a) = \mathbb{E}[G_t|S_t=s, A_t = a] = \mathbb{E}_\pi \left[ {\sum^\infty_{k=0} \gamma^k R_{t+k+1} }| {S_t=s, A_t=a}\right] \tag{11} $$


<br/>

### How to Estimate Value Functions

뒤 챕터들에서 배우겠지만 value function을 estimate하는 방법을 간략하게 소개해본다. 

$v_\pi(s)$와 $q_\pi(s,a)$는 experience를 통해 추정할 수 있다. 예를 들어 agent가 policy $\pi$를 따르고 state $s$ 이후의 infinity states들에 대한 actual return의 평균을 유지하면 그 평균은 state-value $v_\pi(s)$로 수렴한다. 마찬가지로 state에서 취한 action에 대한 평균을 유지하면 그 평균 또한 action-value $q_\pi(s,a)$로 수렴한다. 이처럼 actual returns의 많은 무작위 sample에 대한 평균을 포함하는 종류의 추정 방법을 Monte Carlo 방법이라 부른다. 

state에 대해 개별적으로 평균을 유지하는 것은 실용적이지 않아 대신 agent는 $v_\pi(s)$와 $q_\pi(s,a)$를 매개변수화된 function으로 유지하고 observed return과 더 잘 일치하도록 매개변수를 조정해야 한다. 이러한 function approximator 방법은 정확한 estimate 생성이 가능하다고 한다. 

<br/>

### Bellman Equation

강화학습 및 동적 프로그래밍(DP) 전반에 걸쳐 사용되는 value function의 기본 속성은 특정 재귀 관계를 만족한다. 모든 policy $\pi$ 및 모든 state $s$에 대해 $(12)$의 consistency 조건은 $s$와 후속 states 사이에 유지된다. 
w
$$ \begin{align*} v_\pi(s) &= \mathbb{E}_\pi[G_t|S_t=s] \\ &= \mathbb{E}_\pi[\sum^\infty_{k=0} \gamma^k R_{t+k+1}|S_t=s] \\ &= \mathbb{E}_\pi[R_{t+1} + \gamma \sum^\infty_{k=0} \gamma^k R_{t+k+2}|S_t=s] \\ &= \sum_a \pi(a|s) \sum_{s'}\sum_r p(s',r|s,a) \left [r + \gamma \mathbb{E}_\pi \left [\sum^\infty_{k=0} \gamma^k R_{t+k+2}|S_{t+1}=s' \right ] \right ] \\ &=  \sum_a \pi(a|s)\sum_{s',r}p(s',r|s,a) \left [ r + \gamma v_\pi(s')\right] \tag{12} \end{align*} $$

