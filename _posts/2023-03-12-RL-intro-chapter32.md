---
layout: article
title: Chapter 3. Finite Markov Decision Processes (2)
aside:
  toc: true
sidebar:
  nav: layouts
---

> 강화학습의 바이블이라고 불리는 Richard S. Sutton의 Reinforcement Learning: An Introduction을 공부하고 필자가 이해한 내용과 추가 정보들을 예제와 함께 간략하고 쉽게(?) 설명하는 방식으로 정리해봅니다. 용어 같은 경우, 원문 그대로 사용하겠지만 혹시 모를 번역 오류 및 잘못된 설명에 대한 지적과 보충 환영합니다. 

# 3. Finite Markov Decision Processes
## 3.6 Markov Decision Process
Markov property를 만족하는 강화학습 task를 Markov decision process라 한다. state space와 action space가 finite한 경우 이를 finite Markov decision process (finite MDP)라 한다. 

### 3.6.1 Finite MDP
state와 action 집합과 env의 1 step dynamics으로 정의된다. state $s$와 action $a$가 주어지면 next state $s'$와 next reward $r$은 $(6)$과 같이 표시된다. 

$$ p(s',r|s,a) = \Pr \{S_{t+1} = s', R_{t+1} = r | S_t = s, A_t = a \} \tag{6} $$

<br/>

$(5)$은 finite MDP의 dynamics를 완전히 지정한다. 책의 나머지 부분에서 제시하는 대부분의 이론은 env가 finite MDP라고 암시적으로 가정한다. $(6)$에 지정된 dynamics가 주어지면 state-action pair에 대한 env에 대해 알고 싶은 다른 모든 것들 계산이 가능하다. 다음 식들은 아래 그림의 예제에서 $s \in \{1,2\}, a \in \{1,2\},$ $\pi(a|s)$를 따른다고 가정하고 이해를 돕기위해 설명해본다. 다이어그램에서 열린 원은 state를, 닫힌 원은 state-action pair를 의미하며 앞으로 value function들의 관계를 표시할 때 자주 사용될 것이다. 

<center><img src="https://user-images.githubusercontent.com/127359789/224635776-403a3226-30de-4122-9470-3268822b5350.png" width="50%" height="50%"></center>

<br/>

**\- State-Transition Probabilities**

state $s$에서 action $a$를 수행했을 때, next state가 $s'$, reward 가 $r$일 probability 

$$ p(s', r|s,a) = \Pr\{S_{t+1} = s', R_t = r | S_t=s, A_t = a\} \tag{7}$$

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/224636513-16a742ae-f734-40e0-9228-6f3b4042d593.png" width="47%" height="52%"></center>

$$ p(s_1, r_{11}|s_1, a_1) = \Pr\{S_{t+1}=s_1, R_t = r_{11}|S_t=s_1, A_t=a_1\} = \pi(a_1|s_1)*0.1 $$

<br/>

state $s$에서 action $a$를 수행했을 때, next state가 $s'$일 state-transition probability

$$ p(s'|s,a) = \Pr\{S_{t+1} = s' | S_t=s, A_t = a\} = \sum_{r \in \mathcal{R}}p(s',r|s,a) \tag{8}$$

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/224636807-d09643a9-440d-4547-a90e-b5cb58ac19a7.png" width="50%" height="50%"></center>

$$ p(s_2|s_1, a_1) = \Pr\{S_{t+1}=s_2|S_t=s_1, A_t=a_1\} =\pi(a_1|s_1)*(0.3 + 0.6)$$

<br/>

추가로 모든 state $s$에서 모든 action $a$를 수행했을 때, next state가 $s'$, reward가 $r$이 나올 수 있는 probability은 1이다. $s$, $a$일 때, $s', r$이 나올 수 있는 가능성은 1이 되어야 한다. 

$$ \sum_{r \in \mathcal{R}} \sum_{s' \in \mathcal{S}}p(s',r|s,a) = 1, \, \forall s\in \mathcal{S}, \forall a \in \mathcal{A(s)} \tag{9}$$

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/224636908-38c2cbd4-e797-46c4-a54e-2beddfea615a.png" width="50%" height="50%"></center>

$$ \pi(a_1|s_1)*(0.1 + 0.3 + 0.6)+\pi(a_2|s_1)*(0.2 + 0.4 + 0.4) =1$$

<br/>


**\- Expected Rewards**

state $s$에서 action $a$를 수행했을 때, reward(*return 아님*)

$$ r(s,a) = \mathbb E[R_{t+1} | S_t=s, A_t=a] = \sum_{r \in \mathcal{R}}r \sum_{s' \in \mathcal{S}}p(s',r|s,a) \tag{10} $$

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/224637088-e85d4eba-0055-4509-9231-04895824608b.png" width="50%" height="50%"></center>

$$ r(s_1,a_1) = \mathbb{E} [R_{t+1} | S_t=s_1, A_t=a_1] = \pi(a_1|s_1) \* (0.1*r_{11} +0.3*r_{12} + 0.6 \* r_{13}) $$

<br/>


**\- Expected Rewards for state-action-next state**

state $s$에서 action $a$를 수행했을 때, 전자는 term은 될수있는 모든 reward $r$의 summation이고, 후자는 nest state $s'$에서 reward $r$를 받을 probability / next state $s'$으로 이동할 probability이다.

$$ r(s, a, s') = \mathbb E [R_{t+1} | S_t=s, A_t=a, S_{t+1} = s'] = \sum_{r \in \mathcal R}r\cfrac{p(s',r|s,a)}{p(s'|s,a)} \tag{11} $$ 

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/224637267-167f1061-60a8-4d5e-bc04-f1bb32232c7a.png" width="50%" height="50%"></center>

$$ r(s_1,a_1, s_2) = \mathbb{E} [R_{t+1} | S_t=s_1, A_t=a_1, S_{t+1} = s_2] = \pi(a_1|s_1)* \cfrac{0.3*r_{12}+0.6*r_{13}}{0.3+0.6} $$
<br/>

## 3.7 Value Function
value function은 given state-action pair를 수행하는 것이 얼마나 좋은지 추정하는 것으로 expected future reward 즉, 정확히는 expected return의 개념으로 정의된다. agent의 expected future return은 수행하는 action에 따라 달라지게 되므로 value function은 policy와 관련되며, policy $\pi$는 각 state $s \in \mathcal S$와 $a \in \mathcal{A(s)}$에서 state $s$에 있을 때 action $a$를 취할 probability인 $\pi(a|s)$로의 매핑이다. 

<br/>

### State-Value Function
policy $\pi$ 하의 state $s의 value는 $s$에서 시작하여 이후 $\pi$를 따를 때의 expected return이다. MDP의 경우 $v_\pi(s)$는 $v_\pi(s)$는 $(12)$과 같이 정의된다. $\mathbb{E}[\cdot]$는 policy $\pi$를 따르는 random variable의 expected value를 나타낸다. terminal state의 value는 항상 0이다. 이러한 성질의 함수  $v_\pi$를 policy $\pi$에 대한 state-value function이라고 한다. 

$$ \begin{align*}v_\pi(s) &= \mathbb{E} [G_t|S_t=s]  \\ &= \mathbb{E}_\pi \left[ {\sum^\infty_{k=0} \gamma^k R_{t+k+1} }| {S_t=s}\right] \\ &= \mathbb{E}[R_{t+1} + \gamma G_{t+1}|S_t=s] \tag{12} \end{align*} $$

<br/>

### Action-Value Function
유사하게, policy $\pi$ 하에서 state $s$에서 action $a$를 취했을 때의 value는 $s$에서 시작하여 이후 $\pi$를 따라 $a$를 취하는 expected return을 $q_\pi(s,a)$로 정의한다. 
이러한 성질의 함수 $q_\pi$를 policy $\pi$에 대한 action-value function이라고 한다. 

$$ \begin{align*} q_\pi(s,a) &= \mathbb{E}[G_t|S_t=s, A_t = a] \\ &= \mathbb{E}_\pi \left[ {\sum^\infty_{k=0} \gamma^k R_{t+k+1} }| {S_t=s, A_t=a}\right] \\ &= \mathbb{E}[R_{t+1} + \gamma G_{t+1}|S_t=s, A_t=a]\tag{13} \end{align*}$$

<br/>


### Bellman Equation for $v_\pi$

강화학습 및 동적 프로그래밍(DP) 전반에 걸쳐 사용되는 value function의 기본 속성은 recursion 관계를 만족한다. $\forall s\in \mathcal{S}, \forall a \in \mathcal{A(s)}, \forall r \in \mathcal{R}$에 대해 $(14)$의 consistency 조건은 state $s$와 next states $s'$ 사이에 유지된다. 

$(14)$는 state와 successor state 간의 관계를 나타내는 $v_\pi$의 Bellman equation을 나타낸다. start state의 값은 next state의  discounted 값과 그 과정에서의 expected reward를 더한 값과 같다. $\left[r+\gamma \mathbb{E}_\pi \left [\sum^\infty_{k=0} \gamma^k R_{t+k+2}|S_{t+1}=s' \right ] \right]$를 $\left [r+\gamma v_\pi(s') \right ]$로 치환한 과정은 앞으로도 수식 단순화를 위해 많이 쓰이므로 눈에 익혀둬야 한다. 

수식적으로, 각 $a, s', r$에 대해 probability $\pi(a|s)p(s',r|s,a)$를 계산하고 해당 probability로 괄호 안의 값에 weight를 준 다음 모든 probability를 합산하여 expected value를 얻는다. 즉, Bellman equation은 모든 probability에 대해 평균을 내며 발생한 probability에 따라 weight를 부여한다. 그 중, $(14)$의 마지막 term은 $(15)$와 같이 $q_\pi$로 표현되고 있는데 이는 $v_\pi$가 $q_\pi$로 표현될 수 있다는 것을 알 수 있다. 

$$ \begin{align*} v_\pi(s) &= \mathbb{E}_\pi[G_t|S_t=s] \\ &= \mathbb{E}_\pi[\sum^\infty_{k=0} \gamma^k R_{t+k+1}|S_t=s] \\ &= \mathbb{E}_\pi[R_{t+1} + \gamma \sum^\infty_{k=0} \gamma^k R_{t+k+2}|S_t=s] \\ &= \sum_a \pi(a|s) \sum_{s'}\sum_r p(s',r|s,a) \left [r + \gamma \mathbb{E}_\pi \left [\sum^\infty_{k=0} \gamma^k R_{t+k+2}|S_{t+1}=s' \right ] \right ] \\ &=  \tag{14} \sum_a \pi(a|s)\sum_{s',r}p(s',r|s,a) \left [ r + \gamma v_\pi(s')\right] \\ &= \sum_a \pi(a|s)q_\pi(s,a) \tag{15} \end{align*} $$

<br/>


### Bellman Equation for $q_\pi$

$$ \begin{align*}q_\pi(s) &= \mathbb{E}_\pi[G_t|S_t=s, A_t=a] \\ &= \mathbb{E}_\pi[\sum^\infty_{k=0} \gamma^k R_{t+k+1}|S_t=s, A_t=a] \\ &= \mathbb{E}_\pi[R_{t+1} + \gamma \sum^\infty_{k=0} \gamma^k R_{t+k+2}|S_t=s, A_t=a] \\ &= \sum_{a'}\sum_{s'}\sum_r p(s',r|s,a) \left [r + \gamma \mathbb{E}_\pi \left [\sum^\infty_{k=0} \gamma^k R_{t+k+2}|S_{t+1}=s', A_t=a' \right ] \right ]  \\ &=  \sum_{s', r, a'}p(s',r|s,a) \left [ r + \gamma q_\pi(s',a') \right ] \tag{16}\\ &=   \sum_{s',r}p(s',r|s,a) \left [ r + \gamma v_\pi(s')\right] \tag{17} \end{align*} $$

<br/>


### Backup Diagram

예제에서 언급했듯이 강화학습 방법의 핵심인 value function의 Bellman equation 혹은 그 관계는 다이어그램으로 나타낸다. 이러한 표현으로 successor states에서 state로 값 정보들을 다시 전송해준다. 

<center><img src="https://user-images.githubusercontent.com/127359789/224640549-68ca4b13-aa94-4d43-bd57-4f5bd3f3cab5.png" width="70%" height="70%"></center>

<br/>

