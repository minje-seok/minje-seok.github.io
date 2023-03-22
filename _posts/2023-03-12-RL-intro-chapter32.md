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

Markov property를 만족하는 강화학습 task를 Markov decision process (MDP)라 한다. state space와 action space가 finite한 경우 이를 finite Markov decision process (finite MDP)라 한다. 

### 3.6.1 Finite MDP
state와 action 집합과 env의 1 step dynamics으로 정의된다. state $s$와 action $a$가 주어지면 next state $s'$와 next reward $r$은 $(6)$과 같이 표시된다. $(6)$에 지정된 dynamics가 주어지면 state-action pair에 대한 env에 대해 알고 싶은 다른 모든 것들 계산이 가능하다.

$$ p(s',r|s,a) = \Pr \{S_{t+1} = s', R_{t+1} = r | S_t = s, A_t = a \} \tag{6} $$

<br/>

아래 보게될 식들은 개인적으로 이해를 돕기위해 만든 예제에서 $s \in \{1,2 \}, a \in \{1,2\},\pi(a \mid s)$를 따른다고 가정하고 설명해본다. 다이어그램에서 열린 원은 state를, 닫힌 원은 state-action pair를 의미하며 앞으로 value function들의 관계를 표시할 때 자주 사용될 것이다. 주어진 transition probability와 reward로 현재 같은 state $s$에서 왼쪽은 $v(s)$, 오른쪽은 $q(s,a)$를 나타낸다. 

<center><img src="https://user-images.githubusercontent.com/127359789/226820686-18f53690-caa4-4017-994f-5c8fa0fc3e37.png" width="80%" height="80%"></center>

<br/>

**\- State-Transition Probabilities**

state $s$에서 action $a$를 수행했을 때, next state가 $s'$, reward 가 $r$일 probability 

$$ p(s', r|s,a) = \Pr\{S_{t+1} = s', R_t = r | S_t=s, A_t = a\} \tag{7}$$

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/226821104-d8b6d7ff-06f3-418d-b2ab-d6947fbf5f8e.png" width="80%" height="80%"></center>

$$ p(s_1, r_{11}|s_1, a_1) = \Pr\{S_{t+1}=s_1, R_t = r_{11}|S_t=s_1, A_t=a_1\} = \pi(a_1|s_1) \times 0.1 $$

<br/>

state $s$에서 action $a$를 수행했을 때, next state가 $s'$일 state-transition probability

$$ p(s'|s,a) = \Pr\{S_{t+1} = s' | S_t=s, A_t = a\} = \sum_{r \in \mathcal{R}}p(s',r|s,a) \tag{8}$$

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/226821325-cec989e1-5c0f-4c4d-8ac4-582eb8de3e96.png" width="80%" height="80%"></center>

$$ p(s_2|s_1, a_1) = \Pr\{S_{t+1}=s_2|S_t=s_1, A_t=a_1\} =\pi(a_1|s_1) \times(0.3 + 0.6)$$

<br/>

추가로 모든 state $s$에서 모든 action $a$를 수행했을 때, next state가 $s'$, reward가 $r$이 나올 수 있는 probability은 1이다. $s$, $a$일 때, $s', r$이 나올 수 있는 가능성은 1이 되어야 한다. 

$$ \sum_{r \in \mathcal{R}} \sum_{s' \in \mathcal{S}}p(s',r|s,a) = 1, \, \forall s\in \mathcal{S}, \forall a \in \mathcal{A(s)} \tag{9}$$

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/226821413-3d147da9-54fa-476d-aa67-78113f6e2ccc.png" width="90%" height="90%"></center>

$$ \pi(a_1|s_1)*(0.1 + 0.3 + 0.6)+\pi(a_2|s_1) \times(0.2 + 0.4 + 0.4) =1$$

<br/>


**\- Expected Rewards**

state $s$에서 action $a$를 수행했을 때, reward(*return 아님*)

$$ r(s,a) = \mathbb E[R_{t+1} | S_t=s, A_t=a] = \sum_{r \in \mathcal{R}}r \sum_{s' \in \mathcal{S}}p(s',r|s,a) \tag{10}$$

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/226821520-e943aa73-5871-4891-8fda-6fa94357290c.png" width="80%" height="80%"></center>

$$ r(s_1,a_1) = \mathbb E \left [R_{t+1} | S_t=s_1, A_t=a_1 \right ] = \pi(a_1|s_1) \times (0.1 \times r_{11} +0.3 \times r_{12} + 0.6 \times r_{13}) $$

<br/>


**\- Expected Rewards for state-action-next state**

state $s$에서 action $a$를 수행했을 때, 전자는 term은 될수있는 모든 reward $r$의 summation이고, 후자는 nest state $s'$에서 reward $r$를 받을 probability / next state $s'$으로 이동할 probability이다.

$$ r(s, a, s') = \mathbb E[R_{t+1} | S_t=s, A_t=a, S_{t+1} = s'] = \sum_{r \in \mathcal R}r\cfrac{p(s',r|s,a)}{p(s'|s,a)} \tag{11} $$ 

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/226821635-45190eef-cba4-4d43-ab94-1ad240a01b56.png" width="80%" height="80%"></center>

$$ r(s_1,a_1, s_2) = \mathbb E[R_{t+1} | S_t=s_1, A_t=a_1, S_{t+1} = s_2] = \pi(a_1|s_1)\times \cfrac{0.3 \times r_{12} + 0.6 \times r_{13} }{ 0.3 + 0.6 } $$
<br/>

## 3.7 Value Function
value function은 given state에서 특정 action을 수행하는 것이 얼마나 좋은지 추정하는 것으로 expected future reward; 정확히는 expected return으로 정의된다. agent의 expected future return은 수행하는 action에 따라 달라지게 되므로 value function은 policy와 연관되며, policy $\pi$는 각 state $s \in \mathcal S$와 $a \in \mathcal{A(s)}$에서 state $s$에 있을 때 action $a$를 취할 probability인 $\pi(a|s)$로의 매핑을 의미한다. 

<br/>

### 3.7.1 State-Value Function
policy $\pi$ 하의 state $s$의 value는 $s$에서 시작하여 이후 $\pi$를 따를 때의 expected return을 $v_\pi(s)$로 정의한다. $\mathbb{E}[\cdot]$는 policy $\pi$를 따르는 random variable의 expected value를 나타낸다. terminal state의 value는 항상 0이다. 이러한 성질의 함수를 policy $\pi$에 대한 state-value function $v_\pi$라고 한다. 

$$ \begin{align*}v_\pi(s) &= \mathbb{E}[G_t|S_t=s]  \\ &= \mathbb{E}_\pi \left[ {\sum^\infty_{k=0} \gamma^k R_{t+k+1} }| {S_t=s}\right] \\ &= \mathbb{E}[R_{t+1} + \gamma G_{t+1}|S_t=s] \tag{12} \end{align*} $$

<br/>

### 3.7.2 Action-Value Function
유사하게, policy $\pi$ 하에서 state $s$에서 action $a$를 취했을 때의 value는 $s$에서 시작하여 이후 $\pi$를 따라 $a$를 취하는 expected return을 $q_\pi(s,a)$로 정의한다. 
이러한 성질의 함수 $q_\pi$를 policy $\pi$에 대한 action-value function이라고 한다. 

$$ \begin{align*} q_\pi(s,a) &= \mathbb{E}[G_t|S_t=s, A_t = a] \\ &= \mathbb{E}_\pi \left[ {\sum^\infty_{k=0} \gamma^k R_{t+k+1} }| {S_t=s, A_t=a}\right] \\ &= \mathbb{E}[R_{t+1} + \gamma G_{t+1}|S_t=s, A_t=a]\tag{13} \end{align*}$$

<br/>

## 3.8 Bellman Equation 
Bellman equation을 사용하면 강화학습 및 동적 프로그래밍(DP) 전반에 걸쳐 사용되는 value function을 계산하고 value function 간 특정 recursion 관계를 확인할 수 있다. 중요하게 볼 부분은 expected return인 value function이 successor expected return의 합으로 표현된다는 점과 $(14)$와 $(15)$의 마지막 term에서와 같이 $v_\pi$와 $q_\pi$가 서로를 정의할 수 있다는 것이다. 


### 3.8.1 Bellman Equation for $v_\pi$

$(14)$는 state와 successor state 간의 관계를 나타내는 $v_\pi$의 Bellman equation을 나타낸다.
$\forall s\in \mathcal{S}, \forall a \in \mathcal{A(s)}, \forall r \in \mathcal{R}$에 대해 consistency 조건이 state $s$와 next states $s'$ 사이에 유지된다. start state의 값은 next state의  discounted expected return과 reward를 더한 값과 같다. 

수식적으로, 각 $a, s', r$에 대해 probability $\pi(a \mid s)p(s',r \mid s,a)$를 계산하고 해당 probability로 괄호 안의 값에 weight를 준 다음 모든 probability를 합산하여 expected value를 얻는다. $\left[r+\gamma \mathbb{E}_\pi \left [\sum^\infty_{k=0} \gamma^k R_{t+k+2}\mid S_{t+1}=s' \right ] \right]$를 $\left [r+\gamma v_\pi(s') \right ]$로 치환한 과정은 앞으로도 수식 단순화를 위해 많이 쓰이므로 눈에 익혀둬야 한다. 


$$ \begin{align*} v_\pi(s) &= \mathbb{E}_\pi\left [G_t|S_t=s \right] \\ &= \mathbb{E}_\pi \left [\sum^\infty_{k=0} \gamma^k R_{t+k+1}|S_t=s \right ] \\ &= \mathbb{E}_\pi \left [R_{t+1} + \gamma \sum^\infty_{k=0} \gamma^k R_{t+k+2}|S_t=s \right ] \\ &= \sum_a \pi(a|s) \sum_{s'}\sum_r p(s',r|s,a) \left [r + \gamma \mathbb{E}_\pi \left [\sum^\infty_{k=0} \gamma^k R_{t+k+2}|S_{t+1}=s' \right ] \right ] \\ &=   \sum_a \pi(a|s)\sum_{s',r}p(s',r|s,a) \left [ r + \gamma v_\pi(s') \right ] \\ &= \sum_a \pi(a|s)q_\pi(s,a) \tag{14} \end{align*} $$

<br/>


### 3.8.2 Bellman Equation for $q_\pi$

$(15)$는 state-action pair와 successor state-action pair 간의 관계를 나타내는 $v_\pi$의 Bellman equation을 나타낸다. $\forall s\in \mathcal{S}, \forall a \in \mathcal{A(s)}, \forall r \in \mathcal{R}$에 대해 $(15)$의 consistency 조건이 state $s$-action $a$ pair와 next states $s'$-action $a'$ pair사이에 유지된다. start state-action pair의 값은 next state-action pair의 discounted expected return과 reward를 더한 값과 같다. 

수식적으로, 각 $a', s', r$에 대해 probability $p(s',a',r \mid s,a)$를 계산하고 해당 probability로 괄호 안의 값에 weight를 준 다음 모든 probability를 합산하여 expected value를 얻는다. 여기서도, $\left[r+\gamma \mathbb{E}_\pi \left [\sum^\infty_{k=0} \gamma^k R_{t+k+2} \mid S_{t+1}=s', A_t=a' \right ] \right]$를 $\left [r+\gamma q_\pi(s', a') \right ]$로 치환한 과정을 볼 수 있다. 

$$ \begin{align*}q_\pi(s,a) &= \mathbb{E}_\pi[G_t|S_t=s, A_t=a] \\ &= \mathbb{E}_\pi \left [\sum^\infty_{k=0} \gamma^k R_{t+k+1}|S_t=s, A_t=a \right ] \\ &= \mathbb{E}_\pi \left [R_{t+1} + \gamma \sum^\infty_{k=0} \gamma^k R_{t+k+2}|S_t=s, A_t=a \right ] \\ &= \sum_{a'}\sum_{s'}\sum_r p(s',a',r|s,a) \left [r + \gamma \mathbb{E}_\pi \left [\sum^\infty_{k=0} \gamma^k R_{t+k+2}|S_{t+1}=s', A_t=a' \right ] \right ]  \\ &=  \sum_{s', r, a'}p(s',a',r|s,a) \left [ r + \gamma q_\pi(s',a') \right ] \\ &=   \sum_{s',r}p(s',r|s,a) \left [ r + \gamma v_\pi(s')\right] \tag{15} \end{align*} $$

<br/>


### 3.8.3 Backup Diagram for $v_\pi$ and $q_\pi$

$v_\pi$와 $q_\pi$의 recursive 관계는 current state(or state-action pair)의 expected value를 successor state-action pair(or state)의 expected value로 부터 계산, 다시 말해 아래에서 위 방향으로 계산이 이루어지기 때문에 Backup diagram으로 표현할 수 있다. 


<center><img src="https://user-images.githubusercontent.com/127359789/224640549-68ca4b13-aa94-4d43-bd57-4f5bd3f3cab5.png" width="80%" height="80%"></center>

<br/>

state-value function $v_\pi(s)$는 $q_\pi(s,a)$의 policy $\pi(a|s)$에 기반한 weighted average로, action-value function $q_\pi$는 reward와 $v_\pi(s')$의 state-transition probability에 기반한 weighted average로 이해할 수 있다. 

<center><img src="https://user-images.githubusercontent.com/127359789/226821716-09f4c6ad-6505-4960-b060-d9033410cc9e.png" width="75%" height="75%"></center>

<br/>

## 3.9 Optimal Value Functions

optimal policy 아래 정의되는 value function을 optimal value function이라고 한다. policy $\pi$는 expected return이 모든 states에 대해 policy $\pi'$보다 크거나 같으면 policy $\pi'$보다 낫거나 같다고 정의되며, 즉 $\forall s \in \mathcal S$에 대해 $v_\pi(s) \ge v_{\pi'}(s)$인 경우, $\pi \ge \pi'$이다. optimal policy $\pi'$는 둘 이상 있을 수 있고 모든 optimal policy는 $\pi'$로 표기되며 이때 value function는 동일하다. 


$$ v_\ast(s) = \max_\pi v_\pi(s), \quad \forall s \in \mathcal{S} \tag{16} $$

$$ q_\ast(s,a) = \max_\pi q_\pi(s,a), \quad \forall s \in \mathcal{S} ,\forall a \in \mathcal{A(s)}\tag{17} $$

<br/>

$v_\pi$와 $q_\pi$는 expected return를 계산하는데 action까지 결정된 상태인지 여부에 대한 차이만 존재하기 때문에 직관적으로  $v_\ast(s)$와 $q_\ast(s,a)$는 값이 같음을 알 수 있다. 만약 optimal policy $\pi'$를 따른다면, 같은 optimal state-value function $v_\ast$와 optimal action-value function $q_\ast$를 공유하게 된다.

$$ q_\ast(s,a) = \mathbb{E} \left [ R_{t+1} + \gamma v_\ast(S_{t+1}) \mid S_t =s, A_t =a \right ] \tag{18}$$

<br/>

### 3.9.1 Bellman Optimality Equation

Bellman equation을 optimal policy $\pi_*$하에 유도한 것이다. 직관적으로 Bellman optimality equation은 $\pi_\ast$하의 state-value가 해당 state에서 optimal action에 대한 expected return과 같아야 한다. $(19)$의 마지막 두 equation은 $v_\ast$의 Bellman optimality equation의 두 가지 form을 나타낸다. 

$$ \begin{align*} v_\ast(s) &= \max_{a \in \mathcal A (s)} q_{\pi_\ast}(s,a) \\ 
&= \max_a \mathbb E_{\pi_\ast} \left [ G_t |S_t=s, A_t=a \right ] \\
&= \max_a \mathbb E_{\pi_\ast} \left [ \sum^{\infty}_{k=0} \gamma^k R_{t+k+1} \mid S_t=s, A_t=a \right ] \\
&= \max_a \mathbb E_{\pi_\ast} \left [ R_{t+1} + \sum^\infty_{k=0} \gamma^k R_{t+k+2} \mid S_t=s, A_t=a \right  ] \\
&= \max_a \mathbb E \left [ R_{t+1} + \gamma v_\ast(S_{t+1}) \mid S_t=s, A_t=a  \right ] \\ 
&= \max_{a \in \mathcal A(s)} \sum_{s',r}p(s', r|s,a) \left [r +\gamma v_\ast(s') \right ] \tag{19} \end{align*} $$

$$ \begin{align*} q_\ast(s,a) &= \max_a \mathbb E \left [ R_{t+1} + \gamma v_\ast(S_{t+1}) \mid S_t=s, A_t=a  \right ] \\ 
&= \mathbb E \left [ R_{t+1} + \gamma \max_{a'} q_\ast(S_{t+1}, a') \mid S_t = s, A_t =a \right ] \\ 
&= \sum_{s',r}p(s', r|s,a) \left [ r + \gamma \max_{a'} q_\ast(s',a') \right ] \tag{20} \end{align*} $$

<br/>

### 3.8.3 Backup Diagram for $v_\ast$ and $q_\ast$
finite MDP에서 Bellman optimality equation에는 policy와 독립적인 solution이 존재한다. 실제로, 각 state에 대해 각각 하나의 equation이 존재하는 시스템이므로 env의 dynamics $p(s',r \mid s,a)$가 알려진 경우 이론적으로 non-linear equation system을 풀기 위한 방법을 사용하여 $v_\ast$와  $q_\ast$에 대한 equation set를 풀 수 있다. 

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/226785353-40edec61-ff56-4d5e-954e-47adfcf83bea.png" width="80%" height="80%"></center>


$v_\ast$ 및 $q_\ast$에 대한 Bellman optimaility equation에서 고려된 future state와 future action을 보여주는 Backup diagram이다. $v_\pi$와 $q_\pi$에 대한 Backup diagram과 동일하지만, agent의 선택지점에는 policy에 대해 주어진 expected return이 아니라 해당 선택에 대한 최대값이 취해진다. $v_\ast(s)$는 뒤따르는 $q(s,a)$ 중 최대가 되는 쪽으로, $q_\ast(s,a)$는 뒤따르는 $v_(s')$들의 최대값에 대한 가중 평균이다. 


<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/226822038-54ede24a-ee22-4326-927b-9bcddd7a2af9.png" width="75%" height="75%"></center>

$v_\ast$가 있으면 해당 state에서 max action-value를 가진 action이 optimal action이 된다. $v(s')$은 $q(s,a)$로 표현되므로, 결국 $v_\ast$에 대해 greedy한 것이 optimal policy라는 것이다. $v_\ast$는 이미 가능한 모든 future behavior의 reward 결과를 고려하고 있기때문에, 단기 결과만을 기준으로 action을 선택하더라도 장기적 관점에서 실제로 optimal하다. $v_\ast$를 통해 optimal expected long-term return은 각 state에서 즉시 사용할 수 있는 수치로 변환된다. 따라서 greedy한 action 선택은 long-term optimal action을 제공한다. 


$q_\ast$가 있으면 optimal action 선택이 더 쉬워진다. $q_\ast$는 이미 state에 대한 action-value를 캐시하고 있기 때문에 agent는 action 검색을 수행할 필요조차 없다. 이는 각 state-action pair에 대해 즉시 사용할 수 있는 값으로 optimal expected long-term return을 제공한다. 따라 state-action pair를 나타내는 비용으로 optimal action-value function은 env의 dynamics에 대해 알 필요 없이; 가능한 successor state 및 그들의 value에 대해 알 필요 없이 optimal action을 선택 가능하다. 

<br/>


## 3.9 Optimality and Approximation

### 3.9.1 Constraint of Bellman Optimality Equation

그렇다면 Bellman optimality equation대로 풀면 강화학습 문제에서 최적의 policy를 찾을 수 있다고 생각할 수 있다. 이러한 방식의 solution은 다음의 최소 세 가지 가정에 의존하고 있다. (1) env의 dynamics를 정확하게 알고 있다. (2) solution 계산을 완료하기에 충분한 계산 리소스가 존재한다. (3) Makrov property를 만족한다. 그러나 이러한 가정을 완벽하게 만족하는 경우는 거의 없기 때문에 사실상 강화학습에서는 일반적으로 approximate solution에 만족하게 된다. 많은 강화학습 방법은 expected transition에 대한 지식이 아닌 experienced transition을 사용하여 Bellman optimality equation을 approximately하게 해결한다. 

<br/>

우리가 Bellman optimality equation에 기반하여 구한 optimal value function와 optimal policy는 env의 dynamics에 대한 완벽한 model이 있더라도 엄청난 계산 비용으로만 생성이 가능하다. 우리는 이러한 이론적 속성을 이해하고 agent가 다양한 방식으로 approximate하게끔 노력한다. 결국 agent는 single time step에서 수행할 수 있는 계산 능력 또한 고려되어야 한다. 

사용 가능한 메모리 또한 중요한 제약 조건이다. value function, policy, model 등을 구축하려면 많은 양의 메모리를 필요로 한다. 적은 finite state set가 있는 task에서는 tabular 방식으로 사용할 수 있지만, 일반적인 문제에서는 tabular에 저장할 수 있는 것보다 훨씬 많은 state가 존재한다. 이러한 경우, function은 보다 간결한 매개변수화된 함수 표현을 사용하여 approximate된다. 

계속 강화학습 문제를 approximate하게 푸는 방식의 프레임워크를 강조하는데, 이는 유용한 approximation을 얻을 수 있는 기회를 제공한다. 강화학습의 online 특성은 자주 발생하는 state에서 올바른 action을 내리는 학습에 더 많은 노력을 기울이는 방식으로 optimal policy를 approximate할 수 있게 한다. 이는 MDP를 approximately하게 해결하는 강화학습만의 핵심 속성이다. 

<br/>

## 3.10 Summary

우리는 env와 agent 간 상호작용을 표현하는 프레임워크를 통해 강화학습 문제를 정의하고, task 종류에 의존하지 않는 agent의 목표인 return을 계산했다. 또한 대부분의 강화학습 문제에서 가정되는 Markov property를 만족하는 finite MDP에서의 probability 및 reward 계산과 expected return을 의미하는 value function을 구할 수 있었다. state-value function $v_\pi$와 action-value function $q_\pi$의 Bellman equation에서 확인 가능한 recursive한 관계에서 확장된 Bellman optimality equation으로 optimal policy까지 구할 수 있다는 결론에 이르렀다. 그러나 이러한 방식은 현실적으로 어려운 가정과 막대한 계산 비용을 요구하므로, 우리는 Bellman optimality equation을 approximate하게 해결하는 approach들을 앞으로 배울 예정이다.  
