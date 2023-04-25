---
layout: article
title: Chapter 4. Dynamic Programming (1)
aside:
  toc: true
sidebar:
  nav: layouts
---

> 강화학습의 바이블이라고 불리는 Richard S. Sutton의 Reinforcement Learning: An Introduction을 공부하고 필자가 이해한 내용과 추가 정보들을 예제와 함께 간략하고 쉽게(?) 설명하는 방식으로 정리해봅니다. 용어 같은 경우, 원문 그대로 사용하겠지만 혹시 모를 번역 오류 및 잘못된 설명에 대한 지적과 보충 환영합니다. 

# 4. Dynamic Programming
dynamic programming(DP)는 완벽한 env model이 주어졌을 때, optimal policy를 찾을 수 있는 알고리즘을 의미한다. 막대한 계산 비용으로 인한 적용의 제한성 때문에, 생성된 완벽하지 않은 env model에서 계산을 줄인 강화학습 방법들은 궁극적으로 DP와 동일한 효과를 달성하는 것을 목적한다. DP는 continuous state space와 continuous action space를 quantize한 뒤, finite-state DP 방법을 적용하여 일반적인 approximate solution을 얻는 continuous domain으로의 확장 또한 가능하다. 

이전 챕터에서는 강화학습 문제를 풀기 위해서 Bellman optimality equation을 만족하는 optimal value function $v_\ast$ 또는 $q_\ast$을 사용하여 optimal policy를 찾을 수 있다고 했다. DP에서는 원하는 $v_\ast, q_\ast$에 대한 approximation을 개선하기 위해 $(1),(2)$와 같이 Bellman equation으로부터 얻은 update rule을 적용한다. 

$$ \begin{align*} v_\ast(s) 
&= \max_a \mathbb E \left [ R_{t+1} + \gamma v_\ast(S_{t+1}) \mid S_t=s, A_t=a  \right ] \\ 
&= \max_{a \in \mathcal A(s)} \sum_{s',r}p(s', r|s,a) \left [r +\gamma v_\ast(s') \right ] \tag{1} \end{align*} $$

$$ \begin{align*} q_\ast(s,a) &= \max_a \mathbb E \left [ R_{t+1} + \gamma v_\ast(S_{t+1}) \mid S_t=s, A_t=a  \right ] \\ 
&= \sum_{s',r}p(s', r|s,a) \left [ r + \gamma \max_{a'} q_\ast(s',a') \right ] \tag{2} \end{align*} $$

<br/>

지금부터는 env가 $s \in \mathcal{S}, a \in \mathcal{A}(s), r \in \mathcal {R}, s' \in \mathcal {S^+}$ ($\mathcal {S^+}$는 episodic task에서 $\mathcal{S}$ + terminal state)가 모두 finite한 finite MDP이고, dynamics는 $p(s', r \mid s,a)$ probability의 집합으로 주어진다고 가정한다. 

<br/>

## 4.1 Policy Evaluation
### 4.1.1 Prediction Problem
우선 임시 policy $\pi$를 따르는 state-value function $v_\pi$를 계산하고자 하는 것을 prediction problem이라고 한다. 이전 챕터에서는 Bellman equation에 기반하여 value function을 $(3)$과 같이 계산했다. $v_\pi$의 존재와 고유성은 $\gamma < 1$ 혹은 모든 state에 대해 termination이 policy $\pi$ 하에서 약속되는 한에서 보장된다. 

$$ \begin{align*} v_\pi(s) &= \mathbb{E}_\pi \left [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+2} + \cdots \mid S_t = s\right ] \\
&= \mathbb{E}_\pi \left [R_{t+1} + \gamma v_\pi(S_{t+1})\mid S_t = s\right ] \\
&= \sum_a \pi(s \mid a) \sum_{s',r}p(s', r|s,a) \left [r +\gamma v_\pi(s') \right ] \tag{3} \end{align*} $$

<br/>

### 4.1.2 Iterative Policy Evaluation
만약 env의 dynamics가 완벽하게 fully-known하다면, $(3)$은 simutaneous linear equation $v_\pi(s)$의 system으로 볼 수 있다. 이러한 경우 initial value function $v_0$를 임의로 선택되어 초기화하고(*보통 terminal state를 제외하고 0으로*), 각 next value function의 approximation은 $(4)$의 Bellman equation을 update rule로 iterative하게 사용하여 solution을 얻는다. 

$v_\pi$에 대한 Bellman equation이 $v_k = v_\pi$ 경우에도 동등하게 보장되기 때문에 동일하게 update rule이 적용 가능하다. 실제로 $v_0, v_1, v_2, \ldots$의 sequence $v_k$는 일반적으로 $v_\pi$의 존재를 보장하는 동일 조건 하에서 $k \rightarrow \infty$일 때 $v_\pi$로 수렴한다. 이러한 알고리즘을 iterative policy evaluation이라고 한다. 

$$ \begin{align*} v_{k+1}(s) &= \mathbb{E}_\pi \left [R_{t+1} + \gamma v_k(S_{t+1})\mid S_t = s\right ] \\
&= \sum_a \pi(s \mid a) \sum_{s',r}p(s', r|s,a) \left [r +\gamma v_k(s') \right ] \tag{4} \end{align*} $$

<br/>

### 4.1.3 Full Backup
$v_k$로부터 successive approximation $v_{k+1}$을 생성하기 위해서 iterative policy evaluation는 다음의 과정을 수행한다. 각 state $s$의 new value를 구하기 위해 현재 evaluate되는 policy에서 가능한 모든 one-step transition에서의 successor states $s'$의 old value와 expected immediate reward의 합으로 교체한다. 이러한 과정을 full backup이라고 한다. state 또는 state-action pair이 backup되는지 여부와 successor state의 estimated value가 결합되는 방식에 따라 여러 full backup이 존재한다. 

iterative policy evaluation의 각 iteration은 next approximate value function $v_{k+1}$을 생성하기 위해 모든 state value $v_k$를 backup한다. DP 알고리즘에서 수행되는 모든 backup은 sample의 next state가 아니라 가능한 모든 next state를 기반으로 하기 때문에 full backup이라고 한다. 

<br/>

## 4.2 In-Place Iterative Policy Evaluation

상식적으로 full bakcup을 진행하기 위해서는 old value와 new value를 저장할 두 개의 array를 필요로 하지만 하나의 array만을 사용하여 'in-place'하게 update할 수도 있다. 이렇게 되면 next state가 backup되는 순서에 따라, state의 old value가 아닌 new value가 next state 계산에 사용되게 되지만 결국  $v_\pi$로 수렴한다. in-place 알고리즘에서 state가 backup되는 순서는 수렴 속도에 상당한 영향을 미치고, 2-array보다 빠르게 수렴한다. 일반적으로 DP 알고리즘은 in-place를 염두에 둔다. 

<br/>

### 4.2.1 Termination of Algorithm

iterative policy evaluation는 현실적으로 수렴이 이루어질때까지 반복할 수 없다는 constraint가 존재한다. 일반적으로는 매 sweep(*모든 state에서 value의 순차적인 update*) 이후 $\max_{s \in \mathcal{S}} \mid v_{k+1}(s) - v_k(s) \mid$가 충분히 작을 때 중지한다. pseudo code는 stopping criterion이 고려된 iterative policy evaluation을 의미한다. 

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/229339632-3d8f3997-d8e4-4117-ad15-ac05bfb9fe07.png" width="80%" height="80%"></center>

<br/>


## 4.3 Policy Improvement
policy evaluation을 통해 임시 policy $\pi$에 대한 $v_\pi$를 얻을 수 있었고, 우리는 이제 특정 state $s$에서, 기존의 임시 $\pi$에서 deterministic하게 결정하던 $a$를 바꿔야 하는지에 대한 여부를 판단할 수 있다. $v_\pi$를 통해 $s$에서 현재 $\pi$가 얼마나 좋은지에 대해 알 수 있었으니, 이제 $s$에서 $a$를 선택한 후, 어떤 $\pi$를 따를지 고려하면 된다. 만약 $(5)$가 기존 $v_\pi$보다 크다면, $s$에서마다 $a$를 선택하는 것이 더 좋은 것으로 판단할 수 있으며, 실제로 그에 따른 새 $\pi$는 기존보다 더 나을 것이다. 

$$ \begin{align*} q_\pi(s,a) &= \mathbb{E}_\pi \left [ R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t=s, A_t=a \right ] \\ &= \sum_{s', r} p(s',r \mid s, a) \left [ r + \gamma v_\pi(s') \right ] \tag{5} \end{align*}$$

<br/>

이러한 사실에 기반하여 policy improvement를 적용할 수 있다. 모든 $s \in \mathcal{S}$에서 $\pi$와 $\pi'$가 어떤 deterministic policy이고 $(6)$의 조건을 만족한다면, $\pi'$는 엄밀히 $\pi$보다 좋거나 같다. 즉, $(7)$처럼 모든 $s \in \mathbb{S}$에서 더 좋거나 같은 expected return를 얻는다는 것이다. 

$$ \begin{align*} q_\pi(s, \pi'(s)) \ge v_\pi(s)\tag{6} \end{align*}$$
$$ \begin{align*} v_{\pi'} \ge v_\pi(s)\tag{7} \end{align*}$$

<br/>

어떤 state에서 $(6)$의 부등식이 성립하면 적어도 하나의 state에서 $(7)$의 부등식이 존재하게 되고, 이는 $\pi$와 $\pi'$($\pi \ne \pi'$) 모두에서 적용된다. $(7)$은 모든 state에서 유지되기에, 결과적으로 $ q_\pi(s, a) \ge v_\pi(s)$라면 $\pi'$가 실제로 $\pi$보다 낫다. 

<br/>

### 4.3.1 Proof of Policy Improvement 
$v_\pi$를 얻을 때 까지, $(7)$의 부등식에서 시작해 $q_\pi$를 계속 확장하여, $(7)$를 재적용시켜 증명 가능하다. 

$$ \begin{align*} v_\pi &\le q_\pi(s, \pi'(s))
\\ &= \mathbb{E}_{\pi'} \left [ R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t =s \right ] 
\\ &\le \mathbb{E}_{\pi'} \left [ R_{t+1} + \gamma q_\pi(S_{t+1}, \pi'(S_{t+1})) \mid S_t =s 
\right ] 
\\ &= \mathbb{E}_{\pi'} \left [ R_{t+1} + \gamma \mathbb{E}_{\pi'} \left [ R_{t+2} + \gamma v_\pi(S_{t+2}) \right ]  \mid S_t =s \right ] 
\\ &= \mathbb{E}_{\pi'} \left [ R_{t+1} + \gamma R_{t+2} + \gamma^2 v_\pi(S_{t+2}) \mid S_t =s \right ] 
\\ &\le \mathbb{E}_{\pi'} \left [ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 v_\pi(S_{t+3}) \mid S_t =s \right ] 
\\ &\dots
\\ &\le \mathbb{E}_{\pi'} \left [ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \gamma^3 R_{t+4} + \cdots \mid S_t =s \right ]
\\ &= v_\pi'(s)  
\tag{8} \end{align*} $$

### 4.3.2 Greedy Policy $\pi'$
모든 states와 모든 action에서 가장 좋은 $q_\pi(s,a)$를 선택하는 greedy policy $\pi'$ 적용도 가능하다. greedy policy는 $v_\pi$에서 best action을 취하는데 이 때, $(7)$에서의 조건을 만족하게 되면 policy는 향상된다. 기존 policy의 value function에서 greedy하게 행동하여, 기존보다 더 좋거나 같은 policy를 만드는 과정을 policy improvement라고 한다. 

$$ \begin{align*} \pi'(s) &= \arg\max_a q_\pi(s,a)
\\ &= \arg\max_a \mathbb{E} \left [ R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t = s, A_t = a \right ]
\\ &= \arg\max_a \sum_{s', r} p(s',r \mid s, a) \left [ r + \gamma v_\pi(s') \right ]  
\tag{9} \end{align*} $$

</br>

만약 new greedy policy $\pi'$가 기존 policy $\pi$보다 같지만 더 좋지 않다면($v_\pi = v_{\pi'}$), $(9)$는 모든 state $s \in \mathcal{S}$에서 $(10)$과 같다. 이는 Bellman optimality equation과 같으므로, $v_{\pi'}$는 $v_\ast$이며 $\pi$와 $\pi'$는 optimal policy가 된다. policy improvement는 기존 policy가 이미 optimal하지 않는 이상 엄밀히 더 좋은 policy를 제공한다. 

$$ \begin{align*} v_{\pi'}(s) &= \max_a q_{\pi'}(s,a)
\\ &= \max_a \mathbb{E} \left [ R_{t+1} + \gamma v_{\pi'}(S_{t+1}) \mid S_t = s, A_t = a \right ]
\\ &= \max_a \sum_{s', r} p(s',r \mid s, a) \left [ r + \gamma v_{\pi'}(s') \right ]  
\tag{10} \end{align*} $$


### 4.3.3 Deterministic case vs. Stochastic case

해당 섹션에서는 deterministic policy case를 고려하였지만, 일반적인 경우에 사용되는 stochastic policy는  $\pi$가 $s$에서 $a$를 하는 probability인 $\pi(a \mid s)$를 갖는다. policy improvement에서 stochastic case의 경우 $q_\pi$는 $(11)$로 표현된다. $(9)$에서 maximal action이 여러개인 경우, stochastic case에는 그 중 하나의 action을 선택하는 것이 아닌 각 maximal action이 선택될 probability distribution이 주어진다. 만약 maximal action이 하나라면, 나머지 submaximal action은 probability가 0일 것이다. 

$$ q_\pi(s, \pi'(s)) = \sum_a \pi'(a\mid s) q_\pi(s,a) \tag {11} $$

<br/>

## 4.4 Policy Iteration

$\pi$에서의 $v_\pi$를 사용하여 더 좋은 $\pi'$를 구할 수 있고, $v_{\pi'}$를 사용하여 전보다 더 좋은데 $\pi''$를 구할 수 있다. 따라 일련의 policy evaluation과 policy improvement의 반복을 통해서 단조롭게 개선되는 value function과 policy를 얻을 수 있다. 이러한 방식을 policy iteration이라고 한다. 

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/229339671-18be8977-50e7-43d5-929b-194483b819fa.png" width="70%" height="70%"></center>

<br/>

finite MDP는 finite 개수의 policy만 가지므로 finite iteration에서 optimal policy와 optimal value function으로 수렴해야 한다. 각 policy evaluation이 이전 policy의 value function으로 시작되어 value function이 policy 간 거의 변경되지 않기 때문에 일반적으로 policy evluation의 수렴 속도가 크게 증가한다. 

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/229339595-ac87d298-8980-4160-abda-a39bea3bf9bb.png" width="80%" height="80%"></center>

<br/>

