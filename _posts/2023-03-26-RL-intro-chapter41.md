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
dynamic programming(DP)는 MDP와 같은 완벽한 env의 model이 주어졌을 때, optimal policy를 찾을 수 있는 알고리즘들을 의미한다. 이전 챕터에서 언급했듯 막대한 계산 비용 때문에 강화학습에서는 적용이 제한적이지만, 완벽하지 않은 env에서 계산을 줄인 강화학습 문제 풀이를 위한 방법들은 결국 DP와 동일한 효과를 달성하려는 시도로 볼 수 있다. DP는 continuous state space와 continuous action space를 quantize한 뒤, finite-state DP 방법을 적용하여 일반적인 approximate solution을 얻는 것도 가능하다. 

이전 챕터에서는 강화학습 문제를 풀기 위해서 Bellman optimality equation을 만족하는 optimal value function $v_\ast$ 또는 $q_\ast$을 사용하여 optimal policy를 찾을 수 있다고 했다. DP에서는 원하는 value function에 대한 approximation을 개선하기 위해 $(1),(2)$와 같이 Bellman equation을 전환한 update rule을 적용한다. 지금부터는 env가 $s \in \mathcal{S}, a \in \mathcal{A}(s), r \in \mathcal {R}, s' \in \mathcal {S^+}$ ($\mathcal {S^+}$는 episodic task에서 $\mathcal{S}$ + terminal state)가 모두 finite한 finite MDP이고, dynamics는 $p(s', r \mid s,a)$ probability의 집합으로 주어진다고 가정한다. 

$$ \begin{align*} v_\ast(s) 
&= \max_a \mathbb E \left [ R_{t+1} + \gamma v_\ast(S_{t+1}) \mid S_t=s, A_t=a  \right ] \\ 
&= \max_{a \in \mathcal A(s)} \sum_{s',r}p(s', r|s,a) \left [r +\gamma v_\ast(s') \right ] \tag{1} \end{align*} $$

$$ \begin{align*} q_\ast(s,a) &= \max_a \mathbb E \left [ R_{t+1} + \gamma v_\ast(S_{t+1}) \mid S_t=s, A_t=a  \right ] \\ 
&= \sum_{s',r}p(s', r|s,a) \left [ r + \gamma \max_{a'} q_\ast(s',a') \right ] \tag{2} \end{align*} $$

<br/>

## Policy Evaluation
### Prediction Problem
우선 임시 policy $\pi$를 따르는 state-value function $v_\pi$를 계산하고자 한다. 우리는 이를 prediction problem이라고 부른다. 
이전 챕터에서는 Bellman equation에 기반하여 value function을 $(3)$과 같이 계산했다. $v_\pi$의 존재와 고유성은 $\gamma < 1$ 혹은 모든 state에 대해 termination이 policy $\pi$ 하에서 약속되는 한 보장된다. 

$$ \begin{align*} v_\pi(s) &= \mathbb{E}_\pi \left [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+2} + \cdots \mid S_t = s\right ] \\
&= \mathbb{E}_\pi \left [R_{t+1} + \gamma v_\pi(S_{t+1})\mid S_t = s\right ] \\
&= \sum_a \pi(s \mid a) \sum_{s',r}p(s', r|s,a) \left [r +\gamma v_\pi(s') \right ] \tag{3} \end{align*} $$

<br/>

### Iterative Policy Evaluation
만약 env의 dynamics가 완벽하게 알려져있다면, $(3)$은 simutaneous linear equation $v_\pi(s)$의 system으로 볼 수 있다. 이러한 경우 initial value function $v_0$를 임의로 선택되어 초기화하고(*보통 terminal state를 제외하고 0으로*), 각 next value function approximation은 $(4)$의 Bellman equation을 update rule로 iterative하게 사용하여 solution을 얻는다. $v_\pi$에 대한 Bellman equation이 $v_k = v_\pi$ 경우에도 동등하게 보장되기 때문에 이러한 update rule이 적용 가능하다. 실제로 $v_0, v_1, v_2, \ldots$의 sequence $v_k$는 일반적으로 $v_\pi$의 존재를 보장하는 동일 조건 하에서 $k \rightarrow \infty$일 때 $v_\pi$로 수렴한다. 이러한 알고리즘을 iterative policy evaluation이라고 한다. 

$$ \begin{align*} v_{k+1}(s) &= \mathbb{E}_\pi \left [R_{t+1} + \gamma v_k(S_{t+1})\mid S_t = s\right ] \\
&= \sum_a \pi(s \mid a) \sum_{s',r}p(s', r|s,a) \left [r +\gamma v_k(s') \right ] \tag{4} \end{align*} $$

<br/>

### Full Backup
$v_k$로부터 successive approximation $v_{k+1}$을 생성하기 위해서 iterative policy evaluation는 다음의 과정을 수행한다. 각 state $s$의 new value를 구하기 위해 현재 evaluate되는 policy에서 가능한 모든 one-step transition에서의 successor states $s'$의 old value와 expected immediate reward의 합으로 교체한다. 이러한 과정을 full backup이라고 한다. state 또는 state-action pair이 backup되는지 여부와 successor state의 estimated value가 결합되는 방식에 따라 여러 full backup이 존재한다. 

iterative policy evaluation의 각 iteration은 next approximate value function $v_{k+1}$을 생성하기 위해 모든 state value $v_k$를 backup한다. DP 알고리즘에서 수행되는 모든 backup은 sample의 next state가 아니라 가능한 모든 next state를 기반으로 하기 때문에 full backup이라고 한다. 

<br/>

## In-Place Iterative Policy Evaluation

상식적으로 full bakcup을 진행하기 위해서는 old value와 new value를 저장할 두 개의 array를 필요로 하지만 하나의 array만을 사용하여 'in-place'하게 update할 수도 있다. 이렇게 되면 next state가 backup되는 순서에 따라, state의 old value가 아닌 new value가 사용되게 되지만 $v_\pi$로 수렴한다. in-place 알고리즘에서 state가 backup되는 순서는 수렴 속도에 상당한 영향을 미치고, 2-array보다 빠르게 수렴한다. 일반적으로 DP 알고리즘은 in-place를 염두에 둔다. 

<br/>

### Termination of Algorithm

iterative policy evaluation는 현실적으로 수렴이 이루어지기 전에 멈춰야 하는 constraint가 존재한다. 일반적으로는 각 sweep 이후 $\max_{s \in \mathcal{S}} \mid v_{k+1}(s) - v_k(s) \mid$가 충분히 작을 때 중지한다. 위 pseudo code는 stopping criterion이 고려된 iterative policy evaluation을 의미한다. 
