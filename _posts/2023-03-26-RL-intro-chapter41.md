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

일반적으로 강화학습 문제를 풀기 위해서는 value function을 사용하여 좋은 policy를 측정한다. DP에서는 원하는 value function을 개선하기 위한 $(1),(2)$와 같이 Bellman optimality equation을 만족하는 $v_\ast$ 또는 $q_\ast$를 찾는 update rule을 적용하여 optimal policy를 얻는다. 

$$ \begin{align*} v_\ast(s) 
&= \max_a \mathbb E \left [ R_{t+1} + \gamma v_\ast(S_{t+1}) \mid S_t=s, A_t=a  \right ] \\ 
&= \max_{a \in \mathcal A(s)} \sum_{s',r}p(s', r|s,a) \left [r +\gamma v_\ast(s') \right ] \tag{1} \end{align*} $$

$$ \begin{align*} q_\ast(s,a) &= \max_a \mathbb E \left [ R_{t+1} + \gamma v_\ast(S_{t+1}) \mid S_t=s, A_t=a  \right ] \\ 
&= \sum_{s',r}p(s', r|s,a) \left [ r + \gamma \max_{a'} q_\ast(s',a') \right ] \tag{2} \end{align*} $$

<br/>

## Policy Evaluation
DP의 policy evaluation에서는 임시 policy $\pi$를 따르는 state-value function $v_\pi$를 Bellman equation에 기반하여 $(3)$과 같이 계산한다.

$$ \begin{align*} v_\pi(s) &= \mathbb{E}_\pi \left [R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+2} + \cdots \mid S_t = s\right ] \\
&= \mathbb{E}_\pi \left [R_{t+1} + \gamma v_\pi(S_{t+1})\mid S_t = s\right ] \\
&= \sum_a \pi(s \mid a) \sum_{s',r}p(s', r|s,a) \left [r +\gamma v_\pi(s') \right ] \tag{3} \end{align*} $$

<br/>

만약 env의 dynamics가 완벽하게 알려져있다면, $(3)$은 simutaneous linear equation $v_\pi(s), s \in \mathcal S$의 system으로 볼 수 있다. 이러한 경우 initial value function $v_0$은 임의로 선택되어 초기화하고, 각 다음 value function approximation은 $(4)$의 update rule을 iterative하게 사용하여 solution을 얻는다. 이를 iterative policy evaluation이라고 한다. 

$$ \begin{align*} v_{k+1}(s) &= \mathbb{E}_\pi \left [R_{t+1} + \gamma v_\pi(S_{t+1})\mid S_t = s\right ] \\
&= \sum_a \pi(s \mid a) \sum_{s',r}p(s', r|s,a) \left [r +\gamma v_\pi(s') \right ] \tag{4} \end{align*} $$


<br/>

update rule $(4)$를 현재 time step의 모든 state에 대해 적용하여, $v_{k+1}$을 $v_k$와 expected immediate reward로 생성하는 full backup을 진행한다. iterative policy evaluation의 각 iteration은 새로운 approximate value function $v_{k+1}$를 생성하기 위해 모든 state value function을 한 번 back up 한다. state 혹은 state-action pair
DP에서 수행되는 backup은 모든 가능한 next state를 기반으로 하기 때문에 full backup이라고 부른다. 
