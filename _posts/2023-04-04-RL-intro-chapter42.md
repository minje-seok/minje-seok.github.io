---
layout: article
title: Chapter 4. Dynamic Programming (2)
aside:
  toc: true
sidebar:
  nav: layouts
---

> 강화학습의 바이블이라고 불리는 Richard S. Sutton의 Reinforcement Learning: An Introduction을 공부하고 필자가 이해한 내용과 추가 정보들을 예제와 함께 간략하고 쉽게(?) 설명하는 방식으로 정리해봅니다. 용어 같은 경우, 원문 그대로 사용하겠지만 혹시 모를 번역 오류 및 잘못된 설명에 대한 지적과 보충 환영합니다. 

# 4. Dynamic Programming
## 4.5 Value Iteration
policy iteration의 단점은 매 iteration마다, sweep을 진행하는 장기간의 계산인 policy evaluation이 포함된다는 것이다. policy evaluation은 극한에서만 정확히 \\( v_\pi \\)로 수렴하기에 이를 생략하는 방법이 등장하게 된다. policy iteration의 policy evaluation 단계는 수렴에 대한 보장을 잃지 않으면서 여러 방법으로 잘릴 수 있다. 이 때, 단 한번의 sweep(각 state에 대한 한번의 backup) 후에 policy evaluation이 중지되는 방법을 value iteration이라고 한다. 임시 $v_0$에 대해, 수열 $\{v_k \}$는 $v_\ast$의 존재를 보장하는 동일한 조건에서 $v_\ast$로 수렴하게 된다. 

$$ \begin{align*} v_{k+1}(s) 
&= \max_a \mathbb E \left [ R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t=s, A_t=a  \right ] \\ 
&= \max_{a \in \mathcal A(s)} \sum_{s',r}p(s', r|s,a) \left [r +\gamma v_k(s') \right ] \tag{1} \end{align*} $$

### Bellman Optimality Equation
value iteration은 Bellman optimality equation을 update rule로 변환한 것으로도 볼 수 있다. maximum action value를 가져와야한다는 점을 제외하고 value iteration backup은 policy evaluation backup과 동일하다. 왼쪽의 그림은 policy evaluation의 backup diagram을, 오른쪽의 그림은 value iteration의 backup diagram을 나타낸다. 

<center><img src="https://user-images.githubusercontent.com/127359789/229777724-6ee5a390-7267-40f8-8a81-a68d62975263.png" width="70%" height="70%"></center>

<br/>

value iteration 또한 $v_\ast$에 도달하기 위해서는 infinite iteration이 필요하다. 실제 pseudo code에서는, sweep에서 작은 양의 value function의 변화가 생길때마다 termination condition을 만족하게된다. value iteration은 각 sweep에서 한 번의 policy evaluation과 한 번의 policy improvement를 결합한다. 한번의 improvement 사이에 여러 번의 evaluation을 진행하면 더 빠른 수렴이 가능하다. 이러한 모든 알고리즘들은 discounted finite MDP에서 optimal policy로 수렴한다. 


<center><img src="https://user-images.githubusercontent.com/127359789/229780871-b4bfe63a-b9bc-48fc-9406-e5ab4ab26b25.png" width="70%" height="70%"></center>

<br/>

## Asynchronous Dynamic Programming
현재까지 살펴본 DP 방법들은 MDP 전체 state에 대한 sweep을 진행하는 방식이었으나, 만약 state가 매우 크다면 막대한 계산 비용이 요구된다. asynchronous DP 알고리즘은 in-place iterative DP 알고리즘으로 현재 사용 가능한 다른 state-value를 사용하여 순서에 관계없이 backup한다. 따라 일부 state는 다른 state보다 여러번 backup될 수 있으나, 결론적으로 수렴을 위해서는 모든 state-value의 backup이 필요하다.asynchronous DP는 backup이 적용되는 state를 선택하는데 큰 유용성을 제공한다. 

in-place value itreation를 예로들면, $0 \le \gamma \le 1$인 경우 모든 state에서 무한한 횟수로 in-place update가 발생하면 $v_\ast$에 대한 점근적 수렴이 보장된다. 따라 policy evaluation과 value iteration backup을 혼합하면 asynchronous truncated policy iteration을 생성할 수도 있다. 

그러나 sweep을 피하는 과정은 반드시 적은 계산으로 policy를 개선시킬 수 있다는 것은 아니다. 일부 state는 중요도가 떨어져 자주 backup할 필요가 없을 수도 있고, value가 state에서 state로 효율적으로 전파되도록 backup 할 수도 있다. 따라 알고리즘의 진행 속도를 개선하기 위해, backup을 적용할 state를 선택하여 이러한 유연성을 활용할 수 있다. 

asynchronous 알고리즘은 계산과 실시간 상호 작용을 쉽게 혼합할 수 있다. 즉, agent가 MDP를 경험하는 동시에 iterative DP 알고리즘 실행이 가능하다. agent의 경험을 DP 알고리즘이 lbackup을 적용하는 state를 결정하는데 사용할 수 있다. 동시에 DP 알고리즘의 최신 value 및 policy는 agent의 의사 결정에 도움을 줄 수 있다. 이를 통해 DP 알고리즘의  backup을 agent와 가장 관련성이 높은 state에만 집중할 수 있다. 

## Generalized Policy Iteration
