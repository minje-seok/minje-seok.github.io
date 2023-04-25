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
policy iteration의 단점은 매 iteration마다, sweep을 진행하는 장기간의 계산인 policy evaluation이 포함된다는 것이다. policy evaluation은 극한에서만 정확히 $v_\pi$로 수렴하기에 이를 생략하는 방법이 등장하게 된다. policy iteration의 policy evaluation 단계는 수렴에 대한 보장을 잃지 않으면서 여러 방법으로 truncated 될 수 있다. 이 때, 단 한번의 sweep(각 state에 대한 한번의 backup) 후에 policy evaluation이 중지되는 방법을 value iteration이라고 한다. 임시 $v_0$에 대해, 수열 $\{v_k \}$는 $v_\ast$의 존재를 보장하는 동일한 조건에서 $v_\ast$로 수렴하게 된다. 

$$ \begin{align*} v_{k+1}(s) 
&= \max_a \mathbb E \left [ R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t=s, A_t=a  \right ] \\ 
&= \max_{a \in \mathcal A(s)} \sum_{s',r}p(s', r|s,a) \left [r +\gamma v_k(s') \right ] \tag{1} \end{align*} $$

### 4.5.1 Bellman Optimality Equation
value iteration은 Bellman optimality equation을 update rule로 변환한 것으로도 볼 수 있다. maximum action value를 가져와야한다는 점을 제외하고 value iteration backup은 policy evaluation backup과 동일하다. 왼쪽의 그림은 policy evaluation의 backup diagram을, 오른쪽의 그림은 value iteration의 backup diagram을 나타낸다. 

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/229777724-6ee5a390-7267-40f8-8a81-a68d62975263.png" width="70%" height="70%"></center>

<br/>

value iteration 또한 $v_\ast$에 도달하기 위해서는 infinite iteration이 필요하다. 실제 pseudo code에서는, sweep에서 작은 양의 value function의 변화가 생길때마다 termination condition을 만족하게된다. value iteration은 각 sweep에서 한 번의 policy evaluation과 한 번의 policy improvement를 결합한다. 한번의 improvement 사이에 여러 번의 evaluation을 진행하면 더 빠른 수렴이 가능하다. 이러한 모든 알고리즘들은 discounted finite MDP에서 optimal policy로 수렴한다. 

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/229780871-b4bfe63a-b9bc-48fc-9406-e5ab4ab26b25.png" width="70%" height="70%"></center>

<br/>

## 4.6 Asynchronous Dynamic Programming
현재까지 살펴본 DP 방법들은 MDP 전체 state에 대한 sweep을 진행하는 방식이었으나, 만약 state가 매우 크다면 막대한 계산 비용이 요구된다. asynchronous DP 알고리즘은 in-place iterative DP 알고리즘으로 현재 사용 가능한 다른 state-value를 사용하여 순서에 관계없이 backup한다. 따라 일부 state는 다른 state보다 여러번 backup될 수 있으나, 결론적으로 수렴을 위해서는 모든 state-value의 backup이 필요하다.asynchronous DP는 backup이 적용되는 state를 선택하는데 큰 유용성을 제공한다. 

in-place value itreation를 예로들면, $0 \le \gamma \le 1$인 경우 모든 state에서 무한한 횟수로 in-place update가 발생하면 $v_\ast$에 대한 점근적 수렴이 보장된다. 따라 policy evaluation과 value iteration backup을 혼합하면 asynchronous truncated policy iteration을 생성할 수도 있다. 

그러나 sweep을 피하는 과정은 반드시 적은 계산으로 policy를 개선시킬 수 있다는 것은 아니다. 일부 state는 중요도가 떨어져 자주 backup할 필요가 없을 수도 있고, value가 state에서 state로 효율적으로 전파되도록 backup 할 수도 있다. 따라 알고리즘의 진행 속도를 개선하기 위해, backup을 적용할 state를 선택하여 이러한 유연성을 활용할 수 있다. 

asynchronous 알고리즘은 계산과 실시간 상호 작용을 쉽게 혼합할 수 있다. 즉, agent가 MDP를 경험하는 동시에 iterative DP 알고리즘 실행이 가능하다. agent의 경험을 DP 알고리즘이 lbackup을 적용하는 state를 결정하는데 사용할 수 있다. 동시에 DP 알고리즘의 최신 value 및 policy는 agent의 의사 결정에 도움을 줄 수 있다. 이를 통해 DP 알고리즘의  backup을 agent와 가장 관련성이 높은 state에만 집중할 수 있다. 

<br/>

## 4.7 Generalized Policy Iteration

generalized policy iteration (GPI)는 policy evaluation과 policy improvement가 상호작용하는 일반적인 개념을 나타낸다. 거의 모든 RL 방법은 GPI로 설명이 가능하며, identifiable policy와 value function를 가진다. 아래 그림과 같이 policy는 value function에 연관되어 개선되고, value function은 항상 policy에 따라 계산된다. 

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/233935350-7230cc1a-5719-408f-a0b9-d2db49623d95.png" width="70%" height="70%"></center>


GPI의 evalution과 improvement 프로세스는 경쟁과 협력의 측면으로 볼 수 있다. policy를  greedy하게 만드는 것은 변경된 policy에 대해 value function을 부정확하게 만들고,  value function을 policy와 일관되게 만들면 일반적으로 해당 policy가 더 이상 greedy하지 않게 된다. 그러나 장기적으로 이 두 프로세스는 optimal policy와 optimal value fnction이라는 single joint solution을 찾기 위해 상호작용한다. 


<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/233935592-fbd59779-1a4a-4454-8b84-03458e87f5a6.png" width="70%" height="70%"></center>

다이어그램의 각 프로세스는 두 가지 목표 중 하나에 대한 solution을 나타내는 라인 중 하나를 향해 value funtion 또는 policy를 추진한다. 한 목표를 향해 움직이면 다른 목표에서 또한 움직임이 약간 발생하면서 공동 프로세스는 optimal에 더 가까워 진다. 다이어그램의 화살표는 두 목표 중 하나를 완전히 달성할 때까지 진행하는 점에서  policy iteration이라 볼 수 있다. 두 경우 모두 직접 달성하려하지 않아도, 두 프로세스가 함께 optimal을 달성한다. 

<br/>


## 4.8 Efficiency of Dynamic Programming

DP는 매우 큰 문제에는 실용적이지 않지만, MDP를 해결하는 다른 방법과 비교할 때 실제로 매우 효율적이다. DP는 n개의 state와 m개의 action에서 n과 m의 polynomial function보다 적은 연산을 필요로 한다. 총 determinisitc policy의 수가 n $\times$ m이라고 해도 polynomial time에 최적의 정책을 찾는 것을 보장한다. 이러한 의미에서 DP는 각 policy에 대한 철저한 검사가 필요한 policy space에서 직접 검색하는 것보다 exponentially하게 빠르다. 

DP는 종종 state의 개수에 따른 차원의 저주로 인해, 적용 가능성이 제한적인 것으로 생각되나 이는 문제 자체의 고유한 어려움을 뜻한다. 실제로 DP는 직접 검색 및 linear programming과 같은 방법보다 큰 state space 처리에 비교적 더 적합하다. 일반적으로 이론적으로 최악의 실행시간보다 훨씬 빠르게 수렴한다. 

이렇게 state space가 큰 경우는 asynchronous DP가 선호된다. synchronous의 한 sweep은 모든 state에 대한 계산과 메모리가 필요는 비실용적이고, optimal solution의 trajectory를 따라 발생하는 state가 공통적으로 sweep이 반복되기 때문에 문제의 잠재적 해결이 가능하다. 이러한 방식은 synchronous보다 훨씬 빠르게 optimal policy를 찾을 수 있다. 

<br/>

## 4.9 Summary 

해당 챕터에서는 finite MDP의 해결과 관련된 DP 기본 아이디어를 언급했다. policy evaluation는 given policy에 대한 value function의 반복적인 계산을, policy improvement는 해당 policy에 대한 value function이 주어진 improved policy의 계산을 말한다. 이 두 프로세스를 합치면 policy iteration과 value iteration으로 확장할 수 있었다. 이를 통해 fully-known finite MDP에 대한 optimal policy 및 value function을 안정적으로 계산하는데 사용할 수 있다. 기존 DP 방법은 전체 state에 대해 sweep하여 value function이 더이상 변경되지 않을 때까지 full backup을 수행하면, 해당 Bellman equation을 충족하는 수렴이 발생했다고 볼 수 있다. 

모든 RL 방법은 GPI의 approximate policy와 approximate value function을 중심으로 상호작용하는 두 개의 프로세스에 대한 아이디어를 사용한다. 하나는 given policy에 따라 value function를 update하고, 다른 하나는 given value function에 따라 policy를  update한다. 각 프로세스가 다른 프로세스의 기반을 변경하지만 결국 공동 solution을 찾기 위해 함께 작동하여 optimal policy 및 optimal value function을 찾는다. 이는 다른 경우는 제외하고 해당 챕터에서 언급한 고전적인 DP에서 GPI의 수렴을 입증한다. full backup의 과도한 계산 비용을 피하기 위해, 임의의 순서로 state를 backup하는 in-place iterative 방식의 asynchronous DP 방법도 존재한다. 이는 세분화된 형태의  GPI로도 볼 수 있다. 

DP는 successor state의 estimate를 기반으로 estimate를 update하는 bootstrapping을 수행한다. 많은 RL은 DP가 요구하는 완벽한 env model이 필요하지 않은 경우에도 bootstrapping을 수행한다. 다음 챕터에서는 model과 bootstrapping이 필요하지 않은 RL 방법들을 살펴본다. 그 다음 챕터에서는 model이 필요없지만, bootstrapping을 수행하는 방법을 알아볼 예정이다. 

<br/>
