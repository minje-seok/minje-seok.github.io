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
policy iteration의 단점은 매 iteration마다, sweep을 진행하는 장기간의 계산인 policy evaluation이 포함된다는 것이다. policy evaluation은 극한에서만 정확히 $v_\pi$로 수렴하기에 이를 생략하는 방법이 등장하게 된다. policy iteration의 policy evaluation 단계는 수렴 보장을 잃지 않으면서 여러 방법으로 잘릴 수 있다. 여기서 단 한번의 sweep(각 state에 대한 한번의 backup) 후에 policy evaluation이 중지되는 방법을 value iteration이라고 한다. 임시 $v_0$에 대해, 수열 $\{v_k \}$는 $v_\ast$의 존재를 보장하는 동일한 조건에서 $v_\ast$로 수렴하게 된다. 

$$ \begin{align*} v_{k+1}(s) 
&= \max_a \mathbb E \left [ R_{t+1} + \gamma v_k(S_{t+1}) \mid S_t=s, A_t=a  \right ] \\ 
&= \max_{a \in \mathcal A(s)} \sum_{s',r}p(s', r|s,a) \left [r +\gamma v_k(s') \right ] \tag{1} \end{align*} $$

