---
layout: article
title: Chapter 5. Monte Carlo Methods (1)
aside:
  toc: true
sidebar:
  nav: layouts
---

> 강화학습의 바이블이라고 불리는 Richard S. Sutton의 Reinforcement Learning: An Introduction을 공부하고 필자가 이해한 내용과 추가 정보들을 예제와 함께 간략하고 쉽게(?) 설명하는 방식으로 정리해봅니다. 용어 같은 경우, 원문 그대로 사용하겠지만 혹시 모를 번역 오류 및 잘못된 설명에 대한 지적과 보충 환영합니다. 

# 5. Monte Carlo Methods
현재 챕터부터는 env에 대해 fully-known하지 않다. Monte Carlo (MC)방법은 env와의 상호작용에서 얻은 sample sequence(state, action, reward)의 실제 experiment만 필요하다. experiment를 통한 학습은 env의 dynamics 없이도 optimal action을 얻을 수 있다. env가 아닌 simulator로 얻은 experiment를 통해서도, DP에서처럼 모든 possible transition에 대한 완전한 probability distribution이 아닌 sample transition만 생성하면 된다. 

간단히, MC 방법은 averaging sample return을 기반으로 RL task를 해결한다. 이해를 위해 episodic task를 고려하여, episode가 종료되면서 value function과 policy가 변경된다. 이는 2장의 stationary bandit에서 각 action에 대해 sample average와 비슷하지만, 이번에는 action 선택에 따라 다음 state와 reward가 변경되어 non-stationary하다. 

non-stationary는 DP에서의 policy iteration (GPI) 아이디어를 적용할 수 있다. 이때는, fully-known MDP에서 value function을 계산했다면, MC는 sample return에서 value function을 학습한다.

<br>

## 5.1 Monte Carlo Prediction

given policy에 대해 state-value function을 학습하기 위한 prediction 문제를 MC 방법으로 고려한다. 먼저 state-value function라는 것은 해당 state로부터 시작한 expected return (expected cumulative future discounted reward)라는 것을 상기하자. 이를 experiment를 사용하여 해당 state에서부터 관찰된 return의 average로 estimate한다. 더 많은 return이 관찰될수록 sample average는 expected value에 수렴하게 된다. 이러한 컨셉이 MC 방법의 기반이 된다. 

<br>

### 5.1.1 First-visit Monte Carlo

episode에서 처음 방문하는 state $s$를 $s$에 대한 first-visit이라고 할 때, first-visit MC는 첫 번째 방문 이후의 average return으로 $v_\pi(s)$를 estimate하는 반면 every-visit MC는 $s$에 대한 모든 방문 이후의 average return을 구한다. 해당 챕터에서는 first-visit MC를 집중적으로 다룬다. 