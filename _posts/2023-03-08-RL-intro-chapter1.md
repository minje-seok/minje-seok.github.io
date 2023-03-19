---
layout: article
title: Chapter 1. The Reinforcement Learning Problem
aside:
  toc: true
sidebar:
  nav: layouts
---

> 강화학습의 바이블이라고 불리는 Richard S. Sutton의 Reinforcement Learning: An Introduction을 공부하고 필자가 이해한 내용과 추가 정보들을 예제와 함께 간략하고 쉽게(?) 설명하는 방식으로 정리해봅니다. 용어 같은 경우, 원문 그대로 사용하겠지만 혹시 모를 번역 오류 및 잘못된 설명에 대한 지적과 보충 환영합니다. 

# 1. The Reinforcement Learning Problem

## 1.1 Reinforcement Learning

<center><img src="https://user-images.githubusercontent.com/127359789/226184814-097d39b6-8010-4ac2-a673-40b693fde7a1.png" width="60%" height="60%"></center>
<center><span style="font-size:80%">https://opentutorials.org/module/4916/28934</span> </center>

머신러닝은 크게 세 가지;지도학습(Supervised), 비지도학습(Unsupervised), 강화학습(Reinforced)로 분류된다. 일반적으로 지도학습은 결과를 예측(Regression, Classification), 비지도 학습은 데이터의 본질적 패턴을 발견(Association, Transform, Clustering)하는 것을 목적으로 한다. 그러나 강화학습은 환경과의 상호작용을 통해 얻은 보상을 통해 행동을 학습하는 것을 목적한다. 행동에 대한 결과와 보상의 경험은 궁극적인 목표를 달성하기 위한 정보를 포함하며, 이를 통해 환경이 자신의 행동에 대해 어떻게 변화하는지 학습할 수 있다. 상호작용이란 강화학습에서 모든 학습 이론의 기반이 되는 기본 아이디어라고 할 수 있다. 

<br/>

### 1.1.1 Characteristics of Reinforcement Learning Problem 

**- Closed-loop이다.** 

Closed-loop란 원하는 상태 유지를 위해 사람의 개입없이 시스템이 자체적으로 조절하는 방식을 의미한다. 강화학습은 본질적으로 사람과의 상호작용이 없을 뿐더러 학습자의 동작 이후의 결과만이 다음 입력에 영향을 미치게 된다. 

**- 학습자에게 직접적인 지침이 주어지지 않는다.** 

학습자는 상호작용 속에서 수치적 보상 신호를 최대화 하기 위해 현재의 환경에서 어떤 행동을 해야할지 배워야 한다. 이 때, 학습자는 trial-and-error 방식 즉, 여러 시행착오를 통해 어떤 행동이 가장 큰 보상을 얻을 수 있는지 발견한다.

**- 보상 신호를 포함한 행동의 결과가 장기간에 걸쳐 나타난다.** 

행동에 대한 보상은 즉각적인 보상뿐만 아니라 다음 상황과 그로 발생하는 모든 후속 보상에 영향을 미칠 수 있다. 추후 고려하겠지만, 이러한 특성으로 인해 학습자는 특정 파라미터 조절을 통해 다음 1-step 뒤의 즉각적인 보상만을 목적하기도, 혹은 n-step 뒤의 장기적인 보상을 목적할 수도 있다. 

<br/>

### 1.1.2 Exploration vs. Exploitation

강화학습에서 발생하는 문제는 exploration(탐색)과 exploitation(착취) 간의 균형이다. 학습자는 본인이 과거에 시도했던 높은 보상을 생성하는 행동을 선호해야 하는데, 이러한 행동을 발견하기 위해서는 이전에 선택하지 않았던 행동을 시도해야한다. 학습자는 이전에 알고있던 보상 정보를 exploit해야 하지만, 더 나은 보상의 행동 선택을 하기 위해 explore 해야하는 딜레마에 빠지게 된다. 

<br/>

### 1.1.3 Characteristics of Reinforcement Learning

강화학습은 불확실한 환경에서도 문제의 목표를 명시적으로 고려한다. 학습자는 전체 환경을 고려할 수 없더라도 명확한 목표를 가지고  동작한다고 가정된다. 가령, 어떤 하위 문제로부터 결과가 도출되었는지 확인하기 위한 연구에서는 환경 전체와 완전하게 상호 작용하며 목표를 추구하는 에이전트 내에서 명확한 자신의 역할을 수행하는 하위 문제가 필요하다. 이미 유용한 결과를 산출한 문제라도 그 하위 문제에 초점을 맞추려면 이를 고립시켜야 하는데, 강화학습은 하위 문제에서도 명시적으로 특정 목표를 위해 동작하므로 이러한 연구에 적합하다. 

<br/>

### 1.1.4 Trends in Reinforcement Learning

강화학습은 엔지니어링 및 과학 분야에서 유익한 상호작용을 거치며 발전해 나가고 있다. 통계, 최적화와 같은 수학적 주제를 넘어 매개변수화된 근사기의 적용과 같은 융합은 여러 패러다임을 열며 모델들의 성능을 향상시켰다. 그러나 과거로부터의 이러한 발전은 각 도메인에 치중하여 많은 특수 절차 및 휴리스틱을 가정했고 일반적인 원칙을 찾지 않아왔다. 최신 AI는 이제 학습과 의사결정의 일반 원칙을 찾고 도메인 지식을 통합하기 위한 노력을 하고 있다. 

<br/>

## 1.2 Examples

인간은 운동, 게임, 일과 같은 어떤 행동을 하던간에 반복적으로 수행하고 경험할수록 수행능력과 판단력이 향상된다. 아래의 예제는 게이머가 게임을 하는 상황을 보여준다. 게이머는 현재의 화면을 보고 상태(환경)과 상/벌(보상)을 관찰한다. 해당 관찰의 내용을 통해 우리의 뇌는 더 높은 보상을 얻을 수 있는 판단을 하게되고 행동으로 옮겨지게 된다. 그로 인해 게임의 상황은 게이머가 수행한 행동으로 인해 변하게 되어 게이머는 다시 바뀐 환경을 관찰하고 다음 행동을 선택하는 과정을 반복하면서 학습한다. 

<center><img src="https://user-images.githubusercontent.com/127359789/224031332-538f3656-899a-4de9-9851-be81fa0cc7d5.jpg" width="60%" height="60%"></center>
<center><span style="font-size:80%">https://opentutorials.org/course/4548/28949</span> </center>


이러한 과정은 모두 능동적인 의사 결정과 학습자의 환경과의 상호 작용을 포함하며, 학습자는 환경에 대한 불확실성에도 불구하고 목표를 달성하려고 한다. 주의해야할 것은 학습자와 환경이라는 단어에 의해 고정관념을 가진 경우가 있는데, 학습자는 반드시 전체 로봇이나 유기체가 아닌 그 하위 집합일수도, 환경은 외부가 아닌 내부일 수도 있는 추상적인 개념이므로 주의해야 한다. 

목표를 향한 올바른 행동 선택에는 간접적이고 지연된 결과를 고려해야 하고 환경의 미래 상태에 영향을 미치므로 이는 완전히 예측할 수 없기 때문에 많은 정보 획득을 위해서는 잦은 환경 모니터링과 그에 따른 적절한 대응이 필요하다. 결과적으로, 학습자는  행동 선택과 환경과의 반복적 상호작용으로 얻은 경험을 사용하여 시간이 지남에 따라 학습하게 된다. 학습자가 관찰할 수 있는 것을 기반으로 목표를 향한 진행 상황을 판단할 수 있다는 점에서 명시적인 목표가 포함된다. 

<br/>

## 1.3 Elements of Reinforcement Learning

기본적인 강화학습에서 사용되는 네가지 요소에 대해 개념적으로 짚어본다. 이제부터는 이해를 돕기 위해 학습자라고 불렀던 학습 주체를 agent로, 환경을 env로 표현하겠다. state는 일단 agent 시점에서 관찰 가능한 env 혹은 situation로 이해하자. 강화학습의 최종 목표는 주어진 요소들을 사용하여 action을 수행했을 때의 장기적 관점에서의 보상을 최대화할 수 있도록 agent를 학습시키는 것이다. 

<br/> 

**- policy: agent's behavior function; mapping from state to action.**

agent가 행동하는 방식을 정의하며, state에서 취해야 할 action으로의 매핑이다. policy는 단순한 function이나 table일수도 혹은 매우 큰 계산이 요구되는 black-box function일 수도 있다. 일반적으로 stochastic $(a=\pi(s))$하지만 deterministic $(\pi(a \middle s)=P \left [A_t=a \middle S_t =s \right ] )$한 경우도 존재한다. 

**- reward: immediate(short-term) scalar feedback signal.**

강화학습 문제의 목표를 구성하며 즉각적인 의미에서의 1-step 보상을 정의한다. 각 time-step에서 env는 policy에 따라 action을 수행한 agent에게 scalar number인 reward를 보내어 해당 action의 바람직함 판단을 가능하게 한다. reward는 일반적으로 수행한 action과 state에 따른 stochastic function일 수 있다. 

**- value function: expected cumulative(long-term) reward from state.**

강화학습 문제에서 최대화 하려는 궁극적인 목표이며 장기적인 의미에서의 가치를 정의한다. agent가 해당 state에서 시작하여 미래에 누적될 것으로 예상되는 총 expected cumulative reward를 의미한다. Bellman equation과 함께 언급하겠지만 value function은 일련의 reward summation의 expectation으로 표현되며 시작되는 state가 다르면 값이 달라질 수 있다. 결국 우리는 action의 단기적관점의 reward 보다는 장기적 관점에서의 value function을 통해 가치판단을 하게 된다. 

**- model: duplication of env which generate the next state and reward.**

실제 env의 동작을 모방하거나 일반적으로 환경이 동작하는 방식에 대한 추론을 가능하게 한다. 예를 들어, state와 action이 주어지면 model을 통해 결과로 나타나는 next state와 reward를 예측할 수 있다. model은 추후 model-based method에서 언급될 planning에 사용되며 실제 상호작용 없이도 미래 상황을 고려하여 action을 선택할 수 있다. 

<br/>

<center><img src="https://user-images.githubusercontent.com/127359789/224031342-da896ded-e6d4-4c4e-8dd6-3e9fac93f856.jpg" width="60%" height="60%"></center>
<center><span style="font-size:80%">https://opentutorials.org/course/4548/28949</span> </center>

위에서 보았던 예제를 새롭게 정의된 단어들로 표현하면 다음과 같이 쓸 수 있다; 게임 → 환경(env), 게이머 → 학습자(agent), 게임화면 → 상태(state), 게이머의 조작 → 행동(action), 상과 벌 → 보상(reward), 게이머의 판단력 → 정책(policy). agent는 env로부터 state와 reward를 얻게되고 policy는 state에 대한 매핑으로 action을 선택한다. 수행한 action에 따라 env가 바뀌게 되고 또 다시 state와 reward를 관찰하는 과정을 반복하는 과정에서 policy가 학습하고 value function을 최대화 하는 action을 선택하게 된다. 

<br/>


## 1.4 Limitations and Scope

최대화하려는 요소인 value function을 사용해야만 문제를 해결할 수 있는 것은 아니다. 강화학습 문제에 이를 사용하지 않는 genetic programming, genetic algorithms, simulated annealing와 같은 evolutionary 방법들도 존재한다. 이러한 방식들 또한 env의 특징마다의 강점이 존재하나 강화학습 문제의 유용한 구조를 일반적으로 무시하는 문제로 인해 강화학습 방법에는 포함시키지 않는다. 

그러나 value function을 사용하지는 않지만 evolutionary 방법에는 속하지 않는, agent가 또 다른 estimate를 생성하는 policy gradient method는 포함된다. optimization은 강화학습과 동일하게 reward를 최대화하는 목표를 가지고 있으나, 강화학습에서는 agent가 받는 보상의 양을 늘리려고 매 env마다 노력하고 최대값이 존재하더라도 달성하지 못할 수도 있다는 차이점이 존재한다. 즉, optimization은 optimality와 같다고 볼 수 없다.  

<br/>

## 1.5 Summary

강화학습은 모범적인 감독이나 완전한 model에 의존하지 않고, agent가 env와의 직접적인 상호작용을 통해 학습하며 expected cumulative reward를 최대화 하는 것을 목적으로 하는 머신러닝 방법이다. 강화학습은 state, action, reward 측면에서 agent와 env 간의 상호작용을 정의하는 프레임워크를 사용하여 인공지능 문제의 필수 기능을 나타낸다. 이는 원인과 결과에 대한 감각, 불확실성과 비결정론에 대한 감각, 명확한 목표 존재의 특징을 내포한다. 앞으로 더 구체적으로 공부하게 될 value function은 강화학습 방법의 핵심으로, policy 학습을 위해 매우 중요하게 사용될 것이다.

<br/>


