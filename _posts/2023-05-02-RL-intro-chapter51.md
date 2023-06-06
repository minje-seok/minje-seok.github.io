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

given policy에 대해 state-value function을 학습하기 위한 prediction 문제를 MC 방법으로 고려한다. 먼저 state-value function라는 것은 해당 state로부터 시작한 expected return (expected cumulative future discounted reward)라는 것을 상기하자. experiment를 통해 해당 state에서부터 관찰된 return의 average로 estimate하게 되면, 더 많은 return이 관찰될수록 sample average는 expected value에 수렴하게 된다. 이러한 컨셉이 MC 방법의 기반이 된다. 

<br>

### 5.1.1 First-visit MC vs. Every-visit MC

episode에서 처음 방문하는 state $s$를 고려하는 first-visit MC는 각 $s$의 첫 번째 방문의 return만으로 $v_\pi(s)$를 estimate하는 반면, every-visit MC는 각 $s$에 대한 모든 방문 이후의 average return으로 esitmate한다. every-visit MC는 function approximation과 eligibility traces로 확장되어 추후 챕터에서 다루고, 해당 챕터에서는 더 많은 연구가 진행된 first-visit MC에 집중한다. 

<br>

<center><img src="https://github.com/kitian616/jekyll-TeXt-theme/assets/127359789/e69718c5-3b01-4add-a1ba-5019d21facc7" width="70%" height="70%"></center>


first-visit MC에서 arbitary state-value function $V$는 $s$로의 visit 횟수가 infinity로 가면 $v_\pi(s)$로 수렴한다. each return은 finite variance를 가지는 $v_\pi(s)$의 independent, identically distributed estimate이다. 대수의 법칙에 따라, 이러한 estimate의 average sequence는 expected value로 수렴한다. $n$이 average된 returns의 개수일 때, 각 average는 unbiased estimate이며, error의 standard deviation은 $1/\sqrt{n}$로 감소한다.

<br>  

### 5.1.2 Blackjack Example

blackjack은 보유한 카드들로 숫자의 합이 21이 넘지 않는 가장 큰 수를 만드는 것이 목적이다. 모든 face 카드는 10으로, ace 카드는 1 혹은 11로 계산할 수 있다. 플레이어는 딜러와 독립적으로 경쟁하는 버전을 고려한다. 

1. 게임 시작과 동시에, 딜러와 플레이어에게 2장씩 카드를 제공한다. 
2. 만약 플레이어가 시작부터 21을 가지게 되면(ace + 10) ($natural$), 딜러 또한 $natural$이 아닌 이상, 무조건 플레이어가 승리한다. 
3. 플레이어가 $natural$이 아니라면, 멈추거나($sticks$) 21을 초과할때까지 ($goes \ bust$) 1개씩 추가 카드를 요청($hits$)할 수 있다. 
4. 만약 $bust$된다면, 플레이어가 지고; $sticks$한다면 딜러의 선택 차례로 넘어간다.
5. 딜러는 다음의 고정된 전략에 따른다. 숫자의 합이 17 이상이면 $sticks$, 그렇지 않다면 $hits$한다. 만약 딜러가 $bust$되면 플레이어가 승리한다. 
6. 최종적으로 양족다 $bust$되지 않는다면, 21에 더 가까운 쪽이 승리한다. (win, lose, draw)

<br>

blackjack은 매 게임이 episode인, episodic finite MDP라고 볼 수 있다. reward는 win, lose, draw에 따라 $+1, -1, 0$로 각각 주어진다. 게임 중간에 reward가 주어지지는 않기에, 마지막 reward가 곧 return을 의미한다. state는 플레이어의 카드와 딜러가 보여주는 카드이고, 플레이어의 action은 $hit$ 또는 $stick$이다. 

만약 플레이어가 ace를 들고있을 때, 그를 11로 취급해도 $bust$되지 않는다면 $usable$라 하며 무조건 11로 계산된다. 따라서 플레이어는 현재 자신의 합계(12-21), 딜러가 보여주는 카드(ace-10), $usable$ ace 보유 여부를 기반으로 결정을 내리게 되며 이는 총 200가지의 state가 된다. 해당 blackjack 문제에서는 same state가 episode에서 절대 재반복되지 않으므로 first-visit과 every-visit MC 방법에 차이가 없다. 

<br>

<center><img src="https://github.com/kitian616/jekyll-TeXt-theme/assets/127359789/a2e6122b-d7d1-4f88-8852-0422ed816559" width="70%" height="70%"></center>

위 그림은 플레이어의 숫자 합이 20 또는 21일 때는 $stick$, 아니면 $hit$하는 policy를 고려했을 때의 state-value function를 보여준다. 10,000 게임 이후, $usable$ ace는 자주 등장하지 않기 때문에, estimate가 덜 확실하고 덜 규칙적인 것을 볼 수 있다. 500,000 게임 이후, value function은 어떤 상황에서든지 잘 approximate된 것을 확인 가능하다. 

<br>

### 5.1.3. Comparsion with DP and backup diagram of MC

env의 완벽한 dynamics를 알고 있더라도, value function 계산에 DP 방법을 적용하는 것은 모든 event에 대한 expected reward와 transition probability를 요구할 뿐만 아니라, 이에 대한 계산은 복잡하고 오류를 유발하므로 매우 어렵다. 그러나 MC 방법은 적용이 쉬우므로, 만약 env의 dynamics를 알더라도 sample episode로 동작하는 것이 훨씬 이점이 많다. 

<br>

<center><img src="https://github.com/kitian616/jekyll-TeXt-theme/assets/127359789/2aae04b0-218a-4b85-bd1c-e6c06c93e43f" width="70%" height="70%"></center>

위 그림에서는 MC 방법에서 $v_\pi$를 추정하기 위해, root는 state node, 그 아래는 single episode 동안의 entire trajectory of ransition으로 구성된다. DP 방법에서는 one-step transition을 보여주었지만, MC에서는 episode의 끝까지를 보여준다. 이 때, 중요한 점은 각 state의 estimate가 independent하다는 것이고, 결과적으로 MC는 DP에서 처럼 bootstrap하지 않는다. 특히, 각 independent state에 대한 value estimate 과정에서 다른 모든 state를 무시하고 해당 state에서 return되는 평균만을 계산하기에 experience를 통해 학습이 가능하다. 

<br>

## 5.2. Monte Carlo Estimation of Action Values

만약 DP에서처럼 model이 존재한다면 state-value만 사용하면, one-step 뒤의 reward와 next state의 조합을 통해 어떤 action이 좋은지 알 수 있었다. 그러나 model이 없다면, state-value로는 명시적인 action value의 estimate로 policy를 생성 하기 부족하므로 MC로 $q_\ast$를 추정하고자 한다. 우리는 action value를 위해 policy evaluation 고려한다.   

state-action pair $s,a$는 state $s$에서 action $a$를 수행한 episode라고 할 수 있다. every-visit MC는 episode 내 방문한 모든 $s,a$의 return을 평균하고, first-visit MC는 episode에서 첫번째로 방문한 $s,a$의 return으로 estimate한다. 이러한 방법은 state-action pair가 infinity로 갈수록, quadratically하게 수렴하게 된다. 

<br>

### 5.2.1 Exploring Starts 
그러나 방문되지 않는 많은 state-action pair가 생길 수 있다는 maintaining exploration 문제가 존재한다. 만약 $\pi$가 deterministic policy라면, 특정 state에서 동일한 action만을 선택할 수도 있다. 따라 continual exploration을 강제하는 방식 중 하나는 episode의 start를 state-action pair를 지정하는 것이다. 이는 모든 state-action pair의 방문을 보장해주게 된다.exploring starts는 간간히 유용하지만, env와 직접적으로 상호작용하는 경우에는 특히 적용이 어렵다. 일반적으로, 이에 대한 대안으로 non-zero stochastic policy를 사용한다. 

<br>

## 5.3. Monte Carlo Control
