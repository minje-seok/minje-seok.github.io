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
3. 플레이어가 $natural$이 아니라면, 멈추거나($stick$) 21을 초과할때까지 ($goes \ bust$) 1개씩 추가 카드를 요청($hit$)할 수 있다. 
4. 만약 $bust$된다면, 플레이어가 지고; $stick$한다면 딜러의 선택 차례로 넘어간다.
5. 딜러는 다음의 고정된 전략에 따른다. 숫자의 합이 17 이상이면 $stick$, 그렇지 않다면 $hit$한다. 만약 딜러가 $bust$되면 플레이어가 승리한다. 
6. 최종적으로 양족다 $bust$되지 않는다면, 21에 더 가까운 쪽이 승리한다. (win, lose, draw)

<br>

blackjack은 매 게임이 episode인, episodic finite MDP라고 볼 수 있다. reward는 win, lose, draw에 따라 $+1, -1, 0$로 각각 주어진다. 게임 중간에 reward가 주어지지는 않기에, 마지막 reward가 곧 return을 의미한다. state는 플레이어의 카드와 딜러가 보여주는 카드이고, 플레이어의 action은 $hit$ 또는 $stick$이다. 

만약 플레이어가 ace를 들고있을 때, 그를 11로 취급해도 $bust$되지 않는다면 $usable$라 하며 무조건 11로 계산된다. 따라서 플레이어는 현재 자신의 합계(12-21), 딜러가 보여주는 카드(ace-10), $usable$ ace 보유 여부를 기반으로 결정을 내리게 되며 이는 총 200가지의 state가 된다. 해당 blackjack 문제에서는 same state가 episode에서 절대 재반복되지 않으므로 first-visit과 every-visit MC 방법에 차이가 없다. 

<br>

<center><img src="https://github.com/kitian616/jekyll-TeXt-theme/assets/127359789/a2e6122b-d7d1-4f88-8852-0422ed816559" width="70%" height="70%"></center>

위 그림은 플레이어의 숫자 합이 20 또는 21일 때는 $stick$, 아니면 $hit$하는 policy를 고려했을 때의 state-value function를 보여준다. 10,000 episode 학습 이후, $usable$ ace는 자주 등장하지 않기 때문에, estimate가 덜 확실하고 덜 규칙적인 것을 볼 수 있다. 500,000 episode 학습 이후, value function은 어떤 상황에서든지 잘 approximate된 것을 확인 가능하다. 

<br>

### 5.1.3. Comparsion with DP and backup diagram of MC

env의 완벽한 dynamics를 알고 있더라도, value function 계산에 DP 방법을 적용하는 것은 모든 event에 대한 expected reward와 transition probability를 요구할 뿐만 아니라, 이에 대한 계산은 복잡하고 오류를 유발하므로 매우 어렵다. 그러나 MC 방법은 적용이 쉬우므로, 만약 env의 dynamics를 알더라도 sample episode로 동작하는 것이 훨씬 이점이 많다. 

<br>

<center><img src="https://github.com/kitian616/jekyll-TeXt-theme/assets/127359789/2aae04b0-218a-4b85-bd1c-e6c06c93e43f" width="70%" height="70%"></center>

위 그림에서는 MC 방법에서 $v_\pi$를 추정하기 위해, root는 state node, 그 아래는 single episode 동안의 entire trajectory of ransition으로 구성된다. DP 방법에서는 one-step transition을 보여주었지만, MC에서는 episode의 끝까지를 보여준다. 이 때, 중요한 점은 각 state의 estimate가 independent하다는 것이고, 결과적으로 MC는 DP에서 처럼 bootstrap하지 않는다. 특히, 각 independent state에 대한 value estimate 과정에서 다른 모든 state를 무시하고 해당 state에서 return되는 평균만을 계산하기에 experience를 통해 학습이 가능하다. 

<br>

## 5.2 Monte Carlo Estimation of Action Values

만약 DP에서처럼 model이 존재한다면 state-value만 사용하면, one-step 뒤의 reward와 next state의 조합을 통해 어떤 action이 좋은지 알 수 있었다. 그러나 model이 없다면, state-value로는 명시적인 action value의 estimate로 policy를 생성 하기 부족하므로 MC로 $q_\ast$를 추정하고자 한다. 우리는 action value를 위해 policy evaluation 고려한다.   

state-action pair $s,a$는 state $s$에서 action $a$를 수행한 episode라고 할 수 있다. every-visit MC는 episode 내 방문한 모든 $s,a$의 return을 평균하고, first-visit MC는 episode에서 첫번째로 방문한 $s,a$의 return으로 estimate한다. 이러한 방법은 state-action pair가 infinity로 갈수록, quadratically하게 수렴하게 된다. 

<br>

### 5.2.1 Exploring Starts 
그러나 방문되지 않는 많은 state-action pair가 생길 수 있다는 maintaining exploration 문제가 존재한다. 만약 $\pi$가 deterministic policy라면, 특정 state에서 동일한 action만을 선택할 수도 있다. 따라 continual exploration을 강제하는 방식 중 하나는 episode의 start를 state-action pair를 지정하는 것이다. 이는 모든 state-action pair의 방문을 보장해주게 된다. exploring starts는 간간히 유용하지만, env와 직접적으로 상호작용하는 경우에는 특히 적용이 어렵다. 일반적으로, 이에 대한 대안으로 state에서 모든 action을 선택할 확률이 non-zero인 stochastic policy를 사용한다. 

<br>

## 5.3 Monte Carlo Control

MC estimate를 진행하였으니, 이제 DP chapter에서의 GPI의 아이디어를 이용하여, optimal policy를 approximate하는 MC control이 가능하다. 반복을 통해, value function은 current policy에 가깝게 approximate되고, policy는 current value function을 통해 향상된다. 

<br>

<center><img src="https://github.com/kitian616/jekyll-TeXt-theme/assets/127359789/2a480c0f-762b-4687-be7e-8b8b0642420e" width="70%" height="70%"></center>
이러한 두 종류의 변화는 서로에게 움직이는 목표를 생성하기 때문에, 어느 정도는 서로에게 불리하게 작용하지만 함께 policy와 value function이 모두 optimality에 수렴하도록 만든다. 

<br>

<center><img src="https://github.com/kitian616/jekyll-TeXt-theme/assets/127359789/5d7f6f97-faeb-40f2-989b-f768bf3b3a47" width="70%" height="70%"></center>
evaluation과 improvement를 반복하던 policy iteration의 MC version이라고 생각하면 된다. 많은 episode를 경험할수록, approximate action-value function은 점진적으로 true function에 도달하게 된다. episode들이 exploring starts를 통해서 시작되었고, infinite만큼 경험했다고 가정하면 MC는 arbitary policy $\pi_k$에 대해 정확한 $q_{\pi_k}$를 계산할 수 있다. 

<br>

우리는 더이상 model 없이도 current action-value function에 관해, policy를 greedy하게 만들면 improvement가 수행된다. 각 state $s \in \mathcal{S}$에서 action-value function $q_{\pi_k}$에 대해 deterministically하게 다음과 같이 greedy action을 선택하면 $\pi_{k+1}$가 된다. 

$$ \begin{align*} q_{\pi_k}(s, \pi_{k+1}(s)) &= q_{\pi_k}(s, \arg\max_a q_{\pi_k}(s,a)) \\ &= \max_a q_{\pi_k}(s,a) \\ &\ge q_{\pi_k}(s,\pi_k(s)) \\ &= v_{\pi_k}(s) \tag{1} \end{align*} $$

<br>

MC는 이러한 방식으로 이전 chapter에서 언급했듯이, optimal policy를 찾는 것을 보장한다. 결과적으로, MC 방법은 env의 dynamics 없이, sample episode만으로도 optimal policy를 찾을 수 있다. 

<br>

### 5.3.1. Assumption of Infinite iteration

episode가 infinite하게 반복된다는 가정을 해결하기 위한 첫 번째 방법은 각각의 policy evaluation에서 $q_{\pi_k}$를 approximate한다는 것을 고려하는 것이다. estimate에 대한 error 크기에 대한 bound가 충분히 작도록 충분한 step이 각 policy evaluation에서 이루어져야 한다. 이러한 방법은 어느 정도의 approximation까지 올바른 convergence를 보장한다. 그러나 이 또한 작은 문제를 제외하고는 실용적으로 사용하기에 많은 episode를 여전히 필요로 한다. 

명목상 infinite episode를 피하는 두 번째 방법은 policy improvement 전에 policy evaluation을 중도에 멈추는 것이다. 각 policy evaluation step에서 value function을 $q_{\pi_k}$에 가깝게 이동시키지만, 많은 단계를 거쳐야만 가능하다. policy evaluation 한 번 후에 바로 policy improvement를 진행하는 value iteration에서처럼 동작하면 된다. in-place의 경우는 더욱 극적으로, 매 policy improvement마다가 아닌 single state마다 policy evaluation이 진행된다. 

<br>

### 5.3.2. Monte Carlo ES

Monte Carlo ES에서, 각 state-action pair에 대한 모든 returns는 어떤 policy가 수행되었는지에 관계 없이 accumulate & average 된다. 그러면 value function은 결국 해당 policy로 수렴하고, 결과적으로 policy가 변경될 것이다. stability는 policy와 value function이 모두 optimal일 때만 성립한다. value function의 변동이 시간이 지남에 따라 감소함에 따라, convergence는 불가피하나 수식적으로 증명되지는 않았다. 

<br>

<center><img src="https://github.com/kitian616/jekyll-TeXt-theme/assets/127359789/7ea7c505-1f9f-47fd-bc98-b819137c0b58" width="70%" height="70%"></center>

<br>

### 5.3.3 Applying Monte Carlo ES at Blackjack Example 
이전과 동일한 설정으로 blackjack env에 적용은 간단하다. exploring start를 위해, initial state의 모든 경우의 수를 각각 random equal probability로 설정하여 env를 실행시키면 된다. 

<br>

<center><img src="https://github.com/kitian616/jekyll-TeXt-theme/assets/127359789/bf6e4c65-5618-4ec7-8654-d22232461e87" width="70%" height="70%"></center>

<br>

## 5.4 Monte Carlo Control without Exploring Starts

그러나 언급했듯이 exploring start가 현실적이지 못한 상황이 훨씬 많다. 일반적인 방법으로는 무한하게 action을 선택하는 것이지만, 이를 보장하는 on-policy, off-policy 방법이 존재한다. on-policy는 policy를 evaluate & improve하지만, off-policy는 데이터 생성에 사용되는 policy와 다른 policy를 evaluate & improve한다. Monte Carlo ES는 on-policy에 속한다. 

<br>

### 5.4.1 On-policy Monte Carlo Control

on-policy control 방법은 일반적으로 $\pi(a \mid s) > 0 $ for all $s \in \mathcal{S}, a \in \mathcal{A}(s)$를 충족하는 $soft$하다고 하며, 거의 deterministic optimal policy에 가깝다고 볼 수 있다. chapter 2에서 보았던 $\epsilon$-greedy policy의 모든 non-greedy action들은 선택될 minimal probability $\cfrac{\epsilon}{\mid\mathcal{A}(s)\mid}$로, 그리고 나머지 greedy action은 $1-\epsilon+\cfrac{\epsilon}{\mid\mathcal{A}(s)\mid}$ probability로 선택된다. $\epsilon$-greedy는 $\pi(a \mid s) \ge \cfrac{\epsilon}{\mid\mathcal{A}(s)\mid}, \epsilon > 0$로 정의되는 $\epsilon$-$soft$ policy라고 할 수 있다. $\epsilon$-$soft$ 중, $\epsilon$-greeedy는 가장 greedy에 가깝다고 볼 수 있다.  

<br>

모든 $q_\pi$에 대한 $\epsilon$-greedy poilcy는 policy improvement theorem에 따라 어떤 $\epsilon$-$soft$ policy보다 향상됨을 보장한다. $\pi'$가 $\epsilon$-greedy policy라고 할 때, policy improvement theorem의 $\forall s \in \mathcal{S}$에서의 조건은 다음과 같이 적용된다. 

$$ \begin{align*} q_{\pi}(s, \pi'(s)) &= \sum_a \pi'(a \mid s)q_\pi(s,a) 
\\ &= \cfrac{\epsilon}{|\mathcal{A}(s)|} \sum_a q_\pi(s,a) + (1- \epsilon) \max_a q_\pi (s,a) 
\\ &\ge \cfrac{\epsilon}{|\mathcal{A}(s)|} \sum_a q_\pi(s,a) + (1- \epsilon) \sum_a \cfrac{\pi(a \mid s)-\cfrac{\epsilon}{|\mathcal{A}(s)|}}{1-\epsilon} \ q_\pi(s,a) 
\\&= \cfrac{\epsilon}{|\mathcal{A}(s)|} \sum_a q_\pi(s,a) - \cfrac{\epsilon}{|\mathcal{A}(s)|} \sum_a q_\pi(s,a) + \sum_a \pi(a \mid s)q_\pi(s,a) 
\\&= v_\pi(s)
 \tag{2} \end{align*} $$

<br>

policy improvement theorem에 의해, $\pi' \ge \pi, \forall s\in \mathcal{S}$를 만족한다. 이를 통해, 우리는 $\epsilon$-$soft$ policy 중에서 $\pi'$와 $\pi$가 모두 optimal인 경우, 즉 다른 모든 $\epsilon$-$soft$ policy보다 낫거나 같은 경우에만 equality가 유지될 수 있음을 증명 가능하다. 

<br>

$\epsilon$-$soft$ policy는 기존 exploring start가 적용되던 이전 policy에서의 optimal과 같다. $\tilde{v}_\ast$와 $\tilde{q}_\ast$를 explorint starting가 없는 on-policy optimal value function이라고 할 때, $\pi$는 $\epsilon$-$soft$ policy 중에서 $v_\pi = \tilde{v}_\ast$인 경우에만 optimal이다. $\epsilon$-$soft$의 policy $\pi$가 더이상 향상되지 않는다면 $(5.2)$에 근거하여 $v_\pi$와 $\tilde{v}_\ast$가 동등함을 볼 수 있다. 


$$ \begin{align*} \tilde{v}_\ast(s) &= (1-\epsilon) \max_a \tilde{q}_\ast(s,a) + \cfrac{\epsilon}{|\mathcal{A}(s)|} \sum_a \tilde{q}_\ast(s,a) 
\\ &= (1-\epsilon) \max_a \sum_{s',r} p(s',r \mid s,a)  \left[r + \gamma \tilde{v}_\ast](s')\right] \\ &+  
 \cfrac{\epsilon}{|\mathcal{A}(s)|} \sum_a \sum_{s',r} p(s',r\mid s,a ) [r + \gamma \tilde{v}_\ast(s')]  \end{align*} $$

$$ \begin{align*} v_\pi(s) &= (1-\epsilon) \max_a q_\pi(s,a) + \cfrac{\epsilon}{|\mathcal{A}(s)|} \sum_a q_\pi(s,a) 
\\ &= (1-\epsilon) \max_a \sum_{s',r} p(s',r \mid s,a) \left[r + \gamma v_\pi(s')\right] \\ &+  
 \cfrac{\epsilon}{|\mathcal{A}(s)|} \sum_a \sum_{s',r} p(s',r\mid s,a ) [r + \gamma v_\pi(s')]  \end{align*} $$

<br>

이러한 분석은 각 step에서 action-value function이 결정되는 방법과는 무관하지만 정확하게 계산된다고 가정한다. 결과적으로, $\epsilon$-$soft$ policy는 exploring start 없이도 exploring start가 적용했을 때와 같은 optimal을 보장한다. 

<br>

<center><img src="" width="70%" height="70%"></center>

<br>

## 5.5 Off-policy Predictio nvia Importance Sampling

여기까지는 해당 policy에서의 infinite episode를 사용해서 value function을 estimate하였다. 그러나 각기 다른 policy들로부터 생성된 episode를 가지고있다고 있다고 가정해본다. 우리는 $\pi$를 따르는 target policy를 estimate하고 싶지만, behavior policy $\mu$를 따르는 episode만 가지고 있다. target policy는 learning process에서 목적하는 value function을 지니고, behavior policy는 agent를 조종하고 behavior를 생성한다. 이처럼 target과 behavior policy가 다르기 때문에 off-policy라고 부른다. 

<br>

### 5.5.1 Assumption of Converage

$\mu$를 따르는 episode로부터 $\pi$의 value를 estimate하기 위해서는, $\pi$로부터 수행된 모든 action이 최소한 $\mu$에서도 취해져야한다$(\pi(a\mid s)>0, \mu(a \mid s)>0)$. 만약 $\pi$가 deterministic하다 하더라도, $\mu$는 stochastic해야 한다. 이러한 구조는 stochastic behavior policy가 exploration을 진행하고, target policy가 current action-value function에 근거하여 deterministic하게 움직이는 이전에 언급했던 $\epsilon$-greedy policy의 기반이 된다. 

<br>

### 5.5.2 Importance Sampling (Transition Probability)

importance sampling은 다른 distribution으로부터 주어진 sample을 통해 distribution을 estimate하는 일반적인 방법이다. 우리는 importance-sampling ratio라고 불리는 target과 behavior policy에서 발생하는 trajectory에 대한 relative probability에 따라 return을 weight하는 off-policy learning을 적용한다. 아래는 subsequent state-action trajectory의 probability를 나타낸다. 

$$ \begin{align*} \prod^{T-1}_{k=t} \pi(A_k\mid S_k) p(S_{k+1} \mid S_k, A_k)  \end{align*} $$

<br>

따라, target과 behavior policy를 따르는 trajectory에 대한 relative probability인 importance-sampling ratio는 아래와 같다. 그러나 대부분 우리는 MDP's transition을 모른다. 

$$ \begin{align*} \rho^T_t = \cfrac{\prod^{T-1}_{k=t} \pi(A_k\mid S_k) p(S_{k+1} \mid S_k, A_k)}{\prod^{T-1}_{k=t} \mu(A_k\mid S_k) p(S_{k+1} \mid S_k, A_k)} = \prod^{T-1}_{k=t} \cfrac{\pi(A_k\mid S_k)}{\mu(A_k\mid S_k)} \tag{3} \end{align*} $$

<br>

### 5.5.3 Ordinary Importance Sampling (Batch of Episode)

따라서 우리는 $v_\pi(s)$를 estimate하기 위해 $\mu$로부터 관찰된 batch of episode를 사용한다. 한 episode가 종료된 다음 time step부터 다음 episode를 시작하면, 특정 episode의 특정 step을 의미하여 time step number를 사용 할 수 있다. first-visit 방법으로는 state $s$를 episode 내 처음으로 방문한 time step를 $\mathcal{T}(s)$로 표기한다. $T(s)$는 $t$ 이후에 첫 번째 termination을, $G_t$는 $t$ 방문 이후 $T(s)$까지의 return을 의미한다. $\{G_t \}_{t \in \mathcal{T}(s)}$는 state $s$에 관련된 return, $\{\rho^{T(t)}_t \}_{t \in \mathcal{T}(s)}$ importance-sampling ratios이다. $v_\pi(s)$를 estimate하기 위해서는 return을 ratio에 따라 scale하고 average하면 된다. simple average한다면 보통 ordinary importance sampling이라고 부른다. 

$$ \begin{align*} V(s) = \cfrac{\sum_{t\in \mathcal{T}(s)} \rho^{T(t)}_t G_t}{\mid \mathcal{T}(s)\mid}\tag{4} \end{align*} $$

<br>

### 5.5.3 Weighted Importance Sampling (Batch of Episode)

아래와 같이 weihgted average를 사용하는 weighted importance sampling도 있다.

$$ \begin{align*} V(s) = \cfrac{\sum_{t\in \mathcal{T}(s)} \rho^{T(t)}_t G_t}{\sum_{t\in \mathcal{T}(s)} \rho^{T(t)}_t}\tag{5} \end{align*} $$
혹은 만약 분모가 0인 경우에는 0이다. single return에 대한 ratio $\rho^{T(t)}_t$는 1이므로 비율과 무관하게 observed return과 동일하다. 만약 ratio가 10, behavior policy에서 trajectory가 10번 관측되었다고 가정한다. 

<br>

두 importance sampline 방법의 차이는 variance이다. ordinary의 경우 ratio가 unbounded하기 때문에 variance 또한 unbounded되지만, weighted의 경우 single return에서 가장 큰 weight가 1이다. weighted importance sampline에서 실제로 ratio의 variance가 unbounded하다고 해도, variance는 0으로 수렴한다. 따라 variance가 매우 낮아 일반적으로 weighted 방법이 주로 사용된다. 

<br>

### 5.5.4 Off-policy Estimation of a Blackjack State Value

off-policy data에서 single blackjack state의 value를 estimate해본다. 여기서는 dealer가 deuce를 보이고, player card의 sum이 13, player가 usable ace를 지닌 state를 고려한다. data는 해당 state에서 hit or stick하는 action을 random equal probability로 선택하는 behavior policy로 생성된다. target policy는 이전에서와 같이 sum이 20 혹은 21일 때만 stick한다. 

<br>

<center><img src="" width="70%" height="70%"></center>


위 그림에서 보면 1000 episode 이후, 두 off-policy(behavior policy) 방법 모두 동일하게 수렴한다. weighted 방법은 ordinary보다 전체적으로 error 폭이 훨씬 낮은 것을 볼 수 있다. 

<br>

### 5.5.5 Infinte Variance

ordinary importance sampling의 estimate는 일반적으로 infinite variance를 가지고 convergence에 좋지 않다. 이는 종종 off-policy 방법에서 trajectory가 loop를 가질 때 발생한다. 아래 예제에서는 nonterminal state $s$와 $end$와 $back$ 총 두가지 action만이 존재한다. $end$는 deterministic하게 termination으로, $back$은 0.9 probability로 $s$로 되돌아가거나 0.1로 termination으로 전환된다. reward는 후자의 transition에서 $+1$ 그렇지 않으면 0이다. 

target policy는 항상 $back$을 선택하므로, 해당 policy를 따르는 $s$의 value는 1이다. off-policy data 생성을 위한 behavior policy는 $end$와 $back$을 같은 probability로 선택한다. $(5.8)$는 ordinary importance sampling을 사용한 10개의 독립적인 first-visit MC 알고리즘을 보여준다. millions episodes 이후에도 estimate는 1로 converge하는데 실패한다. 반면 weighted importance sampling은 target policy와 일치하는 ($back$으로 terminate된) 첫 번째 episode 이후에 정확히 1의 estimate를 제공한다. 알고리즘이 target policy와 일치하는 return의 weight average를 생성하기 때문에 모두 정확히 1이된다. 

<center><img src="" width="70%" height="70%"></center>

<br>



<br>

## 5.6 Incremental Implementation

