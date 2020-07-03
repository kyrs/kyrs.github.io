---
title: 'What does it mean to solve an infinite horizon RL ?'
date: 2020-07-05
permalink: /posts/2020/07/infinite-horizon-RL/
<!-- tags:
  - Reinforcement Learning
  - Markov chain
  - Page Rank -->
---

# Introduction
<script type="text/javascript" async
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
Reinforcement learning genreally involves maximizing the expected reward over a defined trajectory for a given problem. 
$$
\begin{align*}\\
argmax_{\theta} E_{\tau \sim P(\theta)}[\sum_{t=0}^{t=k}r(s_{t},a_{t})\gamma^{t}]
\\
\end{align*}
$$
The length of the trajectory often termed as episode length is fixed for most of the problem and involves  multiple sampling from learnt policy to approximate expected reward for maximization. But often their are real life problems where the length of the trajectory can't be fixed. This problem often known as infinite horizon are quite unique and requires special handling. Compared to previously mentioned problem statement, here we want to maximize .
$$
\begin{align*}\\
argmax_{\theta} E_{\tau \sim P(\theta)}[\sum_{t=0}^{t=\infty}r(s_{t},a_{t})\gamma^{t}]
\\
\end{align*}
$$

Let's dig into it with one such example. 


# Random walk over webpages

Most of the page rank algorithms, basically relies on random walk over a fully-connected web pages and then associate given page with a rank based on the distribution of its visit. But let's suppose we want to learn a policy over this infinite random walk which lead to maximization of certain features, for example it could be on type of pages we visit. We can have a reward scheme where we get +1 for visiting a sport related page and 0 if it is not. Then an Infnite horizon RL problem will be to generate a policy which leads to more of a sport related pages either in recent future or a distant one.  We can define our problem as an MDP with pages as it's state, choosing a given hyperlink as an action, +1,0 reward based on whether the page was sport related or not and the probability of getting into the next state based on current state as a deterministic probility distribution defined over the hyperlink of a given page. One thing to note in the given problem is that, we are dealing with an infinite trajectory of sites over web pages, in other word we are not looking into maximizing expected reward over a fixed length of hop i.e.  let's say finding the trajectory of reaching most sport related pages in 50 or 60 hops. But, we are interested in exploring such a path over an infinite continuous hop. Such, a process basically  lead to a much better exploration of web and don't induce any biases related to trajectory length. To solve this problem, we should first try to understand what a markov process is and what is it's properties.   

# Markov chain and markov property
In a laymen term a markov property states that, if for a sequence based problem a decison on a given state depend only on the property of a given state and is irrelevant to the history of sequence(states visited) before it then it's said to satisfy markov property. E.g. let's suppose my task is to reach the market from my home, and after 1 hour, I am at lane no "x".  For me to reach market from "x", is irrelevant of how I reached "x". The final task of reaching market just depend upon what decision I will make at lane no "x". mathematically this can be represented as 

$$
\begin{align*}\\
P(s_{t+1}|s_{1},s_{2},s_{3} ... s_{t)} = P(s_{t+1}| s_{t})\\
\end{align*}
$$

corrresponding graphical representation will look something like this.
![](https://kyrs.github.io/files/infnite-horizon/markov_chain.png).

# Ergodic markov chain
Ergodic markov chain are a special type of markov chain which satisfy two important property :
- **aperiodic :** If the highest common factor among all the cycles possible within a given graph is 1 then it is said to be aperiodic. One easy way to make any graph aperiodic is to have a self loop for each of the node or in other word a non zero probability to stay in same state i.e.

$$
\begin{align*}\\
P(s_{t+1} = s_{t}|s_{t)}) >0\\
\end{align*}
$$
- **Fully connected :** a graph is said to be fully connected if there is a non zero probability for reaching any state from any other state through a arbitrary long or short path. 

# stationary distribution
A random walk over a markov chain which is ergodic in nature converges to a stationary distribution, defined over it's states. In other word, If you keep on running a markov chain for a very long duration then probability of it's visit to a given state is said to converge to a fixed value. 
let's calculate the stationary probability for a defined small set of states. 
let 

$$\textit(S)$$ - state space 

$$\textit(T)$$ - trainsition operator 

$$\mu_{t,i}$$ - probability of being at $$i^{th}$$ state during $$t^{th}$$ transition $$ P(s_t = i)$$ 

let $$\textit(T_{i,j}) =  P(s_{t+1} = i \mid s_{t} = j )$$ 

Now the state probability distribution can be expressed as :
$$
 \overrightarrow{\mu_{t+1}}  = \textit(T_{i,j}) * \overrightarrow{\mu_{t}}
$$

on limit of $$t \rightarrow \infty$$ $$
 \overrightarrow{\mu_{t}}  = \textit(T_{i,j}) * \overrightarrow{\mu_{t}}
$$

in such case $$ \overrightarrow{\mu_{t}} $$ is equivalent to principal eigen vector of $$\textit(T)$$ operator.

Genereally, for problem where state space is huge, it's highly impractical to calculate the eigen vector for such a big transition operator and researcher genreally use MCMC algorithm for sampling and approximation of state space. 

# Reinforcement learning as a markov chain
a graphical representation of reinforcement learning genereally look something like this.

![](https://kyrs.github.io/files/infnite-horizon/RL.png)
This graph can easily be approximated as a markov chain with given structure.
![](https://kyrs.github.io/files/infnite-horizon/RL_markov_chain.png)
although the state transition operator doesn't have a closed form equation and infact is governed by environment.


<!-- $$
\begin{align*}
  & \phi(x,y) = \phi \left(\sum_{i=1}^n x_ie_i, \sum_{j=1}^n y_je_j \right)
  = \sum_{i=1}^n \sum_{j=1}^n x_i y_j \phi(e_i, e_j) = \\
  & (x_1, \ldots, x_n) \left( \begin{array}{ccc}
      \phi(e_1, e_1) & \cdots & \phi(e_1, e_n) \\
      \vdots & \ddots & \vdots \\
      \phi(e_n, e_1) & \cdots & \phi(e_n, e_n)
    \end{array} \right)
  \left( \begin{array}{c}
      y_1 \\
      \vdots \\
      y_n
    \end{array} \right)
\end{align*}
$$ -->