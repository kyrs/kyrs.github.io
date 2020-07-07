---
title: 'Markov chains and Infinite horizon problems'
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
 
 A standard reinforcement learning (RL) algorithms try to devise a policy or procedure of actions which maximize the expected reward over a trajectory for a given problem.  Such a problem can range from playing a regular atari game to generating the best strategy for self-driving cars to drive on the road.  RL has many applications in our day to day life. 
$$
\begin{align*}\\
argmax_{\theta} E_{\tau \sim P(\theta)}[\sum_{t=0}^{t=k}r(s_{t},a_{t})]
\\
\end{align*}
$$


One of the vital attributes in any RL problem is the length of the trajectory.  Often termed as episode length, it is the duration for which the Rl problem has to maximize the expected reward and is fixed for most of the problem. Nevertheless, there are real-life problems where one cannot put a limit over the length of the trajectory. This problem, often known as an infinite horizon, is unique and requires special handling. Compared to the standard RL problem statement, here we want to maximize the expected reward for infinite length objectives.  

$$
\begin{align*}\\
argmax_{\theta} E_{\tau \sim P(\theta)}[\sum_{t=0}^{t=\infty}r(s_{t},a_{t})]
\\
\end{align*}
$$

Let us try to understand it with one problem statement. 

# Random walk over webpages

Most of the page rank algorithms rely on a random walk over a fully-connected web page and then associate a given page with a rank based on the distribution of its visit.  Let us devise our RL problem statement over a similar task,  where we want to learn a policy/trajectory to maximize hit to a particular kind of web page with some specific property. To make sure that such a trajectory is not biased over our starting state, let us look into an infinite random walk over such a graph.  For our RL problem, we will have +1 reward for visiting a sport-related page and 0 if it is not. Our Infinite horizon RL problem will be to generate a policy that leads to a more sport-related page either in the recent future or in a distant one.  We can define our problem as an MDP with pages as its state and choosing a given hyperlink as the action in a given state. The probability of getting into the next state based on the current state can be posed as a property of environment and solved using a model-free RL algorithm, or if we know the transition probability, then we can use a value iteration/policy iteration to deduce our policy. 
To solve this problem, let us look into the second part of our discussion i.e., Markov chain and Markov random fields.

# Markov chain and Markov property
In a laymen term, a Markov property states that, for a sequence-based problem, a decision on a given state depend only on the property of the current state and is irrelevant to the history of previously visited sequence(states visited). For example, let us suppose a robot has to reach the market from our home, and after 1 hour, it is at lane no "x".  For the robot, the final task of reaching the market from "x" is irrelevant to how it reached "x" . The final task of reaching the market depends only on the decision at lane no "x". Mathematically this can be represented as 

$$
\begin{align*}\\
P(s_{t+1}|s_{1},s_{2},s_{3} ... s_{t)} = P(s_{t+1}| s_{t})\\
\end{align*}
$$

the corresponding graphical representation will look something like this.
![](https://kyrs.github.io/files/infnite-horizon/markov_chain.png).

# Ergodic Markov chain
Ergodic Markov chain is a particular type of Markov chain which satisfy two crucial property :
- **aperiodic :** If the highest common factor among all the cycles possible within a given graph is one, then it is called aperiodic. One easy way to make any graph aperiodic is to have a self-loop for each of the node or in another word a non zero probability to stay in same state i.e.

$$
\begin{align*}\\
P(s_{t+1} = s_{t}|s_{t)}) >0\\
\end{align*}
$$
- **Fully connected:** a graph is said to be fully connected if there is a non zero probability for reaching any state from any other state through an arbitrarily long or short path. 



# stationary distribution
A random walk over a Markov chain, which is ergodic converges to a stationary distribution, defined over its states. In other words, If a random walk keeps on running over a Markov chain for a very long trajectory, then the probability of its visits to a given state is said to converge to a fixed value. 
Let us calculate the stationary probability for a defined small set of states. 
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

For a problem where state space is enormous, and transition probability is not known.  It becomes highly impractical to calculate the eigenvector for such a big transition operator. Researcher in such scenario prefers to use MCMC algorithm for sampling and approximating the state space. 

# Reinforcement learning as a Markov chain
a graphical representation of reinforcement learning looks something like this.

![](https://kyrs.github.io/files/infnite-horizon/Rl.png)
This graph can easily be modified as a Markov chain with the given structure.
![](https://kyrs.github.io/files/infnite-horizon/RL_markov_chain.png)

Considering we have a mechanism to sample from the stationary distribution of the state, action pair. The overall RL objective over an infinite horizon can be rewritten as 

$$
\begin{align*}
 argmax_{\theta} E_{(s_t,a_t) \sim P_\theta(s_t,a_t))}[r(s_{t},a_{t})] 
 \end{align*}
 $$ 



where $$ P_\theta(s_t,a_t)$$ denotes the stationary distribution of the given markov chain. 

For a fixed-length episode this problem will take a form of  


$$
\begin{align*}
 argmax_{\theta} E_{(s_t,a_t) \sim P_\theta(s_t,a_t))}[\sum_{t=0}^{t=k}r(s_{t},a_{t})] 
 \end{align*}
 $$ 



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