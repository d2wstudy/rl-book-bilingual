# Chapter 4

# Dynamic Programming

The term dynamic programming (DP) refers to a collection of algorithms that can beused to compute optimal policies given a perfect model of the environment as a Markovdecision process (MDP). Classical DP algorithms are of limited utility in reinforcementlearning both because of their assumption of a perfect model and because of their greatcomputational expense, but they are still important theoretically. DP provides an essentialfoundation for the understanding of the methods presented in the rest of this book. Infact, all of these methods can be viewed as attempts to achieve much the same e↵ect asDP, only with less computation and without assuming a perfect model of the environment.

Starting with this chapter, we usually assume that the environment is a finite MDP.That is, we assume that its state, action, and reward sets, S, $\mathcal A$ , and $\mathcal { R }$ , are finite, andthat its dynamics are given by a set of probabilities $p ( s ^ { \prime } , r | s , a )$ , for all $s \in \mathcal { S }$ , $a \in { \mathcal { A } } ( s )$ ,$r \in \mathcal { R }$ , and $s ^ { \prime } \in \mathcal { S } ^ { + }$ ${ { \mathcal { S } } ^ { + } }$ is S plus a terminal state if the problem is episodic). AlthoughDP ideas can be applied to problems with continuous state and action spaces, exactsolutions are possible only in special cases. A common way of obtaining approximatesolutions for tasks with continuous states and actions is to quantize the state and actionspaces and then apply finite-state DP methods. The methods we explore in Part II areapplicable to continuous problems and are a significant extension of that approach.

The key idea of DP, and of reinforcement learning generally, is the use of value functionsto organize and structure the search for good policies. In this chapter we show how DPcan be used to compute the value functions defined in Chapter 3. As discussed there, wecan easily obtain optimal policies once we have found the optimal value functions, $v _ { * }$ or$q _ { * }$ , which satisfy the Bellman optimality equations:

$$
\begin{array}{l} v _ {*} (s) = \max  _ {a} \mathbb {E} \left[ R _ {t + 1} + \gamma v _ {*} \left(S _ {t + 1}\right) \mid S _ {t} = s, A _ {t} = a \right] \\ = \max  _ {a} \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r \mid s, a\right) \left[ r + \gamma v _ {*} \left(s ^ {\prime}\right) \right], \tag {4.1} \\ \end{array}
$$

or

$$
\begin{array}{l} q _ {*} (s, a) = \mathbb {E} \left[ R _ {t + 1} + \gamma \max  _ {a ^ {\prime}} q _ {*} (S _ {t + 1}, a ^ {\prime}) \mid S _ {t} = s, A _ {t} = a \right] \\ = \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r \mid s, a\right) \left[ r + \gamma \max  _ {a ^ {\prime}} q _ {*} \left(s ^ {\prime}, a ^ {\prime}\right) \right], \tag {4.2} \\ \end{array}
$$

for all $s \in \mathcal { S }$ , $a \in { \mathcal { A } } ( s )$ , and $s ^ { \prime } \in \mathcal { S } ^ { + }$ . As we shall see, DP algorithms are obtained byturning Bellman equations such as these into assignments, that is, into update rules forimproving approximations of the desired value functions.

# 4.1 Policy Evaluation (Prediction)

First we consider how to compute the state-value function $v _ { \pi }$ for an arbitrary policy $\pi$ .This is called policy evaluation in the DP literature. We also refer to it as the predictionproblem. Recall from Chapter 3 that, for all $s \in \mathcal { S }$ ,

$$
\begin{array}{l} v _ {\pi} (s) \doteq \mathbb {E} _ {\pi} \left[ G _ {t} \mid S _ {t} = s \right] \\ = \mathbb {E} _ {\pi} \left[ R _ {t + 1} + \gamma G _ {t + 1} \mid S _ {t} = s \right] \quad (\text {f r o m (3 . 9)}) \\ = \mathbb {E} _ {\pi} \left[ R _ {t + 1} + \gamma v _ {\pi} \left(S _ {t + 1}\right) \mid S _ {t} = s \right] (4.3) \\ = \sum_ {a} \pi (a | s) \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r \mid s, a\right) \left[ r + \gamma v _ {\pi} \left(s ^ {\prime}\right) \right], (4.4) \\ \end{array}
$$

where $\pi ( a | s )$ is the probability of taking action $a$ in state $s$ under policy $\pi$ , and theexpectations are subscripted by $\pi$ to indicate that they are conditional on $\pi$ being followed.The existence and uniqueness of $v _ { \pi }$ are guaranteed as long as either $\gamma < 1$ or eventualtermination is guaranteed from all states under the policy $\pi$ .

If the environment’s dynamics are completely known, then (4.4) is a system of |S|simultaneous linear equations in |S| unknowns (the $v _ { \pi } ( s )$ , $s \in \mathcal { S }$ ). In principle, its solutionis a straightforward, if tedious, computation. For our purposes, iterative solution methodsare most suitable. Consider a sequence of approximate value functions $v _ { 0 } , v _ { 1 } , v _ { 2 } , \ldots$ ,each mapping ${ \mathcal { S } } ^ { + }$ to $\mathbb { R }$ (the real numbers). The initial approximation, $v _ { 0 }$ , is chosenarbitrarily (except that the terminal state, if any, must be given value 0), and eachsuccessive approximation is obtained by using the Bellman equation for $v _ { \pi }$ (4.4) as anupdate rule:

$$
\begin{array}{l} v _ {k + 1} (s) \quad \doteq \quad \mathbb {E} _ {\pi} \left[ R _ {t + 1} + \gamma v _ {k} \left(S _ {t + 1}\right) \mid S _ {t} = s \right] \\ = \sum_ {a} \pi (a | s) \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r \mid s, a\right) \left[ r + \gamma v _ {k} \left(s ^ {\prime}\right) \right], \tag {4.5} \\ \end{array}
$$

for all $s \in \mathcal { S }$ . Clearly, $v _ { k } = v _ { \pi }$ is a fixed point for this update rule because the Bellmanequation for $v _ { \pi }$ assures us of equality in this case. Indeed, the sequence $\{ v _ { k } \}$ can beshown in general to converge to $v _ { \pi }$ as $k  \infty$ under the same conditions that guaranteethe existence of $v _ { \pi }$ . This algorithm is called iterative policy evaluation.

To produce each successive approximation, $v _ { k + 1 }$ from $v _ { k }$ , iterative policy evaluationapplies the same operation to each state $s$ : it replaces the old value of $s$ with a new valueobtained from the old values of the successor states of $s$ , and the expected immediaterewards, along all the one-step transitions possible under the policy being evaluated. Wecall this kind of operation an expected update. Each iteration of iterative policy evaluationupdates the value of every state once to produce the new approximate value function

$v _ { k + 1 }$ . There are several di↵erent kinds of expected updates, depending on whether astate (as here) or a state–action pair is being updated, and depending on the precise waythe estimated values of the successor states are combined. All the updates done in DPalgorithms are called expected updates because they are based on an expectation over allpossible next states rather than on a sample next state. The nature of an update canbe expressed in an equation, as above, or in a backup diagram like those introduced inChapter 3. For example, the backup diagram corresponding to the expected update usedin iterative policy evaluation is shown on page 59.

To write a sequential computer program to implement iterative policy evaluation asgiven by (4.5) you would have to use two arrays, one for the old values, $v _ { k } ( s )$ , and onefor the new values, $v _ { k + 1 } ( s )$ . With two arrays, the new values can be computed one byone from the old values without the old values being changed. Alternatively, you coulduse one array and update the values “in place,” that is, with each new value immediatelyoverwriting the old one. Then, depending on the order in which the states are updated,sometimes new values are used instead of old ones on the right-hand side of (4.5). Thisin-place algorithm also converges to $v _ { \pi }$ ; in fact, it usually converges faster than thetwo-array version, as you might expect, because it uses new data as soon as they areavailable. We think of the updates as being done in a sweep through the state space. Forthe in-place algorithm, the order in which states have their values updated during thesweep has a significant influence on the rate of convergence. We usually have the in-placeversion in mind when we think of DP algorithms.

A complete in-place version of iterative policy evaluation is shown in pseudocode inthe box below. Note how it handles termination. Formally, iterative policy evaluationconverges only in the limit, but in practice it must be halted short of this. The pseudocodetests the quantity $\mathrm { m a x } _ { s \in \mathcal { S } } | v _ { k + 1 } ( s ) - v _ { k } ( s ) |$ after each sweep and stops when it is su cientlysmall.

# Iterative Policy Evaluation, for estimating V ⇡ v⇡

Input $\pi$ , the policy to be evaluated

Algorithm parameter: a small threshold $\theta > 0$ determining accuracy of estimation

Initialize $V ( s )$ arbitrarily, for $s \in \mathcal { S }$ , and $V ( t e r m i n a l )$ t o 0

Loop:

$$
\Delta \gets 0
$$

Loop for each $s \in \mathcal { S }$ :

$$
v \leftarrow V (s)
$$

$$
V (s) \leftarrow \sum_ {a} \pi (a | s) \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r | s, a\right) \left[ r + \gamma V \left(s ^ {\prime}\right) \right]
$$

$$
\Delta \leftarrow \max  (\Delta , | v - V (s) |)
$$

ntil $\Delta < \theta$

Example 4.1 Consider the $4 \times 4$ gridworld shown below.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/a6f81d64f3ee50baf5062704d0d30771cf3d64355554418091f202785790b8f5.jpg)


<table><tr><td></td><td>1</td><td>2</td><td>3</td></tr><tr><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>8</td><td>9</td><td>10</td><td>11</td></tr><tr><td>12</td><td>13</td><td>14</td><td></td></tr></table>

$R _ { t } = - 1$on all transitions

The nonterminal states are $\mathcal { S } = \{ 1 , 2 , \ldots , 1 4 \}$ . There are four actions possible in eachstate, ${ \mathcal { A } } = \{ { \mathrm { u p } } $ , down, right, left}, which deterministically cause the correspondingstate transitions, except that actions that would take the agent o↵ the grid in fact leavethe state unchanged. Thus, for instance, $p ( 6 , - 1 | 5 , \mathtt { r i g h t } ) = 1$ , $p ( 7 , - 1 | 7 , \mathtt { r i g h t } ) = 1$ ,and $p ( { 1 0 } , { r } | 5 , \mathtt { r i g h t } ) = 0$ for all $r \in \mathcal { R }$ . This is an undiscounted, episodic task. Thereward is $- 1$ on all transitions until the terminal state is reached. The terminal state isshaded in the figure (although it is shown in two places, it is formally one state). Theexpected reward function is thus $r ( s , a , s ^ { \prime } ) = - 1$ for all states $s , s ^ { \prime }$ and actions $a$ . Supposethe agent follows the equiprobable random policy (all actions equally likely). The left sideof Figure 4.1 shows the sequence of value functions $\{ v _ { k } \}$ computed by iterative policyevaluation. The final estimate is in fact $v _ { \pi }$ , which in this case gives for each state thenegation of the expected number of steps from that state until termination. ■

Exercise 4.1 In Example 4.1, if $\pi$ is the equiprobable random policy, what is $q _ { \pi } ( 1 1 , \mathsf { d o w n } ) ^ { \cdot }$ ?What is $q _ { \pi } ( 7 , \mathsf { d o w n } ) ^ { \star }$ ? ⇤

Exercise 4.2 In Example 4.1, suppose a new state 15 is added to the gridworld just belowstate 13, and its actions, left, up, right, and down, take the agent to states 12, 13, 14,and 15, respectively. Assume that the transitions from the original states are unchanged.What, then, is $v _ { \pi } ( 1 5 )$ for the equiprobable random policy? Now suppose the dynamics ofstate 13 are also changed, such that action down from state 13 takes the agent to the newstate 15. What is $v _ { \pi } ( 1 5 )$ for the equiprobable random policy in this case? ⇤

Exercise 4.3 What are the equations analogous to (4.3), (4.4), and (4.5), but for action-value functions instead of state-value functions? ⇤

# 4.2 Policy Improvement

Our reason for computing the value function for a policy is to help find better policies.Suppose we have determined the value function $v _ { \pi }$ for an arbitrary deterministic policy$\pi$ . For some state $s$ we would like to know whether or not we should change the policyto deterministically choose an action $a \neq \pi ( s )$ . We know how good it is to follow thecurrent policy from $s$ —that is $v _ { \pi } ( s )$ —but would it be better or worse to change to thenew policy? One way to answer this question is to consider selecting $a$ in $s$ and thereafter

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/53e11f81c05c83ea3c53dd641a5b0144755f0a33882f19277da1861c6c3b968f.jpg)



Figure 4.1: Convergence of iterative policy evaluation on a small gridworld. The left column isthe sequence of approximations of the state-value function for the random policy (all actionsequally likely). The right column is the sequence of greedy policies corresponding to the valuefunction estimates (arrows are shown for all actions achieving the maximum, and the numbersshown are rounded to two significant digits). The last policy is guaranteed only to be animprovement over the random policy, but in this case it, and all policies after the third iteration,are optimal.


following the existing policy, $\pi$ . The value of this way of behaving is

$$
\begin{array}{l} q _ {\pi} (s, a) \doteq \mathbb {E} \left[ R _ {t + 1} + \gamma v _ {\pi} \left(S _ {t + 1}\right) \mid S _ {t} = s, A _ {t} = a \right] \tag {4.6} \\ = \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r \mid s, a\right) \left[ r + \gamma v _ {\pi} \left(s ^ {\prime}\right) \right]. \\ \end{array}
$$

The key criterion is whether this is greater than or less than $v _ { \pi } ( s )$ . If it is greater—thatis, if it is better to select $a$ once in $s$ and thereafter follow $\pi$ than it would be to follow$\pi$ all the time—then one would expect it to be better still to select $a$ every time $s$ isencountered, and that the new policy would in fact be a better one overall.

That this is true is a special case of a general result called the policy improvementtheorem. Let $\pi$ and $\pi ^ { \prime }$ be any pair of deterministic policies such that, for all $s \in \mathcal { S }$ ,

$$
q _ {\pi} (s, \pi^ {\prime} (s)) \geq v _ {\pi} (s). \tag {4.7}
$$

Then the policy $\pi ^ { \prime }$ must be as good as, or better than, $\pi$ . That is, it must obtain greateror equal expected return from all states $s \in \mathcal { S }$ :

$$
v _ {\pi^ {\prime}} (s) \geq v _ {\pi} (s). \tag {4.8}
$$

Moreover, if there is strict inequality of (4.7) at any state, then there must be strictinequality of (4.8) at that state.

The policy improvement theorem applies to the two policies that we considered at thebeginning of this section: an original deterministic policy, $\pi$ , and a changed policy, $\pi ^ { \prime }$ ,that is identical to $\pi$ except that $\pi ^ { \prime } ( s ) = a \neq \pi ( s )$ . For states other than $s$ , (4.7) holdsbecause the two sides are equal. Thus, if $q _ { \pi } ( s , a ) > v _ { \pi } ( s )$ , then the changed policy isindeed better than $\pi$ .

The idea behind the proof of the policy improvement theorem is easy to understand.Starting from (4.7), we keep expanding the $q _ { \pi }$ side with (4.6) and reapplying (4.7) untilwe get $v _ { \pi ^ { \prime } } ( s )$ :

$$
\begin{array}{l} v _ {\pi} (s) \leq q _ {\pi} (s, \pi^ {\prime} (s)) \\ = \mathbb {E} \left[ R _ {t + 1} + \gamma v _ {\pi} \left(S _ {t + 1}\right) \mid S _ {t} = s, A _ {t} = \pi^ {\prime} (s) \right] (by(4.6)) \\ = \mathbb {E} _ {\pi^ {\prime}} \left[ R _ {t + 1} + \gamma v _ {\pi} \left(S _ {t + 1}\right) \mid S _ {t} = s \right] \\ \leq \mathbb {E} _ {\pi^ {\prime}} \left[ R _ {t + 1} + \gamma q _ {\pi} \left(S _ {t + 1}, \pi^ {\prime} \left(S _ {t + 1}\right)\right) \mid S _ {t} = s \right] (by(4.7)) \\ = \mathbb {E} _ {\pi^ {\prime}} \left[ R _ {t + 1} + \gamma \mathbb {E} \left[ R _ {t + 2} + \gamma v _ {\pi} \left(S _ {t + 2}\right) \mid S _ {t + 1}, A _ {t + 1} = \pi^ {\prime} \left(S _ {t + 1}\right) \right] \mid S _ {t} = s \right] \\ = \mathbb {E} _ {\pi^ {\prime}} \left[ R _ {t + 1} + \gamma R _ {t + 2} + \gamma^ {2} v _ {\pi} \left(S _ {t + 2}\right) \mid S _ {t} = s \right] \\ \leq \mathbb {E} _ {\pi^ {\prime}} \left[ R _ {t + 1} + \gamma R _ {t + 2} + \gamma^ {2} R _ {t + 3} + \gamma^ {3} v _ {\pi} \left(S _ {t + 3}\right) \mid S _ {t} = s \right] \\ \end{array}
$$

$$
\begin{array}{l} \leq \mathbb {E} _ {\pi^ {\prime}} \left[ R _ {t + 1} + \gamma R _ {t + 2} + \gamma^ {2} R _ {t + 3} + \gamma^ {3} R _ {t + 4} + \dots \mid S _ {t} = s \right] \\ = v _ {\pi^ {\prime}} (s). \\ \end{array}
$$

So far we have seen how, given a policy and its value function, we can easily evaluatea change in the policy at a single state. It is a natural extension to consider changes at

all states, selecting at each state the action that appears best according to $q _ { \pi } ( s , a )$ . Inother words, to consider the new greedy policy, $\pi ^ { \prime }$ , given by

$$
\begin{array}{l} \pi^ {\prime} (s) \quad \doteq \quad \underset {a} {\operatorname {a r g m a x}} q _ {\pi} (s, a) \\ = \underset {a} {\arg \max } \mathbb {E} \left[ R _ {t + 1} + \gamma v _ {\pi} \left(S _ {t + 1}\right) \mid S _ {t} = s, A _ {t} = a \right] \tag {4.9} \\ = \arg \max  _ {a} \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r \mid s, a\right) \left[ r + \gamma v _ {\pi} \left(s ^ {\prime}\right) \right], \\ \end{array}
$$

where $\mathrm { a r g m a x } _ { a }$ $^ a$ denotes the value of $a$ at which the expression that follows is maximized(with ties broken arbitrarily). The greedy policy takes the action that looks best in theshort term—after one step of lookahead—according to $v _ { \pi }$ . By construction, the greedypolicy meets the conditions of the policy improvement theorem (4.7), so we know that itis as good as, or better than, the original policy. The process of making a new policy thatimproves on an original policy, by making it greedy with respect to the value function ofthe original policy, is called policy improvement.

Suppose the new greedy policy, $\pi ^ { \prime }$ , is as good as, but not better than, the old policy $\pi$Then $v _ { \pi } = v _ { \pi ^ { \prime } }$ , and from (4.9) it follows that for all $s \in \mathcal { S }$ :

$$
\begin{array}{l} v _ {\pi^ {\prime}} (s) = \max  _ {a} \mathbb {E} \left[ R _ {t + 1} + \gamma v _ {\pi^ {\prime}} \left(S _ {t + 1}\right) \mid S _ {t} = s, A _ {t} = a \right] \\ = \max  _ {a} \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r \mid s, a\right) \left[ r + \gamma v _ {\pi^ {\prime}} \left(s ^ {\prime}\right) \right]. \\ \end{array}
$$

But this is the same as the Bellman optimality equation (4.1), and therefore, $v _ { \pi ^ { \prime } }$ must be$v _ { * }$ , and both $\pi$ and $\pi ^ { \prime }$ must be optimal policies. Policy improvement thus must give us astrictly better policy except when the original policy is already optimal.

So far in this section we have considered the special case of deterministic policies.In the general case, a stochastic policy $\pi$ specifies probabilities, $\pi ( a | s )$ , for taking eachaction, $a$ , in each state, $s$ . We will not go through the details, but in fact all the ideas ofthis section extend easily to stochastic policies. In particular, the policy improvementtheorem carries through as stated for the stochastic case. In addition, if there are ties inpolicy improvement steps such as (4.9)—that is, if there are several actions at which themaximum is achieved—then in the stochastic case we need not select a single action fromamong them. Instead, each maximizing action can be given a portion of the probabilityof being selected in the new greedy policy. Any apportioning scheme is allowed as longas all submaximal actions are given zero probability.

The last row of Figure 4.1 shows an example of policy improvement for stochasticpolicies. Here the original policy, $\pi$ , is the equiprobable random policy, and the newpolicy, $\pi ^ { \prime }$ , is greedy with respect to $v _ { \pi }$ . The value function $v _ { \pi }$ is shown in the bottom-leftdiagram and the set of possible $\pi ^ { \prime }$ is shown in the bottom-right diagram. The stateswith multiple arrows in the $\pi ^ { \prime }$ diagram are those in which several actions achieve themaximum in (4.9); any apportionment of probability among these actions is permitted.For any such policy, its state values $v _ { \pi ^ { \prime } } ( s )$ can be seen by inspection to be either $^ { - 1 }$ , $- 2$ ,or $^ { - 3 }$ , for all states $s \in \mathcal { S }$ , whereas $v _ { \pi } ( s )$ is at most $^ { - 1 4 }$ . Thus, $v _ { \pi ^ { \prime } } ( s ) \geq v _ { \pi } ( s )$ , for all

$s \in \mathcal { S }$ , illustrating policy improvement. Although in this case the new policy $\pi ^ { \prime }$ happensto be optimal, in general only an improvement is guaranteed.

# 4.3 Policy Iteration

Once a policy, $\pi$ , has been improved using $v _ { \pi }$ to yield a better policy, $\pi ^ { \prime }$ , we can thencompute $v _ { \pi ^ { \prime } }$ and improve it again to yield an even better $\pi ^ { \prime \prime }$ . We can thus obtain asequence of monotonically improving policies and value functions:

$$
\pi_ {0} \xrightarrow {\mathrm {E}} v _ {\pi_ {0}} \xrightarrow {\mathrm {I}} \pi_ {1} \xrightarrow {\mathrm {E}} v _ {\pi_ {1}} \xrightarrow {\mathrm {I}} \pi_ {2} \xrightarrow {\mathrm {E}} \dots \xrightarrow {\mathrm {I}} \pi_ {*} \xrightarrow {\mathrm {E}} v _ {*},
$$

where $\xrightarrow { \textrm { \textbf { E } } }$ denotes a policy evaluation and $\xrightarrow { \textrm { I } }$ denotes a policy improvement. Eachpolicy is guaranteed to be a strict improvement over the previous one (unless it is alreadyoptimal). Because a finite MDP has only a finite number of deterministic policies, thisprocess must converge to an optimal policy and the optimal value function in a finitenumber of iterations.

This way of finding an optimal policy is called policy iteration. A complete algorithm isgiven in the box below. Note that each policy evaluation, itself an iterative computation,is started with the value function for the previous policy. This typically results in a greatincrease in the speed of convergence of policy evaluation (presumably because the valuefunction changes little from one policy to the next).

# Policy Iteration (using iterative policy evaluation) for estimating ⇡ ⇡ ⇡⇤

1. Initialization

$V ( s ) \in \mathbb { R }$ and $\pi ( s ) \in { \mathcal { A } } ( s )$ arbitrarily for all $s \in \mathcal { S }$ ; $V ( t e r m i n a l ) \doteq 0$

2. Policy Evaluation

Loop:

$\Delta  0$

Loop for each $s \in \mathcal { S }$

$v  V ( s )$

$\begin{array} { r } { V ( s ) \gets \sum _ { s ^ { \prime } , r } p ( s ^ { \prime } , r | s , \pi ( s ) ) \big | r + \gamma V ( s ^ { \prime } ) \big | } \end{array}$

$\Delta \gets \operatorname* { m a x } ( \Delta , | v - V ( s ) | )$

until $\Delta < \theta$ (a small positive number determining the accuracy of estimation)

3. Policy Improvement

policy-stable true

For each $s \in \mathcal { S }$ :

old- $a c t i o n \gets \pi ( s )$

$\begin{array} { r } { \pi ( s ) \gets \mathrm { a r g m a x } _ { a } \sum _ { s ^ { \prime } , r } p ( s ^ { \prime } , r | s , a ) \lfloor r + \gamma V ( s ^ { \prime } ) \rfloor } \end{array}$

If ol $d - a c t i o n \neq \pi ( s )$ , then $\ p o l i c y - s t a b l e \gets f a l s e$

If policy-stable, then stop and return $V \approx v _ { * }$ and $\pi \approx \pi _ { * }$ ; else go to 2

Example 4.2: Jack’s Car Rental Jack manages two locations for a nationwide carrental company. Each day, some number of customers arrive at each location to rent cars.If Jack has a car available, he rents it out and is credited $\$ 10$ by the national company.If he is out of cars at that location, then the business is lost. Cars become available forrenting the day after they are returned. To help ensure that cars are available wherethey are needed, Jack can move them between the two locations overnight, at a cost of$\$ 2$ per car moved. We assume that the number of cars requested and returned at eachlocation are Poisson random variables, meaning that the probability that the number is$n$ is $\frac { \lambda ^ { \prime \prime } } { n ! } e ^ { - \lambda }$ , where $\lambda$ is the expected number. Suppose $\lambda$ is 3 and 4 for rental requests atthe first and second locations and 3 and 2 for returns. To simplify the problem slightly,we assume that there can be no more than 20 cars at each location (any additional carsare returned to the nationwide company, and thus disappear from the problem) and amaximum of five cars can be moved from one location to the other in one night. We takethe discount rate to be $\gamma = 0 . 9$ and formulate this as a continuing finite MDP, wherethe time steps are days, the state is the number of cars at each location at the end ofthe day, and the actions are the net numbers of cars moved between the two locationsovernight. Figure 4.2 shows the sequence of policies found by policy iteration startingfrom the policy that never moves any cars.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/801a259df45a29540fe6066ae462dbeee8893bd321011f411d6db743429f50be.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/6d6f79adaf21bef0ab8b239d2b6c162387ba95af74d7472e0073b94ef162c313.jpg)



Figure 4.2: The sequence of policies found by policy iteration on Jack’s car rental problem,and the final state-value function. The first five diagrams show, for each number of cars ateach location at the end of the day, the number of cars to be moved from the first location tothe second (negative numbers indicate transfers from the second location to the first). Eachsuccessive policy is a strict improvement over the previous policy, and the last policy is optimal. $-$


Policy iteration often converges in surprisingly few iterations, as illustrated in theexample of Jack’s car rental and in the example in Figure 4.1. The bottom-left diagram ofFigure 4.1 shows the value function for the equiprobable random policy, and the bottom-right diagram shows a greedy policy for this value function. The policy improvementtheorem assures us that these policies are better than the original random policy. In thiscase, however, these policies are not just better, but optimal, proceeding to the terminalstates in the minimum number of steps. In this example, policy iteration would find theoptimal policy after just one iteration.

Exercise 4.4 The policy iteration algorithm on page 80 has a subtle bug in that it maynever terminate if the policy continually switches between two or more policies that areequally good. This is okay for pedagogy, but not for actual use. Modify the pseudocodeso that convergence is guaranteed. ⇤

Exercise 4.5 How would policy iteration be defined for action values? Give a completealgorithm for computing $q _ { * }$ , analogous to that on page 80 for computing $v _ { * }$ . Please payspecial attention to this exercise, because the ideas involved will be used throughout therest of the book. ⇤

Exercise 4.6 Suppose you are restricted to considering only policies that are $\varepsilon$ -soft,meaning that the probability of selecting each action in each state, $s$ , is at least $\varepsilon / | { \mathcal { A } } ( s ) |$Describe qualitatively the changes that would be required in each of the steps 3, 2, and 1,in that order, of the policy iteration algorithm for $v _ { * }$ on page 80. ⇤

Exercise 4.7 (programming) Write a program for policy iteration and re-solve Jack’s carrental problem with the following changes. One of Jack’s employees at the first locationrides a bus home each night and lives near the second location. She is happy to shuttleone car to the second location for free. Each additional car still costs $\$ 2$ , as do all carsmoved in the other direction. In addition, Jack has limited parking space at each location.If more than 10 cars are kept overnight at a location (after any moving of cars), then anadditional cost of $4 must be incurred to use a second parking lot (independent of howmany cars are kept there). These sorts of nonlinearities and arbitrary dynamics oftenoccur in real problems and cannot easily be handled by optimization methods other thandynamic programming. To check your program, first replicate the results given for theoriginal problem. ⇤

# 4.4 Value Iteration

One drawback to policy iteration is that each of its iterations involves policy evaluation,which may itself be a protracted iterative computation requiring multiple sweeps throughthe state set. If policy evaluation is done iteratively, then convergence exactly to $v _ { \pi }$occurs only in the limit. Must we wait for exact convergence, or can we stop short ofthat? The example in Figure 4.1 certainly suggests that it may be possible to truncatepolicy evaluation. In that example, policy evaluation iterations beyond the first threehave no e↵ect on the corresponding greedy policy.

In fact, the policy evaluation step of policy iteration can be truncated in several wayswithout losing the convergence guarantees of policy iteration. One important special

case is when policy evaluation is stopped after just one sweep (one update of each state).This algorithm is called value iteration. It can be written as a particularly simple updateoperation that combines the policy improvement and truncated policy evaluation steps:

$$
\begin{array}{l} v _ {k + 1} (s) \quad \doteq \quad \max  _ {a} \mathbb {E} \left[ R _ {t + 1} + \gamma v _ {k} \left(S _ {t + 1}\right) \mid S _ {t} = s, A _ {t} = a \right] \\ = \max  _ {a} \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r \mid s, a\right) \left[ r + \gamma v _ {k} \left(s ^ {\prime}\right) \right], \tag {4.10} \\ \end{array}
$$

for all $s \in \mathcal { S }$ . For arbitrary $v _ { 0 }$ , the sequence $\{ v _ { k } \}$ can be shown to converge to $v _ { * }$ underthe same conditions that guarantee the existence of $v _ { * }$ .

Another way of understanding value iteration is by reference to the Bellman optimalityequation (4.1). Note that value iteration is obtained simply by turning the Bellmanoptimality equation into an update rule. Also note how the value iteration update isidentical to the policy evaluation update (4.5) except that it requires the maximum to betaken over all actions. Another way of seeing this close relationship is to compare thebackup diagrams for these algorithms on page 59 (policy evaluation) and on the left ofFigure 3.4 (value iteration). These two are the natural backup operations for computing$v _ { \pi }$ and $v _ { * }$ .

Finally, let us consider how value iteration terminates. Like policy evaluation, valueiteration formally requires an infinite number of iterations to converge exactly to $v _ { * }$ . Inpractice, we stop once the value function changes by only a small amount in a sweep.The box below shows a complete algorithm with this kind of termination condition.

# Value Iteration, for estimating ⇡ ⇡ ⇡

Algorithm parameter: a small threshold $\theta > 0$ determining accuracy of estimationInitialize $V ( s )$ , for all $s \in \mathcal { S } ^ { + }$ , arbitrarily except that $V ( t e r m i n a l ) = 0$

Loop:

$\begin{array}{rl} & {\Delta \gets 0}\\ & {\mathrm{Loop~for~each} s\in \mathcal{S}:}\\ & {v\gets V(s)}\\ & {V(s)\gets \max_a\sum_{s',r}p(s',r|s,a)[r + \gamma V(s')]}\\ & {\Delta \gets \max (\Delta ,|v - V(s)|)}\\ & {\mathrm{until}\Delta <  \theta} \end{array}$

Output a deterministic policy, $\pi \approx \pi _ { * }$ , such that

$$
\pi (s) = \operatorname {a r g m a x} _ {a} \sum_ {s ^ {\prime}, r} p (s ^ {\prime}, r | s, a) \left[ r + \gamma V (s ^ {\prime}) \right]
$$

Value iteration e↵ectively combines, in each of its sweeps, one sweep of policy evaluationand one sweep of policy improvement. Faster convergence is often achieved by interposingmultiple policy evaluation sweeps between each policy improvement sweep. In general,the entire class of truncated policy iteration algorithms can be thought of as sequencesof sweeps, some of which use policy evaluation updates and some of which use valueiteration updates. Because the max operation in (4.10) is the only di↵erence between

these updates, this just means that the max operation is added to some sweeps of policyevaluation. All of these algorithms converge to an optimal policy for discounted finiteMDPs.

Example 4.3: Gambler’s Problem A gambler has the opportunity to make bets on theoutcomes of a sequence of coin flips. If the coin comes up heads, he wins as many dollars ashe has staked on that flip; if it is tails, he loses his stake. The game ends when the gamblerwins by reaching his goal of $\$ 100$ , or loses by running out of money. On each flip, the gam-bler must decide what portion of his capital to stake, in integer numbers of dollars. Thisproblem can be formulated as an undiscounted, episodic, finite MDP. The state is the gam-bler’s capital, $s \in \{ 1 , 2 , \ldots , 9 9 \}$ and the actions are stakes, $a \in \{ 0 , 1 , \dots , \operatorname* { m i n } ( s , 1 0 0 - s ) \}$ .

The reward is zero on all transi-tions except those on which the gam-bler reaches his goal, when it is $+ 1$ .The state-value function then givesthe probability of winning from eachstate. A policy is a mapping fromlevels of capital to stakes. The opti-mal policy maximizes the probabilityof reaching the goal. Let $p _ { h }$ denotethe probability of the coin comingup heads. If $p _ { h }$ is known, then theentire problem is known and it canbe solved, for instance, by value iter-ation. Figure 4.3 shows the changein the value function over successivesweeps of value iteration, and thefinal policy found, for the case of$p _ { h } = 0 . 4$ . This policy is optimal, butnot unique. In fact, there is a wholefamily of optimal policies, all corre-sponding to ties for the argmax ac-tion selection with respect to the op-timal value function. Can you guesswhat the entire family looks like?

Exercise 4.8 Why does the optimal

policy for the gambler’s problem have such a curious form? In particular, for capital of 50it bets it all on one flip, but for capital of 51 it does not. Why is this a good policy? $\boxed { \begin{array} { r l } \end{array} }$

Exercise 4.9 (programming) Implement value iteration for the gambler’s problem andsolve it for $p _ { h } = 0 . 2 5$ and $p _ { h } = 0 . 5 5$ . In programming, you may find it convenient tointroduce two dummy states corresponding to termination with capital of 0 and 100,giving them values of 0 and 1 respectively. Show your results graphically, as in Figure 4.3.Are your results stable as $\theta  0$ ? ⇤

Exercise 4.10 What is the analog of the value iteration update (4.10) for action values,$q _ { k + 1 } ( s , a )$ ? ⇤

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/8c16e2044e99b67ed691166f3356c6aa8f0e89b28d53ddbd1200ae2a6c79d4de.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/f223974dcc9149a8f8eaecd98d37ec983b1a6a9563f26d9e1792ff1b7fe9ca55.jpg)



Figure 4.3: The solution to the gambler’s problem for$p _ { h } = 0 . 4$ . The upper graph shows the value functionfound by successive sweeps of value iteration. Thelower graph shows the final policy.


# 4.5 Asynchronous Dynamic Programming

A major drawback to the DP methods that we have discussed so far is that they involveoperations over the entire state set of the MDP, that is, they require sweeps of the stateset. If the state set is very large, then even a single sweep can be prohibitively expensive.For example, the game of backgammon has over $1 0 ^ { 2 0 }$ states. Even if we could performthe value iteration update on a million states per second, it would take over a thousandyears to complete a single sweep.

Asynchronous DP algorithms are in-place iterative DP algorithms that are not organizedin terms of systematic sweeps of the state set. These algorithms update the values ofstates in any order whatsoever, using whatever values of other states happen to beavailable. The values of some states may be updated several times before the values ofothers are updated once. To converge correctly, however, an asynchronous algorithmmust continue to update the values of all the states: it can’t ignore any state after somepoint in the computation. Asynchronous DP algorithms allow great flexibility in selectingstates to update.

For example, one version of asynchronous value iteration updates the value, in place, ofonly one state, $s _ { k }$ , on each step, $k$ , using the value iteration update (4.10). If $0 \leq \gamma < 1$ ,asymptotic convergence to $v _ { * }$ is guaranteed given only that all states occur in the sequence$\left\{ s _ { k } \right\}$ an infinite number of times (the sequence could even be random).1 Similarly, it ispossible to intermix policy evaluation and value iteration updates to produce a kind ofasynchronous truncated policy iteration. Although the details of this and other moreunusual DP algorithms are beyond the scope of this book, it is clear that a few di↵erentupdates form building blocks that can be used flexibly in a wide variety of sweepless DPalgorithms.

Of course, avoiding sweeps does not necessarily mean that we can get away with lesscomputation. It just means that an algorithm does not need to get locked into anyhopelessly long sweep before it can make progress improving a policy. We can try totake advantage of this flexibility by selecting the states to which we apply updates soas to improve the algorithm’s rate of progress. We can try to order the updates to letvalue information propagate from state to state in an e cient way. Some states may notneed their values updated as often as others. We might even try to skip updating somestates entirely if they are not relevant to optimal behavior. Some ideas for doing this arediscussed in Chapter 8.

Asynchronous algorithms also make it easier to intermix computation with real-timeinteraction. To solve a given MDP, we can run an iterative DP algorithm at the sametime that an agent is actually experiencing the MDP. The agent’s experience can be usedto determine the states to which the DP algorithm applies its updates. At the same time,the latest value and policy information from the DP algorithm can guide the agent’sdecision making. For example, we can apply updates to states as the agent visits them.This makes it possible to focus the DP algorithm’s updates onto parts of the state set

that are most relevant to the agent. This kind of focusing is a repeated theme inreinforcement learning.

# 4.6 Generalized Policy Iteration

Policy iteration consists of two simultaneous, interacting processes, one making the valuefunction consistent with the current policy (policy evaluation), and the other makingthe policy greedy with respect to the current value function (policy improvement). Inpolicy iteration, these two processes alternate, each completing before the other begins,but this is not really necessary. In value iteration, for example, only a single iterationof policy evaluation is performed in between each policy improvement. In asynchronousDP methods, the evaluation and improvement processes are interleaved at an evenfiner grain. In some cases a single state is updated in one process before returningto the other. As long as both processes continue to update all states, the ultimateresult is typically the same—convergence to the optimal value function and an optimalpolicy.

We use the term generalized policy iteration (GPI) to referto the general idea of letting policy-evaluation and policy-improvement processes interact, independent of the granularityand other details of the two processes. Almost all reinforcementlearning methods are well described as GPI. That is, all haveidentifiable policies and value functions, with the policy alwaysbeing improved with respect to the value function and the valuefunction always being driven toward the value function for thepolicy, as suggested by the diagram to the right. If both theevaluation process and the improvement process stabilize, thatis, no longer produce changes, then the value function and policymust be optimal. The value function stabilizes only when itis consistent with the current policy, and the policy stabilizesonly when it is greedy with respect to the current value function.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/ec3118926fc23e6ba0d7a08b1607e26c4560a383b6bc7eae9687f6fc7007284f.jpg)


Thus, both processes stabilize only when a policy has been found that is greedy withrespect to its own evaluation function. This implies that the Bellman optimality equation(4.1) holds, and thus that the policy and the value function are optimal.

The evaluation and improvement processes in GPI can be viewed as both competingand cooperating. They compete in the sense that they pull in opposing directions. Makingthe policy greedy with respect to the value function typically makes the value functionincorrect for the changed policy, and making the value function consistent with the policytypically causes that policy no longer to be greedy. In the long run, however, thesetwo processes interact to find a single joint solution: the optimal value function and anoptimal policy.

One might also think of the interaction between the evaluation and improvementprocesses in GPI in terms of two constraints or goals—for example, as two lines in

a two-dimensional space as suggested by the dia-gram to the right. Although the real geometry ismuch more complicated than this, the diagram sug-gests what happens in the real case. Each processdrives the value function or policy toward one ofthe lines representing a solution to one of the twogoals. The goals interact because the two lines arenot orthogonal. Driving directly toward one goalcauses some movement away from the other goal.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/c08d5498a8b54da8b461b295d73f6822ab7b19f3680c0e5feb9c8e70f7e76144.jpg)


Inevitably, however, the joint process is brought closer to the overall goal of optimality.The arrows in this diagram correspond to the behavior of policy iteration in that eachtakes the system all the way to achieving one of the two goals completely. In GPIone could also take smaller, incomplete steps toward each goal. In either case, the twoprocesses together achieve the overall goal of optimality even though neither is attemptingto achieve it directly.

# 4.7 E ciency of Dynamic Programming

DP may not be practical for very large problems, but compared with other methodsfor solving MDPs, DP methods are actually quite e cient. If we ignore a few technicaldetails, then, in the worst case, the time that DP methods take to find an optimal policyis polynomial in the number of states and actions. If $n$ and $k$ denote the number of statesand actions, this means that a DP method takes a number of computational operationsthat is less than some polynomial function of $n$ and $k$ . A DP method is guaranteed tofind an optimal policy in polynomial time even though the total number of (deterministic)policies is $k ^ { n }$ . In this sense, DP is exponentially faster than any direct search in policyspace could be, because direct search would have to exhaustively examine each policyto provide the same guarantee. Linear programming methods can also be used to solveMDPs, and in some cases their worst-case convergence guarantees are better than thoseof DP methods. But linear programming methods become impractical at a much smallernumber of states than do DP methods (by a factor of about 100). For the largest problems,only DP methods are feasible.

DP is sometimes thought to be of limited applicability because of the curse of dimen-sionality, the fact that the number of states often grows exponentially with the numberof state variables. Large state sets do create di culties, but these are inherent di cultiesof the problem, not of DP as a solution method. In fact, DP is comparatively bettersuited to handling large state spaces than competing methods such as direct search andlinear programming.

In practice, DP methods can be used with today’s computers to solve MDPs withmillions of states. Both policy iteration and value iteration are widely used, and it is notclear which, if either, is better in general. In practice, these methods usually convergemuch faster than their theoretical worst-case run times, particularly if they are startedwith good initial value functions or policies.

On problems with large state spaces, asynchronous DP methods are often preferred. Tocomplete even one sweep of a synchronous method requires computation and memory forevery state. For some problems, even this much memory and computation is impractical,yet the problem is still potentially solvable because relatively few states occur alongoptimal solution trajectories. Asynchronous methods and other variations of GPI can beapplied in such cases and may find good or optimal policies much faster than synchronousmethods can.

# 4.8 Summary

In this chapter we have become familiar with the basic ideas and algorithms of dynamicprogramming as they relate to solving finite MDPs. Policy evaluation refers to the (typi-cally) iterative computation of the value function for a given policy. Policy improvementrefers to the computation of an improved policy given the value function for that policy.Putting these two computations together, we obtain policy iteration and value iteration,the two most popular DP methods. Either of these can be used to reliably computeoptimal policies and value functions for finite MDPs given complete knowledge of theMDP.

Classical DP methods operate in sweeps through the state set, performing an expectedupdate operation on each state. Each such operation updates the value of one statebased on the values of all possible successor states and their probabilities of occurring.Expected updates are closely related to Bellman equations: they are little more thanthese equations turned into assignment statements. When the updates no longer result inany changes in value, convergence has occurred to values that satisfy the correspondingBellman equation. Just as there are four primary value functions ( $v _ { \pi }$ , $v _ { * }$ , $q _ { \pi }$ , and $q _ { * }$ ),there are four corresponding Bellman equations and four corresponding expected updates.An intuitive view of the operation of DP updates is given by their backup diagrams.

Insight into DP methods and, in fact, into almost all reinforcement learning methods,can be gained by viewing them as generalized policy iteration (GPI). GPI is the general ideaof two interacting processes revolving around an approximate policy and an approximatevalue function. One process takes the policy as given and performs some form of policyevaluation, changing the value function to be more like the true value function for thepolicy. The other process takes the value function as given and performs some formof policy improvement, changing the policy to make it better, assuming that the valuefunction is its value function. Although each process changes the basis for the other,overall they work together to find a joint solution: a policy and value function that areunchanged by either process and, consequently, are optimal. In some cases, GPI can beproved to converge, most notably for the classical DP methods that we have presented inthis chapter. In other cases convergence has not been proved, but still the idea of GPIimproves our understanding of the methods.

It is not necessary to perform DP methods in complete sweeps through the stateset. Asynchronous ${ D P }$ methods are in-place iterative methods that update states in anarbitrary order, perhaps stochastically determined and using out-of-date information.Many of these methods can be viewed as fine-grained forms of GPI.

Finally, we note one last special property of DP methods. All of them update estimatesof the values of states based on estimates of the values of successor states. That is, theyupdate estimates on the basis of other estimates. We call this general idea bootstrapping.Many reinforcement learning methods perform bootstrapping, even those that do notrequire, as DP requires, a complete and accurate model of the environment. In the nextchapter we explore reinforcement learning methods that do not require a model and donot bootstrap. In the chapter after that we explore methods that do not require a modelbut do bootstrap. These key features and properties are separable, yet can be mixed ininteresting combinations.

# Bibliographical and Historical Remarks

The term “dynamic programming” is due to Bellman (1957a), who showed how thesemethods could be applied to a wide range of problems. Extensive treatments of DP canbe found in many texts, including Bertsekas (2005, 2012), Bertsekas and Tsitsiklis (1996),Dreyfus and Law (1977), Ross (1983), White (1969), and Whittle (1982, 1983). Ourinterest in DP is restricted to its use in solving MDPs, but DP also applies to other typesof problems. Kumar and Kanal (1988) provide a more general look at DP.

To the best of our knowledge, the first connection between DP and reinforcementlearning was made by Minsky (1961) in commenting on Samuel’s checkers player. Ina footnote, Minsky mentioned that it is possible to apply DP to problems in whichSamuel’s backing-up process can be handled in closed analytic form. This remark mayhave misled artificial intelligence researchers into believing that DP was restricted toanalytically tractable problems and therefore largely irrelevant to artificial intelligence.Andreae (1969) mentioned DP in the context of reinforcement learning. Werbos (1977)suggested an approach to approximating DP called “heuristic dynamic programming”that emphasizes gradient-descent methods for continuous-state problems (Werbos, 1982,1987, 1988, 1989, 1992). These methods are closely related to the reinforcement learningalgorithms that we discuss in this book. Watkins (1989) was explicit in connectingreinforcement learning to DP, characterizing a class of reinforcement learning methods as“incremental dynamic programming.”

4.1–4 These sections describe well-established DP algorithms that are covered in any ofthe general DP references cited above. The policy improvement theorem and thepolicy iteration algorithm are due to Bellman (1957a) and Howard (1960). Ourpresentation was influenced by the local view of policy improvement taken byWatkins (1989). Our discussion of value iteration as a form of truncated policyiteration is based on the approach of Puterman and Shin (1978), who presented aclass of algorithms called modified policy iteration, which includes policy iterationand value iteration as special cases. An analysis showing how value iteration canbe made to find an optimal policy in finite time is given by Bertsekas (1987).

Iterative policy evaluation is an example of a classical successive approximationalgorithm for solving a system of linear equations. The version of the algorithm

that uses two arrays, one holding the old values while the other is updated, isoften called a Jacobi-style algorithm, after Jacobi’s classical use of this method.It is also sometimes called a synchronous algorithm because the e↵ect is as if allthe values are updated at the same time. The second array is needed to simulatethis parallel computation sequentially. The in-place version of the algorithmis often called a Gauss–Seidel-style algorithm after the classical Gauss–Seidelalgorithm for solving systems of linear equations. In addition to iterative policyevaluation, other DP algorithms can be implemented in these di↵erent versions.Bertsekas and Tsitsiklis (1989) provide excellent coverage of these variations andtheir performance di↵erences.

4.5 Asynchronous DP algorithms are due to Bertsekas (1982, 1983), who also calledthem distributed DP algorithms. The original motivation for asynchronousDP was its implementation on a multiprocessor system with communicationdelays between processors and no global synchronizing clock. These algorithmsare extensively discussed by Bertsekas and Tsitsiklis (1989). Jacobi-style andGauss–Seidel-style DP algorithms are special cases of the asynchronous version.Williams and Baird (1990) presented DP algorithms that are asynchronous at afiner grain than the ones we have discussed: the update operations themselvesare broken into steps that can be performed asynchronously.

4.7 This section, written with the help of Michael Littman, is based on Littman,Dean, and Kaelbling (1995). The phrase “curse of dimensionality” is due toBellman (1957a).

Foundational work on the linear programming approach to reinforcement learningwas done by Daniela de Farias (de Farias, 2002; de Farias and Van Roy, 2003).

