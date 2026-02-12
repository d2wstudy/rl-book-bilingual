# Chapter 7

# n-step Bootstrapping

In this chapter we unify the Monte Carlo (MC) methods and the one-step temporal-di↵erence (TD) methods presented in the previous two chapters. Neither MC methods norone-step TD methods are always the best. In this chapter we present $n$ -step TD methodsthat generalize both methods so that one can shift from one to the other smoothly asneeded to meet the demands of a particular task. $\boldsymbol { n }$ -step methods span a spectrum withMC methods at one end and one-step TD methods at the other. The best methods areoften intermediate between the two extremes.

Another way of looking at the benefits of $n$ -step methods is that they free you fromthe tyranny of the time step. With one-step TD methods the same time step determineshow often the action can be changed and the time interval over which bootstrappingis done. In many applications one wants to be able to update the action very fast totake into account anything that has changed, but bootstrapping works best if it is over alength of time in which a significant and recognizable state change has occurred. Withone-step TD methods, these time intervals are the same, and so a compromise must bemade. $n$ -step methods enable bootstrapping to occur over multiple steps, freeing us fromthe tyranny of the single time step.

The idea of $\textit { n }$ -step methods is usually used as an introduction to the algorithmicidea of eligibility traces (Chapter 12), which enable bootstrapping over multiple timeintervals simultaneously. Here we instead consider the $n$ -step bootstrapping idea on itsown, postponing the treatment of eligibility-trace mechanisms until later. This allows usto separate the issues better, dealing with as many of them as possible in the simpler$n$ -step setting.

As usual, we first consider the prediction problem and then the control problem. Thatis, we first consider how $n$ -step methods can help in predicting returns as a function ofstate for a fixed policy (i.e., in estimating $v _ { \pi }$ ). Then we extend the ideas to action valuesand control methods.

# 7.1 $\mathbf { \nabla } ^ { \prime } \mathbf { \mathit { n } } _ { \mathbf { \mathit { i } } }$ -step TD Prediction

What is the space of methods lying between Monte Carlo and TD methods? Considerestimating $v _ { \pi }$ from sample episodes generated using $\pi$ . Monte Carlo methods performan update for each state based on the entire sequence of observed rewards from thatstate until the end of the episode. The update of one-step TD methods, on the otherhand, is based on just the one next reward, bootstrapping from the value of the stateone step later as a proxy for the remaining rewards. One kind of intermediate method,then, would perform an update based on an intermediate number of rewards: more thanone, but less than all of them until termination. For example, a two-step update wouldbe based on the first two rewards and the estimated value of the state two steps later.Similarly, we could have three-step updates, four-step updates, and so on. Figure 7.1shows the backup diagrams of the spectrum of $n$ -step updates for $v _ { \pi }$ , with the one-stepTD update on the left and the up-until-termination Monte Carlo update on the right.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/0337feb0f0099de25d9ac471e509d5c7c6540a7e02d44bca7c148f2c3c422bb3.jpg)



Figure 7.1: The backup diagrams of $\textit { n }$ -step methods. These methods form a spectrum rangingfrom one-step TD methods to Monte Carlo methods.


The methods that use $n$ -step updates are still TD methods because they still changean earlier estimate based on how it di↵ers from a later estimate. Now the later estimateis not one step later, but $n$ steps later. Methods in which the temporal di↵erence extendsover $n$ steps are called $\boldsymbol { n }$ -step TD methods. The TD methods introduced in the previouschapter all used one-step updates, which is why we called them one-step TD methods.

More formally, consider the update of the estimated value of state $S _ { t }$ as a result of thestate–reward sequence, $S _ { t } , R _ { t + 1 } , S _ { t + 1 } , R _ { t + 2 } , \ldots , R _ { T } , S _ { T }$ (omitting the actions). We knowthat in Monte Carlo updates the estimate of $v _ { \pi } ( S _ { t } )$ is updated in the direction of the

complete return:

$$
G _ {t} \doteq R _ {t + 1} + \gamma R _ {t + 2} + \gamma^ {2} R _ {t + 3} + \dots + \gamma^ {T - t - 1} R _ {T},
$$

where $T$ is the last time step of the episode. Let us call this quantity the target of theupdate. Whereas in Monte Carlo updates the target is the return, in one-step updatesthe target is the first reward plus the discounted estimated value of the next state, whichwe call the one-step return:

$$
G _ {t: t + 1} \doteq R _ {t + 1} + \gamma V _ {t} (S _ {t + 1}),
$$

where $V _ { t } : \mathcal { S }  \mathbb { R }$ here is the estimate at time $t$ of $v _ { \pi }$ . The subscripts on $G _ { t : t + 1 }$ indicatethat it is a truncated return for time $t$ using rewards up until time $t { + } 1$ , with the discountedestimate $\gamma V _ { t } ( S _ { t + 1 } )$ taking the place of the other terms $\gamma R _ { t + 2 } + \gamma ^ { 2 } R _ { t + 3 } + \cdot \cdot \cdot + \gamma ^ { T - t - 1 } R _ { T }$of the full return, as discussed in the previous chapter. Our point now is that this ideamakes just as much sense after two steps as it does after one. The target for a two-stepupdate is the two-step return:

$$
G _ {t: t + 2} \doteq R _ {t + 1} + \gamma R _ {t + 2} + \gamma^ {2} V _ {t + 1} \left(S _ {t + 2}\right),
$$

where now $\gamma ^ { 2 } V _ { t + 1 } ( S _ { t + 2 } )$ corrects for the absence of the terms $\gamma ^ { 2 } R _ { t + 3 } + \gamma ^ { 3 } R _ { t + 4 } + \cdot \cdot \cdot +$$\gamma ^ { T - t - 1 } R _ { T }$ . Similarly, the target for an arbitrary $\boldsymbol { n }$ -step update is the $\boldsymbol { n }$ -step return:

$$
G _ {t: t + n} \dot {=} R _ {t + 1} + \gamma R _ {t + 2} + \dots + \gamma^ {n - 1} R _ {t + n} + \gamma^ {n} V _ {t + n - 1} \left(S _ {t + n}\right), \tag {7.1}
$$

for all $n , t$ such that $n \geq 1$ and $0 \leq t < T - n$ . All $\textit { n }$ -step returns can be consideredapproximations to the full return, truncated after $n$ steps and then corrected for theremaining missing terms by $V _ { t + n - 1 } ( S _ { t + n } )$ . If $t + n \ge T$ (if the $n$ -step return extendsto or beyond termination), then all the missing terms are taken as zero, and the $\boldsymbol { n }$ -stepreturn defined to be equal to the ordinary full return ( $G _ { t : t + n } \doteq G _ { t }$ if $t + n \geq T$ ).

Note that $n$ -step returns for $n > 1$ involve future rewards and states that are notavailable at the time of transition from $t$ to $t + 1$ . No real algorithm can use the $\boldsymbol { n }$ -stepreturn until after it has seen $R _ { t + n }$ and computed $V _ { t + n - 1 }$ . The first time these areavailable is $t + n$ . The natural state-value learning algorithm for using $\boldsymbol { n }$ -step returns isthus

$$
V _ {t + n} \left(S _ {t}\right) \doteq V _ {t + n - 1} \left(S _ {t}\right) + \alpha \left[ G _ {t: t + n} - V _ {t + n - 1} \left(S _ {t}\right) \right], \quad 0 \leq t <   T, \tag {7.2}
$$

while the values of all other states remain unchanged: $V _ { t + n } ( s ) = V _ { t + n - 1 } ( s )$ , for all $s \neq S _ { t }$ .We call this algorithm $n$ -step TD. Note that no changes at all are made during the first$n - 1$ steps of each episode. To make up for that, an equal number of additional updatesare made at the end of the episode, after termination and before starting the next episode.Complete pseudocode is given in the box on the next page.

Exercise 7.1 In Chapter 6 we noted that the Monte Carlo error can be written as thesum of TD errors (6.6) if the value estimates don’t change from step to step. Show thatthe $n$ -step error used in (7.2) can also be written as a sum of TD errors (again if thevalue estimates don’t change) generalizing the earlier result. ⇤

Exercise 7.2 (programming) With an $\boldsymbol { n }$ -step method, the value estimates do change fromstep to step, so an algorithm that used the sum of TD errors (see previous exercise) in


n-step TD for estimating V ⇡ v⇡


Input: a policy  $\pi$    
Algorithm parameters: step size  $\alpha \in (0,1]$  , a positive integer  $n$    
Initialize  $V(s)$  arbitrarily, for all  $s\in S$    
All store and access operations (for  $S_{t}$  and  $R_{t}$  ) can take their index mod  $n + 1$    
Loop for each episode: Initialize and store  $S_0\neq$  terminal  $T\gets \infty$  Loop for  $t = 0,1,2,\ldots$  . If  $t <   T$  , then: Take an action according to  $\pi (\cdot |S_t)$  Observe and store the next reward as  $R_{t + 1}$  and the next state as  $S_{t + 1}$  If  $S_{t + 1}$  is terminal, then  $T\gets t + 1$ $\tau \leftarrow t - n + 1$  (  $\tau$  is the time whose state's estimate is being updated) If  $\tau \geq 0$  ..  $G\gets \sum_{i = \tau +1}^{\min (\tau +n,T)}\gamma^{i - \tau -1}R_i$  If  $\tau +n <   T$  , then:  $G\gets G + \gamma^n V(S_{\tau +n})$ $(G_{\tau :\tau +n})$ $V(S_{\tau})\gets V(S_{\tau}) + \alpha [G - V(S_{\tau})]$    
Until  $\tau = T - 1$

place of the error in (7.2) would actually be a slightly di↵erent algorithm. Would it be abetter algorithm or a worse one? Devise and program a small experiment to answer thisquestion empirically. ⇤

The $n$ -step return uses the value function $V _ { t + n - 1 }$ to correct for the missing rewardsbeyond $R _ { t + n }$ . An important property of $n$ -step returns is that their expectation isguaranteed to be a better estimate of $v _ { \pi }$ than $V _ { t + n - 1 }$ is, in a worst-state sense. That is,the worst error of the expected $n$ -step return is guaranteed to be less than or equal to $\gamma ^ { \pi }$times the worst error under $V _ { t + n - 1 }$ :

$$
\left. \max  _ {s} \left| \mathbb {E} _ {\pi} \left[ G _ {t: t + n} \mid S _ {t} = s \right] - v _ {\pi} (s) \right| \leq \gamma^ {n} \max  _ {s} \left| V _ {t + n - 1} (s) - v _ {\pi} (s) \right|, \right. \tag {7.3}
$$

for all $n \geq 1$ . This is called the error reduction property of $n$ -step returns. Because of theerror reduction property, one can show formally that all $\boldsymbol { n }$ -step TD methods converge tothe correct predictions under appropriate technical conditions. The $\boldsymbol { n }$ -step TD methodsthus form a family of sound methods, with one-step TD methods and Monte Carlomethods as extreme members.

Example 7.1: $\mathbf { \nabla } ^ { \prime } n _ { \mathbf { \mu } }$ -step TD Methods on the Random Walk Consider using $n$ -stepTD methods on the 5-state random walk task described in Example 6.2 (page 125).Suppose the first episode progressed directly from the center state, C, to the right,through $\mathsf { D }$ and $\mathsf { E }$ , and then terminated on the right with a return of 1. Recall that theestimated values of all the states started at an intermediate value, $V ( s ) = 0 . 5$ . As a resultof this experience, a one-step method would change only the estimate for the last state,

$V ( \mathsf E )$ , which would be incremented toward 1, the observed return. A two-step method,on the other hand, would increment the values of the two states preceding termination:$V ( \mathsf { D } )$ and $V ( \mathsf E )$ both would be incremented toward 1. A three-step method, or any $n$ -stepmethod for $n > 2$ , would increment the values of all three of the visited states toward 1,all by the same amount.

Which value of $n$ is better? Figure 7.2 shows the results of a simple empirical test fora larger random walk process, with 19 states instead of 5 (and with a $- 1$ outcome on theleft, all values initialized to $0$ ), which we use as a running example in this chapter. Resultsare shown for $n$ -step TD methods with a range of values for $n$ and $\alpha$ . The performancemeasure for each parameter setting, shown on the vertical axis, is the square-root ofthe average squared error between the predictions at the end of the episode for the 19states and their true values, then averaged over the first 10 episodes and 100 repetitionsof the whole experiment (the same sets of walks were used for all parameter settings).Note that methods with an intermediate value of $n$ worked best. This illustrates howthe generalization of TD and Monte Carlo methods to $n$ -step methods can potentiallyperform better than either of the two extreme methods.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/dc3c46a1526f50c0c1a77637d47172d7d52546a205d46589cc2d5b18d6026ce4.jpg)



Figure 7.2: Performance of $n$ -step TD methods as a function of $\alpha$ , for various values of $n$ , ona 19-state random walk task (Example 7.1).


Exercise 7.3 Why do you think a larger random walk task (19 states instead of 5) wasused in the examples of this chapter? Would a smaller walk have shifted the advantageto a di↵erent value of $n$ ? How about the change in left-side outcome from 0 to $^ { - 1 }$ madein the larger walk? Do you think that made any di↵erence in the best value of $n$ ? ⇤

# 7.2 n-step Sarsa

How can $n$ -step methods be used not just for prediction, but for control? In this sectionwe show how $\textit { n }$ -step methods can be combined with Sarsa in a straightforward way to

produce an on-policy TD control method. The $n$ -step version of Sarsa we call $n$ -stepSarsa, and the original version presented in the previous chapter we henceforth callone-step Sarsa, or Sarsa(0).

The main idea is to simply switch states for actions (state–action pairs) and then usean $\varepsilon$ -greedy policy. The backup diagrams for $\textit { n }$ -step Sarsa (shown in Figure 7.3), likethose of $n$ -step TD (Figure 7.1), are strings of alternating states and actions, except thatthe Sarsa ones all start and end with an action rather a state. We redefine $n$ -step returns(update targets) in terms of estimated action values:

$$
G _ {t: t + n} \doteq R _ {t + 1} + \gamma R _ {t + 2} + \dots + \gamma^ {n - 1} R _ {t + n} + \gamma^ {n} Q _ {t + n - 1} \left(S _ {t + n}, A _ {t + n}\right), \quad n \geq 1, 0 \leq t <   T - n, \tag {7.4}
$$

with $G _ { t : t + n } \doteq G _ { t }$ if $t + n \geq T$ . The natural algorithm is then

$$
Q _ {t + n} \left(S _ {t}, A _ {t}\right) \doteq Q _ {t + n - 1} \left(S _ {t}, A _ {t}\right) + \alpha \left[ G _ {t: t + n} - Q _ {t + n - 1} \left(S _ {t}, A _ {t}\right) \right], \quad 0 \leq t <   T, \tag {7.5}
$$

while the values of all other states remain unchanged: $Q _ { t + n } ( s , a ) = Q _ { t + n - 1 } ( s , a )$ , for all$s , a$ such that $s \neq S _ { t }$ or $a \neq A _ { t }$ . This is the algorithm we call $n$ -step Sarsa. Pseudocodeis shown in the box on the next page, and an example of why it can speed up learningcompared to one-step methods is given in Figure 7.4.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/7a33cd018abaa5c63b5c5154550d61633637528ef4c3f96edf2585a30c56be50.jpg)



Figure 7.3: The backup diagrams for the spectrum of $n$ -step methods for state–action values.They range from the one-step update of Sarsa(0) to the up-until-termination update of theMonte Carlo method. In between are the $\textit { n }$ -step updates, based on $n$ steps of real rewards andthe estimated value of the $_ { n }$ th next state–action pair, all appropriately discounted. On the farright is the backup diagram for $\textit { n }$ -step Expected Sarsa.



n-step Sarsa for estimating Q ⇡ q⇤ or q⇡


Initialize  $Q(s,a)$  arbitrarily, for all  $s\in \mathcal{S},a\in \mathcal{A}$  Initialize  $\pi$  to be  $\varepsilon$  greedy with respect to  $Q$  , or to a fixed given policy Algorithm parameters: step size  $\alpha \in (0,1]$  small  $\varepsilon >0$  , a positive integer  $n$  All store and access operations (for  $S_{t}$ $A_{t}$  ,and  $R_{t}$  ) can take their index mod  $n + 1$  Loop for each episode: Initialize and store  $S_0\neq$  terminal Select and store an action  $A_0\sim \pi (\cdot |S_0)$ $T\gets \infty$  Loop for  $t = 0,1,2,\ldots$  If  $t <   T$  , then: Take action  $A_{t}$  Observe and store the next reward as  $R_{t + 1}$  and the next state as  $S_{t + 1}$  If  $S_{t + 1}$  is terminal, then:  $T\gets t + 1$  else: Select and store an action  $A_{t + 1}\sim \pi (\cdot |S_{t + 1})$ $\tau \leftarrow t - n + 1$  (  $\tau$  is the time whose estimate is being updated) If  $\tau \geq 0$  .  $G\gets \sum_{i = \tau +1}^{\min (\tau +n,T)}\gamma^{i - \tau -1}R_i$  If  $\tau +n <   T$  , then  $G\gets G + \gamma^n Q(S_{\tau +n},A_{\tau +n})$ $(G_{\tau :\tau +n})$ $Q(S_{\tau},A_{\tau})\gets Q(S_{\tau},A_{\tau}) + \alpha [G - Q(S_{\tau},A_{\tau})]$  If  $\pi$  is being learned, then ensure that  $\pi (\cdot |\bar{S}_{\tau})$  is  $\varepsilon$  -greedy wrt  $Q$  Until  $\tau = T - 1$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/959aa84afa0c52ca9d998de2852c24f1c34c8fea91658c8afc8bc672be6d14d4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/59b93a1408785562ec7af3798d8ae0c319e348d6521c0646effd436d921f576c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/2db916ab15974af02bcb5f7819ce8b1788932e3f95b8cb3f69902d866d6712c1.jpg)



Figure 7.4: Gridworld example of the speedup of policy learning due to the use of $n$ -stepmethods. The first panel shows the path taken by an agent in a single episode, ending at alocation of high reward, marked by the $\mathsf { G }$ . In this example the values were all initially 0, and allrewards were zero except for a positive reward at $\mathsf { G }$ . The arrows in the other two panels showwhich action values were strengthened as a result of this path by one-step and $n$ -step Sarsamethods. The one-step method strengthens only the last action of the sequence of actions thatled to the high reward, whereas the $n$ -step method strengthens the last $n$ actions of the sequence,so that much more is learned from the one episode.


Exercise 7.4 Prove that the $n$ -step return of Sarsa (7.4) can be written exactly in termsof a novel TD error, as

$$
G _ {t: t + n} = Q _ {t - 1} \left(S _ {t}, A _ {t}\right) + \sum_ {k = t} ^ {\min  (t + n, T) - 1} \gamma^ {k - t} \left[ R _ {k + 1} + \gamma Q _ {k} \left(S _ {k + 1}, A _ {k + 1}\right) - Q _ {k - 1} \left(S _ {k}, A _ {k}\right) \right]. \tag {7.6}
$$

⇤

What about Expected Sarsa? The backup diagram for the $\boldsymbol { n }$ -step version of ExpectedSarsa is shown on the far right in Figure 7.3. It consists of a linear string of sampleactions and states, just as in $n$ -step Sarsa, except that its last element is a branch overall action possibilities weighted, as always, by their probability under $\pi$ . This algorithmcan be described by the same equation as $n$ -step Sarsa (above) except with the $n$ -stepreturn redefined as

$$
G _ {t: t + n} \dot {=} R _ {t + 1} + \dots + \gamma^ {n - 1} R _ {t + n} + \gamma^ {n} \bar {V} _ {t + n - 1} \left(S _ {t + n}\right), \quad t + n <   T, \tag {7.7}
$$

(with $G _ { t : t + n } \dot { = } G _ { t }$ for $t + n \geq T$ ) where $V _ { t } ( s )$ is the expected approximate value of state $s$ ,using the estimated action values at time $t$ , under the target policy:

$$
\bar {V} _ {t} (s) \doteq \sum_ {a} \pi (a | s) Q _ {t} (s, a), \quad \text {f o r a l l} s \in \mathbb {S}. \tag {7.8}
$$

Expected approximate values are used in developing many of the action-value methodsin the rest of this book. If $s$ is terminal, then its expected approximate value is definedto be 0.

# 7.3 $\mathbf { \nabla } ^ { \prime } \mathbf { \mathit { n } } _ { \mathbf { \mathit { i } } }$ -step O↵-policy Learning

Recall that o↵-policy learning is learning the value function for one policy, $\pi$ , whilefollowing another policy, $b$ . Often, $\pi$ is the greedy policy for the current action-value-function estimate, and $b$ is a more exploratory policy, perhaps $\varepsilon$ -greedy. In order touse the data from $b$ we must take into account the di↵erence between the two policies,using their relative probability of taking the actions that were taken (see Section 5.5). In$n$ -step methods, returns are constructed over $n$ steps, so we are interested in the relativeprobability of just those $n$ actions. For example, to make a simple o↵-policy version of$n$ -step TD, the update for time $t$ (actually made at time $t + n$ ) can simply be weightedby $\rho _ { t : t + n - 1 }$ :

$$
V _ {t + n} \left(S _ {t}\right) \doteq V _ {t + n - 1} \left(S _ {t}\right) + \alpha \rho_ {t: t + n - 1} \left[ G _ {t: t + n} - V _ {t + n - 1} \left(S _ {t}\right) \right], \quad 0 \leq t <   T, \tag {7.9}
$$

where $\rho _ { t : t + n - 1 }$ , called the importance sampling ratio, is the relative probability underthe two policies of taking the $n$ actions from $A _ { t }$ to $A _ { t + n - 1 }$ (cf. Eq. 5.3):

$$
\rho_ {t: h} \doteq \prod_ {k = t} ^ {\min  (h, T - 1)} \frac {\pi \left(A _ {k} \mid S _ {k}\right)}{b \left(A _ {k} \mid S _ {k}\right)}. \tag {7.10}
$$

For example, if any one of the actions would never be taken by $\pi$ (i.e., $\pi ( A _ { k } | S _ { k } ) = 0$ ) thenthe $n$ -step return should be given zero weight and be totally ignored. On the other hand,if by chance an action is taken that $\pi$ would take with much greater probability than $b$ does, then this will increase the weight that would otherwise be given to the return. Thismakes sense because that action is characteristic of $\pi$ (and therefore we want to learnabout it) but is selected only rarely by $b$ and thus rarely appears in the data. To makeup for this we have to over-weight it when it does occur. Note that if the two policiesare actually the same (the on-policy case) then the importance sampling ratio is always1. Thus our new update (7.9) generalizes and can completely replace our earlier $n$ -stepTD update. Similarly, our previous $\boldsymbol { n }$ -step Sarsa update can be completely replaced by asimple o↵-policy form:

$$
Q _ {t + n} \left(S _ {t}, A _ {t}\right) \doteq Q _ {t + n - 1} \left(S _ {t}, A _ {t}\right) + \alpha \rho_ {t + 1: t + n} \left[ G _ {t: t + n} - Q _ {t + n - 1} \left(S _ {t}, A _ {t}\right) \right], \tag {7.11}
$$

for $0 \leq t < T$ . Note that the importance sampling ratio here starts and ends one steplater than for $\textit { n }$ -step TD (7.9). This is because here we are updating a state–actionpair. We do not have to care how likely we were to select the action; now that we haveselected it we want to learn fully from what happens, with importance sampling only forsubsequent actions. Pseudocode for the full algorithm is shown in the box below.


O↵-policy n-step Sarsa for estimating Q ⇡ q⇤ or q⇡


Input: an arbitrary behavior policy  $b$  such that  $b(a|s) > 0$ , for all  $s \in \mathcal{S}, a \in \mathcal{A}$   
Initialize  $Q(s, a)$  arbitrarily, for all  $s \in \mathcal{S}, a \in \mathcal{A}$   
Initialize  $\pi$  to be greedy with respect to  $Q$ , or as a fixed given policy  
Algorithm parameters: step size  $\alpha \in (0,1]$ , a positive integer  $n$   
All store and access operations (for  $S_t, A_t, \text{and } R_t$ ) can take their index mod  $n + 1$   
Loop for each episode:  
Initialize and store  $S_0 \neq$  terminal  
Select and store an action  $A_0 \sim b(\cdot | S_0)$ $T \leftarrow \infty$   
Loop for  $t = 0,1,2,\ldots$ :  
If  $t < T$ , then:  
Take action  $A_t$   
Observe and store the next reward as  $R_{t+1}$  and the next state as  $S_{t+1}$   
If  $S_{t+1}$  is terminal, then:  
 $T \leftarrow t + 1$   
else:  
Select and store an action  $A_{t+1} \sim b(\cdot | S_{t+1})$ $\tau \leftarrow t - n + 1$  ( $\tau$  is the time whose estimate is being updated)  
If  $\tau \geq 0$ :  
 $\rho \leftarrow \prod_{i=\tau+1}^{\min(\tau+n, T-1)} \frac{\pi(A_i | S_i)}{b(A_i | S_i)}$  ( $\rho_{\tau+1:\tau+n}$ )  
 $G \leftarrow \sum_{i=\tau+1}^{\min(\tau+n, T)} \gamma^{i-\tau-1} R_i$   
If  $\tau + n < T$ , then:  $G \leftarrow G + \gamma^n Q(S_{\tau+n}, A_{\tau+n})$  ( $G_{\tau:\tau+n}$ )  
 $Q(S_\tau, A_\tau) \leftarrow Q(S_\tau, A_\tau) + \alpha \rho [G - Q(S_\tau, A_\tau)]$   
If  $\pi$  is being learned, then ensure that  $\pi(\cdot | S_\tau)$  is greedy wrt  $Q$   
Until  $\tau = T - 1$

The o↵-policy version of $\boldsymbol { n }$ -step Expected Sarsa would use the same update as abovefor $n$ -step Sarsa except that the importance sampling ratio would have one less factor init. That is, the above equation would use $\rho _ { t + 1 : t + n - 1 }$ instead of $\rho _ { t + 1 : t + n }$ , and of courseit would use the Expected Sarsa version of the $\textit { n }$ -step return (7.7). This is because inExpected Sarsa all possible actions are taken into account in the last state; the oneactually taken has no e↵ect and does not have to be corrected for.

# 7.4 *Per-decision Methods with Control Variates

The multi-step o↵-policy methods presented in the previous section are simple andconceptually clear, but are probably not the most e cient. A more sophisticated approachwould use per-decision importance sampling ideas such as were introduced in Section 5.9.To understand this approach, first note that the ordinary $\textit { n }$ -step return (7.1), like allreturns, can be written recursively. For the $n$ steps ending at horizon $h$ , the $n$ -step returncan be written

$$
G _ {t: h} = R _ {t + 1} + \gamma G _ {t + 1: h}, \quad t <   h <   T, \tag {7.12}
$$

where $G _ { h : h } \doteq V _ { h - 1 } ( S _ { h } )$ . (Recall that this return is used at time $h$ , previously denoted$t + n$ .) Now consider the e↵ect of following a behavior policy $b$ that is not the sameas the target policy $\pi$ . All of the resulting experience, including the first reward $R _ { t + 1 }$and the next state $S _ { t + 1 }$ , must be weighted by the importance sampling ratio for time $t$ ,⇢t = ⇡(At|St)b(At|St) . $\begin{array} { r } { \rho _ { t } = \frac { \pi ( A _ { t } | S _ { t } ) } { b ( A _ { t } | S _ { t } ) } } \end{array}$ One might be tempted to simply weight the righthand side of the aboveequation, but one can do better. Suppose the action at time $t$ would never be selected by$\pi$ , so that $\rho _ { t }$ is zero. Then a simple weighting would result in the $\textit { n }$ -step return beingzero, which could result in high variance when it was used as a target. Instead, in thismore sophisticated approach, one uses an alternate, o↵-policy definition of the $n$ -stepreturn ending at horizon $h$ , as

$$
G _ {t: h} \doteq \rho_ {t} \left(R _ {t + 1} + \gamma G _ {t + 1: h}\right) + \left(1 - \rho_ {t}\right) V _ {h - 1} \left(S _ {t}\right), \quad t <   h <   T, \tag {7.13}
$$

where again $G _ { h : h } \doteq V _ { h - 1 } ( S _ { h } )$ . In this approach, if $\rho _ { t }$ is zero, then instead of the targetbeing zero and causing the estimate to shrink, the target is the same as the estimate andcauses no change. The importance sampling ratio being zero means we should ignore thesample, so leaving the estimate unchanged seems appropriate. The second, additionalterm in (7.13) is called a control variate (for obscure reasons). Notice that the controlvariate does not change the expected update; the importance sampling ratio has expectedvalue one (Section 5.9) and is uncorrelated with the estimate, so the expected valueof the control variate is zero. Also note that the o↵-policy definition (7.13) is a strictgeneralization of the earlier on-policy definition of the $n$ -step return (7.1), as the two areidentical in the on-policy case, in which $\rho _ { t }$ is always 1.

For a conventional $n$ -step method, the learning rule to use in conjunction with (7.13)is the $\textit { n }$ -step TD update (7.2), which has no explicit importance sampling ratios otherthan those embedded in the return.

Exercise 7.5 Write the pseudocode for the o↵-policy state-value prediction algorithmdescribed above. ⇤

For action values, the o↵-policy definition of the $\textit { n }$ -step return is a little di↵erentbecause the first action does not play a role in the importance sampling. That first actionis the one being learned; it does not matter if it was unlikely or even impossible under thetarget policy—it has been taken and now full unit weight must be given to the rewardand state that follows it. Importance sampling will apply only to the actions that followit.

First note that for action values the $\textit { n }$ -step on-policy return ending at horizon $h$ ,expectation form (7.7), can be written recursively just as in (7.12), except that for actionvalues the recursion ends with $G _ { h : h } \doteq V _ { h - 1 } ( S _ { h } )$ as in (7.8). An o↵-policy form withcontrol variates is

$$
\begin{array}{l} G _ {t: h} \doteq R _ {t + 1} + \gamma \Big (\rho_ {t + 1} G _ {t + 1: h} + \bar {V} _ {h - 1} (S _ {t + 1}) - \rho_ {t + 1} Q _ {h - 1} (S _ {t + 1}, A _ {t + 1}) \Big), \\ = R _ {t + 1} + \gamma \rho_ {t + 1} \left(G _ {t + 1: h} - Q _ {h - 1} \left(S _ {t + 1}, A _ {t + 1}\right)\right) + \gamma \bar {V} _ {h - 1} \left(S _ {t + 1}\right), \quad t <   h \leq T. \tag {7.14} \\ \end{array}
$$

If $h \ < \ T$ , then the recursion ends with $G _ { h : h } \doteq Q _ { h - 1 } ( S _ { h } , A _ { h } )$ , whereas, if $h \geq T$ ,the recursion ends with and $G _ { T - 1 : h } \doteq R _ { T }$ . The resultant prediction algorithm (aftercombining with (7.5)) is analogous to Expected Sarsa.

Exercise 7.6 Prove that the control variate in the above equations does not change theexpected value of the return. ⇤

⇤ Exercise 7.7 Write the pseudocode for the o↵-policy action-value prediction algorithmdescribed immediately above. Pay particular attention to the termination conditions forthe recursion upon hitting the horizon or the end of episode. ⇤

Exercise 7.8 Show that the general (o↵-policy) version of the $n$ -step return (7.13) canstill be written exactly and compactly as the sum of state-based TD errors (6.5) if theapproximate state value function does not change. ⇤

Exercise 7.9 Repeat the above exercise for the action version of the o↵-policy $n$ -stepreturn (7.14) and the Expected Sarsa TD error (the quantity in brackets in Equation 6.9).⇤

Exercise 7.10 (programming) Devise a small o↵-policy prediction problem and use it toshow that the o↵-policy learning algorithm using (7.13) and (7.2) is more data e cientthan the simpler algorithm using (7.1) and (7.9). ⇤

The importance sampling that we have used in this section, the previous section, andin Chapter 5, enables sound o↵-policy learning, but also results in high variance updates,forcing the use of a small step-size parameter and thereby causing learning to be slow. Itis probably inevitable that o↵-policy training is slower than on-policy training—after all,the data is less relevant to what is being learned. However, it is probably also true thatthese methods can be improved on. The control variates are one way of reducing thevariance. Another is to rapidly adapt the step sizes to the observed variance, as in theAutostep method (Mahmood, Sutton, Degris and Pilarski, 2012). Yet another promisingapproach is the invariant updates of Karampatziakis and Langford (2010) as extendedto TD by Tian (in preparation). The usage technique of Mahmood (2017; Mahmood

and Sutton, 2015) may also be part of the solution. In the next section we consider ano↵-policy learning method that does not use importance sampling.

# 7.5 O↵-policy Learning Without Importance Sampling:The $\mathbf { \nabla } ^ { \prime } \mathbf { \mathit { n } } _ { \mathbf { \mathit { i } } }$ -step Tree Backup Algorithm

Is o↵-policy learning possible without importance sampling? Q-learning and ExpectedSarsa from Chapter 6 do this for the one-step case, but is there a corresponding multi-stepalgorithm? In this section we present just such an $n$ -step method, called the tree-backupalgorithm.

The idea of the algorithm is suggested by the 3-step tree-backup backupdiagram shown to the right. Down the central spine and labeled in thediagram are three sample states and rewards, and two sample actions.These are the random variables representing the events occurring after theinitial state–action pair $S _ { t } , A _ { t }$ . Hanging o↵ to the sides of each state arethe actions that were not selected. (For the last state, all the actions areconsidered to have not (yet) been selected.) Because we have no sampledata for the unselected actions, we bootstrap and use the estimates oftheir values in forming the target for the update. This slightly extends theidea of a backup diagram. So far we have always updated the estimatedvalue of the node at the top of the diagram toward a target combiningthe rewards along the way (appropriately discounted) and the estimatedvalues of the nodes at the bottom. In the tree-backup update, the targetincludes all these things plus the estimated values of the dangling actionnodes hanging o↵ the sides, at all levels. This is why it is called a tree-backup update; it is an update from the entire tree of estimated actionvalues.

More precisely, the update is from the estimated action values of theleaf nodes of the tree. The action nodes in the interior, corresponding to

the actual actions taken, do not participate. Each leaf node contributes to the targetwith a weight proportional to its probability of occurring under the target policy $\pi$ . Thuseach first-level action $a$ contributes with a weight of $\pi ( a | S _ { t + 1 } )$ , except that the actionactually taken, $A _ { t + 1 }$ , does not contribute at all. Its probability, $\pi ( A _ { t + 1 } | S _ { t + 1 } )$ , is usedto weight all the second-level action values. Thus, each non-selected second-level action$a ^ { \prime }$ contributes with weight $\pi ( A _ { t + 1 } | S _ { t + 1 } ) \pi ( \boldsymbol { a } ^ { \prime } | S _ { t + 2 } )$ . Each third-level action contributeswith weight $\pi ( { \cal A } _ { t + 1 } | S _ { t + 1 } ) \pi ( { \cal A } _ { t + 2 } | S _ { t + 2 } ) \pi ( a ^ { \prime \prime } | S _ { t + 3 } )$ , and so on. It is as if each arrow to anaction node in the diagram is weighted by the action’s probability of being selected underthe target policy and, if there is a tree below the action, then that weight applies to allthe leaf nodes in the tree.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/f3503bc590fb8bfb861dfd89e15840a5b32082def14d1ccc206d60a1fa801daa.jpg)


We can think of the 3-step tree-backup update as consisting of 6 half-steps, alternatingbetween sample half-steps from an action to a subsequent state, and expected half-stepsconsidering from that state all possible actions with their probabilities of occurring underthe policy.

Now let us develop the detailed equations for the $\boldsymbol { n }$ -step tree-backup algorithm. Theone-step return (target) is the same as that of Expected Sarsa,

$$
G _ {t: t + 1} \doteq R _ {t + 1} + \gamma \sum_ {a} \pi (a | S _ {t + 1}) Q _ {t} \left(S _ {t + 1}, a\right), \tag {7.15}
$$

for $t < T - 1$ , and the two-step tree-backup return is

$$
\begin{array}{l} G _ {t: t + 2} \doteq R _ {t + 1} + \gamma \sum_ {a \neq A _ {t + 1}} \pi (a | S _ {t + 1}) Q _ {t + 1} (S _ {t + 1}, a) \\ + \gamma \pi (A _ {t + 1} | S _ {t + 1}) \left(R _ {t + 2} + \gamma \sum_ {a} \pi (a | S _ {t + 2}) Q _ {t + 1} (S _ {t + 2}, a)\right) \\ = R _ {t + 1} + \gamma \sum_ {a \neq A _ {t + 1}} \pi (a | S _ {t + 1}) Q _ {t + 1} (S _ {t + 1}, a) + \gamma \pi (A _ {t + 1} | S _ {t + 1}) G _ {t + 1: t + 2}, \\ \end{array}
$$

for $t < T - 2$ . The latter form suggests the general recursive definition of the tree-backup$n$ -step return:

$$
G _ {t: t + n} \doteq R _ {t + 1} + \gamma \sum_ {a \neq A _ {t + 1}} \pi (a | S _ {t + 1}) Q _ {t + n - 1} \left(S _ {t + 1}, a\right) + \gamma \pi \left(A _ {t + 1} \mid S _ {t + 1}\right) G _ {t + 1: t + n}, \tag {7.16}
$$

for $t < T - 1 , n \geq 2$ , with the $n = 1$ case handled by (7.15) except for $G _ { T - 1 : t + n } \doteq R _ { T }$ .This target is then used with the usual action-value update rule from $n$ -step Sarsa:

$$
Q _ {t + n} \left(S _ {t}, A _ {t}\right) \doteq Q _ {t + n - 1} \left(S _ {t}, A _ {t}\right) + \alpha \left[ G _ {t: t + n} - Q _ {t + n - 1} \left(S _ {t}, A _ {t}\right) \right],
$$

for $0 \leq t < T$ , while the values of all other state–action pairs remain unchanged:$Q _ { t + n } ( s , a ) = Q _ { t + n - 1 } ( s , a )$ , for all $s , a$ such that $s \neq S _ { t }$ or $a \neq A _ { t }$ . Pseudocode for thisalgorithm is shown in the box on the next page.

Exercise 7.11 Show that if the approximate action values are unchanging, then thetree-backup return (7.16) can be written as a sum of expectation-based TD errors:

$$
G _ {t: t + n} = Q \left(S _ {t}, A _ {t}\right) + \sum_ {k = t} ^ {\min  (t + n - 1, T - 1)} \delta_ {k} \prod_ {i = t + 1} ^ {k} \gamma \pi \left(A _ {i} \mid S _ {i}\right),
$$

where $\delta _ { t } \doteq R _ { t + 1 } + \gamma V _ { t } ( S _ { t + 1 } ) - Q ( S _ { t } , A _ { t } )$ and $V _ { t }$ is given by (7.8).


n-step Tree Backup for estimating Q ⇡ q⇤ or q⇡


Initialize  $Q(s,a)$  arbitrarily, for all  $s\in \mathcal{S},a\in \mathcal{A}$    
Initialize  $\pi$  to be greedy with respect to  $Q$  , or as a fixed given policy   
Algorithm parameters: step size  $\alpha \in (0,1]$  , a positive integer  $n$    
All store and access operations can take their index mod  $n + 1$    
Loop for each episode: Initialize and store  $S_0\neq$  terminal Choose an action  $A_{0}$  arbitrarily as a function of  $S_0$  ; Store  $A_0$ $T\gets \infty$  Loop for  $t = 0,1,2,\ldots$  If  $t <   T$  Take action  $A_{t}$  ; observe and store the next reward and state as  $R_{t + 1},S_{t + 1}$  If  $S_{t + 1}$  is terminal:  $T\gets t + 1$  else: Choose an action  $A_{t + 1}$  arbitrarily as a function of  $S_{t + 1}$  ; Store  $A_{t + 1}$ $\tau \leftarrow t + 1 - n\quad (\tau$  is the time whose estimate is being updated) If  $\tau \geq 0$  If  $t + 1\geq T$  .  $G\gets R_T$  else  $G\gets R_{t + 1} + \gamma \sum_{a}\pi (a|S_{t + 1})Q(S_{t + 1},a)$  Loop for  $k = \min (t,T - 1)$  down through  $\tau +1$  ..  $G\gets R_k + \gamma \sum_{a\neq A_k}\pi (a|S_k)Q(S_k,a) + \gamma \pi (A_k|S_k)G$ $Q(S_{\tau},A_{\tau})\gets Q(S_{\tau},A_{\tau}) + \alpha [G - Q(S_{\tau},A_{\tau})]$  If  $\pi$  is being learned, then ensure that  $\pi (\cdot |S_{\tau})$  is greedy wrt  $Q$  Until  $\tau = T - 1$

# 7.6 $^ *$ A Unifying Algorithm: $\mathbf { \nabla } ^ { \prime } \mathbf { \mathit { n } } _ { \mathbf { \mathit { i } } }$ -step $Q ( \sigma )$

So far in this chapter we have considered three di↵erent kinds of action-value algorithms,corresponding to the first three backup diagrams shown in Figure 7.5. $n$ -step Sarsa hasall sample transitions, the tree-backup algorithm has all state-to-action transitions fullybranched without sampling, and $\boldsymbol { n }$ -step Expected Sarsa has all sample transitions exceptfor the last state-to-action one, which is fully branched with an expected value. To whatextent can these algorithms be unified?

One idea for unification is suggested by the fourth backup diagram in Figure 7.5. Thisis the idea that one might decide on a step-by-step basis whether one wanted to take theaction as a sample, as in Sarsa, or consider the expectation over all actions instead, as inthe tree-backup update. Then, if one chose always to sample, one would obtain Sarsa,whereas if one chose never to sample, one would get the tree-backup algorithm. ExpectedSarsa would be the case where one chose to sample for all steps except for the last one.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/feeabcc8ffa0125bc36fd28e110c1318ec16244d9883ae2faaa79c203190d76f.jpg)



Figure 7.5: The backup diagrams of the three kinds of $n$ -step action-value updates consideredso far in this chapter (4-step case) plus the backup diagram of a fourth kind of update that unifiesthem all. The label ‘ $\rho$ ’ indicates half transitions on which importance sampling is required in theo↵-policy case. The fourth kind of update unifies all the others by choosing on a state-by-statebasis whether to sample $\sigma _ { t } = 1$ ) or not $\sigma _ { t } = 0$ ).


And of course there would be many other possibilities, as suggested by the last diagramin the figure. To increase the possibilities even further we can consider a continuousvariation between sampling and expectation. Let $\sigma _ { t } \in [ 0 , 1 ]$ denote the degree of samplingon step $t$ , with $\sigma = 1$ denoting full sampling and $\sigma = 0$ denoting a pure expectation withno sampling. The random variable $\sigma _ { t }$ might be set as a function of the state, action, orstate–action pair at time $t$ . We call this proposed new algorithm $\boldsymbol { n }$ -step $Q ( \sigma )$ .

Now let us develop the equations of $n$ -step $Q ( \sigma )$ . First we write the tree-backup$n$ -step return (7.16) in terms of the horizon $h = t + n$ and then in terms of the expectedapproximate value $\bar { V }$ (7.8):

$$
\begin{array}{l} G _ {t: h} = R _ {t + 1} + \gamma \sum_ {a \neq A _ {t + 1}} \pi (a | S _ {t + 1}) Q _ {h - 1} \left(S _ {t + 1}, a\right) + \gamma \pi \left(A _ {t + 1} \mid S _ {t + 1}\right) G _ {t + 1: h} \\ = R _ {t + 1} + \gamma \bar {V} _ {h - 1} \left(S _ {t + 1}\right) - \gamma \pi \left(A _ {t + 1} \mid S _ {t + 1}\right) Q _ {h - 1} \left(S _ {t + 1}, A _ {t + 1}\right) + \gamma \pi \left(A _ {t + 1} \mid S _ {t + 1}\right) G _ {t + 1: h} \\ = R _ {t + 1} + \gamma \pi (A _ {t + 1} | S _ {t + 1}) \Big (G _ {t + 1: h} - Q _ {h - 1} (S _ {t + 1}, A _ {t + 1}) \Big) + \gamma \bar {V} _ {h - 1} (S _ {t + 1}), \\ \end{array}
$$

after which it is exactly like the $n$ -step return for Sarsa with control variates (7.14) exceptwith the action probability $\pi ( A _ { t + 1 } | S _ { t + 1 } )$ substituted for the importance-sampling ratio$\rho _ { t + 1 }$ . For $Q ( \sigma )$ , we slide linearly between these two cases:

$$
\begin{array}{l} G _ {t: h} \doteq R _ {t + 1} + \gamma \left(\sigma_ {t + 1} \rho_ {t + 1} + (1 - \sigma_ {t + 1}) \pi \left(A _ {t + 1} \mid S _ {t + 1}\right)\right) \left(G _ {t + 1: h} - Q _ {h - 1} \left(S _ {t + 1}, A _ {t + 1}\right)\right) \\ + \gamma \bar {V} _ {h - 1} \left(S _ {t + 1}\right), \tag {7.17} \\ \end{array}
$$

for $t < h \leq T$ . The recursion ends with $G _ { h : h } \doteq Q _ { h - 1 } ( S _ { h } , A _ { h } )$ if $h \ < \ T$ , or with$G _ { T - 1 : T } \ \doteq \ R _ { T }$ if $h = T$ . Then we use the earlier update for $n$ -step Sarsa withoutimportance-sampling ratios (7.5) instead of (7.11), because now the ratios are incorporatedin the $n$ -step return. A complete algorithm is given in the box.

# O↵-policy n-step Q( ) for estimating Q ⇡ q⇤ or q⇡

Input: an arbitrary behavior policy $b$ such that $b ( a | s ) > 0$ , for all $s \in \mathcal { S } , a \in \mathcal { A }$

Initialize $Q ( s , a )$ arbitrarily, for all $s \in \mathcal { S } , a \in \mathcal { A }$

Initialize $\pi$ to be greedy with respect to $Q$ , or else it is a fixed given policy

Algorithm parameters: step size $\alpha \in ( 0 , 1 ]$ , a positive integer $n$

All store and access operations can take their index mod $n + 1$

Loop for each episode:

Initialize and store S0 = terminal

Choose and store an action $A _ { 0 } \sim b ( \cdot | S _ { 0 } )$

$T \gets \infty$

Loop for $t = 0 , 1 , 2 , \ldots$ :

If $t < T$ :

Take action $A _ { t }$ ; observe and store the next reward and state as $R _ { t + 1 } , S _ { t + 1 }$

If $S _ { t + 1 }$ is terminal:

$T \gets t + 1$

else:

Choose and store an action $A _ { t + 1 } \sim b ( \cdot | S _ { t + 1 } )$

Select and store $\sigma _ { t + 1 }$

Store ⇡(At+1|St+1) $\frac { \pi ( A _ { t + 1 } | S _ { t + 1 } ) } { b ( A _ { t + 1 } | S _ { t + 1 } ) }$ as ⇢t+1 $\rho _ { t + 1 }$

$\tau  t - n + 1$ ( $\tau$ is the time whose estimate is being updated)

If $\tau \geq 0$

If $t + 1 < T$

$G  Q ( S _ { t + 1 } , A _ { t + 1 } )$

Loop for $k = \operatorname* { m i n } ( t + 1 , T )$ down through $\tau + 1$ :

if $k = T$ :

$G \gets R _ { T }$

else:

$\begin{array} { r } { V  \sum _ { a } \pi ( a | S _ { k } ) Q ( S _ { k } , a ) } \end{array}$

$G \gets R _ { k } + \gamma \big ( \sigma _ { k } \rho _ { k } + ( 1 - \sigma _ { k } ) \pi ( A _ { k } | S _ { k } ) \big ) \big ( G - Q ( S _ { k } , A _ { k } ) \big ) + \gamma V$

$Q ( S _ { \tau } , A _ { \tau } )  Q ( S _ { \tau } , A _ { \tau } ) + \alpha \lfloor G - Q ( S _ { \tau } , A _ { \tau } ) \rfloor$

If $\pi$ is being learned, then ensure that $\pi ( \cdot | S _ { \tau } )$ is greedy wrt $Q$

Until $\tau = T - 1$

# 7.7 Summary

In this chapter we have developed a range of temporal-di↵erence learning methods that liein between the one-step TD methods of the previous chapter and the Monte Carlo methodsof the chapter before. Methods that involve an intermediate amount of bootstrappingare important because they will typically perform better than either extreme.

Our focus in this chapter has been on $n$ -step methods, whichlook ahead to the next $n$ rewards, states, and actions. The two4-step backup diagrams to the right together summarize most of themethods introduced. The state-value update shown is for $n$ -stepTD with importance sampling, and the action-value update is for$n$ -step $Q ( \sigma )$ , which generalizes Expected Sarsa and Q-learning. All$n$ -step methods involve a delay of $n$ time steps before updating,as only then are all the required future events known. A furtherdrawback is that they involve more computation per time stepthan previous methods. Compared to one-step methods, $\textit { n }$ -stepmethods also require more memory to record the states, actions,rewards, and sometimes other variables over the last $n$ time steps.Eventually, in Chapter 12, we will see how multi-step TD methodscan be implemented with minimal memory and computationalcomplexity using eligibility traces, but there will always be someadditional computation beyond one-step methods. Such costs canbe well worth paying to escape the tyranny of the single time step.

Although $\textit { n }$ -step methods are more complex than those usingeligibility traces, they have the great benefit of being conceptuallyclear. We have sought to take advantage of this by developing twoapproaches to o↵-policy learning in the $n$ -step case. One, based on

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/7b6350afcc42372a23aa4344672447dd3837ef48a6f848849d40c2ba109535bc.jpg)


importance sampling is conceptually simple but can be of high variance. If the target andbehavior policies are very di↵erent it probably needs some new algorithmic ideas beforeit can be e cient and practical. The other, based on tree-backup updates, is the naturalextension of Q-learning to the multi-step case with stochastic target policies. It involvesno importance sampling but, again if the target and behavior policies are substantiallydi↵erent, the bootstrapping may span only a few steps even if $n$ is large.

# Bibliographical and Historical Remarks

The notion of $n$ -step returns is due to Watkins (1989), who also first discussed their errorreduction property. $n$ -step algorithms were explored in the first edition of this book,in which they were treated as of conceptual interest, but not feasible in practice. Thework of Cichosz (1995) and particularly van Seijen (2016) showed that they are actuallycompletely practical algorithms. Given this, and their conceptual clarity and simplicity,we have chosen to highlight them here in the second edition. In particular, we nowpostpone all discussion of the backward view and of eligibility traces until Chapter 12.

7.1–2 The results in the random walk examples were made for this text based on workof Sutton (1988) and Singh and Sutton (1996). The use of backup diagrams todescribe these and other algorithms in this chapter is new.

7.3–5 The developments in these sections are based on the work of Precup, Sutton,and Singh (2000), Precup, Sutton, and Dasgupta (2001), and Sutton, Mahmood,Precup, and van Hasselt (2014).

The tree-backup algorithm is due to Precup, Sutton, and Singh (2000), but thepresentation of it here is new.

7.6 The $Q ( \sigma )$ algorithm is new to this text, but closely related algorithms have beenexplored further by De Asis, Hernandez-Garcia, Holland, and Sutton (2017).

