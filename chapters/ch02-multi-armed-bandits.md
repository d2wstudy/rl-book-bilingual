# Chapter 2

# Multi-armed Bandits

The most important feature distinguishing reinforcement learning from other types oflearning is that it uses training information that evaluates the actions taken ratherthan instructs by giving correct actions. This is what creates the need for activeexploration, for an explicit search for good behavior. Purely evaluative feedback indicateshow good the action taken was, but not whether it was the best or the worst actionpossible. Purely instructive feedback, on the other hand, indicates the correct action totake, independently of the action actually taken. This kind of feedback is the basis ofsupervised learning, which includes large parts of pattern classification, artificial neuralnetworks, and system identification. In their pure forms, these two kinds of feedbackare quite distinct: evaluative feedback depends entirely on the action taken, whereasinstructive feedback is independent of the action taken.

In this chapter we study the evaluative aspect of reinforcement learning in a simplifiedsetting, one that does not involve learning to act in more than one situation. Thisnonassociative setting is the one in which most prior work involving evaluative feedbackhas been done, and it avoids much of the complexity of the full reinforcement learningproblem. Studying this case enables us to see most clearly how evaluative feedback di↵ersfrom, and yet can be combined with, instructive feedback.

The particular nonassociative, evaluative feedback problem that we explore is a simpleversion of the $k$ -armed bandit problem. We use this problem to introduce a numberof basic learning methods which we extend in later chapters to apply to the full rein-forcement learning problem. At the end of this chapter, we take a step closer to the fullreinforcement learning problem by discussing what happens when the bandit problembecomes associative, that is, when the best action depends on the situation.

# 2.1 A k-armed Bandit Problem

Consider the following learning problem. You are faced repeatedly with a choice among$k$ di↵erent options, or actions. After each choice you receive a numerical reward chosenfrom a stationary probability distribution that depends on the action you selected. Your

objective is to maximize the expected total reward over some time period, for example,over 1000 action selections, or time steps.

This is the original form of the $k$ -armed bandit problem, so named by analogy to a slotmachine, or “one-armed bandit,” except that it has $k$ levers instead of one. Each actionselection is like a play of one of the slot machine’s levers, and the rewards are the payo↵sfor hitting the jackpot. Through repeated action selections you are to maximize yourwinnings by concentrating your actions on the best levers. Another analogy is that ofa doctor choosing between experimental treatments for a series of seriously ill patients.Each action is the selection of a treatment, and each reward is the survival or well-beingof the patient. Today the term “bandit problem” is sometimes used for a generalizationof the problem described above, but in this book we use it to refer just to this simplecase.

In our $k$ -armed bandit problem, each of the $k$ actions has an expected or mean rewardgiven that that action is selected; let us call this the value of that action. We denote theaction selected on time step $t$ as $A _ { t }$ , and the corresponding reward as $R _ { t }$ . The value thenof an arbitrary action $a$ , denoted $q _ { * } ( a )$ , is the expected reward given that $a$ is selected:

$$
q _ {*} (a) \doteq \mathbb {E} \left[ R _ {t} \mid A _ {t} = a \right].
$$

If you knew the value of each action, then it would be trivial to solve the $k$ -armed banditproblem: you would always select the action with highest value. We assume that you donot know the action values with certainty, although you may have estimates. We denotethe estimated value of action $a$ at time step $t$ as $Q _ { t } ( a )$ . We would like $Q _ { t } ( a )$ to be closeto $q _ { * } ( a )$ .

If you maintain estimates of the action values, then at any time step there is at leastone action whose estimated value is greatest. We call these the greedy actions. When youselect one of these actions, we say that you are exploiting your current knowledge of thevalues of the actions. If instead you select one of the nongreedy actions, then we say youare exploring, because this enables you to improve your estimate of the nongreedy action’svalue. Exploitation is the right thing to do to maximize the expected reward on the onestep, but exploration may produce the greater total reward in the long run. For example,suppose a greedy action’s value is known with certainty, while several other actions areestimated to be nearly as good but with substantial uncertainty. The uncertainty issuch that at least one of these other actions probably is actually better than the greedyaction, but you don’t know which one. If you have many time steps ahead on whichto make action selections, then it may be better to explore the nongreedy actions anddiscover which of them are better than the greedy action. Reward is lower in the shortrun, during exploration, but higher in the long run because after you have discoveredthe better actions, you can exploit them many times. Because it is not possible both toexplore and to exploit with any single action selection, one often refers to the “conflict”between exploration and exploitation.

In any specific case, whether it is better to explore or exploit depends in a complexway on the precise values of the estimates, uncertainties, and the number of remainingsteps. There are many sophisticated methods for balancing exploration and exploitationfor particular mathematical formulations of the $k$ -armed bandit and related problems.

However, most of these methods make strong assumptions about stationarity and priorknowledge that are either violated or impossible to verify in most applications and inthe full reinforcement learning problem that we consider in subsequent chapters. Theguarantees of optimality or bounded loss for these methods are of little comfort when theassumptions of their theory do not apply.

In this book we do not worry about balancing exploration and exploitation in asophisticated way; we worry only about balancing them at all. In this chapter we presentseveral simple balancing methods for the $k$ -armed bandit problem and show that theywork much better than methods that always exploit. The need to balance explorationand exploitation is a distinctive challenge that arises in reinforcement learning; thesimplicity of our version of the $k$ -armed bandit problem enables us to show this in aparticularly clear form.

# 2.2 Action-value Methods

We begin by looking more closely at methods for estimating the values of actions andfor using the estimates to make action selection decisions, which we collectively callaction-value methods. Recall that the true value of an action is the mean reward whenthat action is selected. One natural way to estimate this is by averaging the rewardsactually received:

$$
Q _ {t} (a) \doteq \frac {\text {s u m o f r e w a r d s w h e n a t a k e n p r i o r t o} t}{\text {n u m b e r o f t i m e s a t a k e n p r i o r t o} t} = \frac {\sum_ {i = 1} ^ {t - 1} R _ {i} \cdot \mathbb {1} _ {A _ {i} = a}}{\sum_ {i = 1} ^ {t - 1} \mathbb {1} _ {A _ {i} = a}}, \tag {2.1}
$$

where $\mathbb { 1 } _ { p r e d i c a t e }$ denotes the random variable that is 1 if predicate is true and 0 if it is not.If the denominator is zero, then we instead define $Q _ { t } ( a )$ as some default value, such as0. As the denominator goes to infinity, by the law of large numbers, $Q _ { t } ( a )$ converges to$q _ { * } ( a )$ . We call this the sample-average method for estimating action values because eachestimate is an average of the sample of relevant rewards. Of course this is just one wayto estimate action values, and not necessarily the best one. Nevertheless, for now let usstay with this simple estimation method and turn to the question of how the estimatesmight be used to select actions.

The simplest action selection rule is to select one of the actions with the highestestimated value, that is, one of the greedy actions as defined in the previous section.If there is more than one greedy action, then a selection is made among them in somearbitrary way, perhaps randomly. We write this greedy action selection method as

$$
A _ {t} \doteq \underset {a} {\operatorname {a r g m a x}} Q _ {t} (a), \tag {2.2}
$$

where argmax denotes the action $a$ for which the expression that follows is maximized$^ a$(with ties broken arbitrarily). Greedy action selection always exploits current knowledge tomaximize immediate reward; it spends no time at all sampling apparently inferior actionsto see if they might really be better. A simple alternative is to behave greedily most ofthe time, but every once in a while, say with small probability $\varepsilon$ , instead select randomly

from among all the actions with equal probability, independently of the action-valueestimates. We call methods using this near-greedy action selection rule $\varepsilon$ -greedy methods.An advantage of these methods is that, in the limit as the number of steps increases,every action will be sampled an infinite number of times, thus ensuring that all the $Q _ { t } ( a )$converge to their respective $q _ { * } ( a )$ . This of course implies that the probability of selectingthe optimal action converges to greater than $1 - \varepsilon$ , that is, to near certainty. These arejust asymptotic guarantees, however, and say little about the practical e↵ectiveness ofthe methods.

Exercise 2.1 In $\varepsilon$ -greedy action selection, for the case of two actions and $\varepsilon = 0 . 5$ , what isthe probability that the greedy action is selected? ⇤

# 2.3 The 10-armed Testbed

To roughly assess the relative e↵ectiveness of the greedy and $\varepsilon$ -greedy action-valuemethods, we compared them numerically on a suite of test problems. This was a setof 2000 randomly generated $k$ -armed bandit problems with $k = 1 0$ . For each banditproblem, such as the one shown in Figure 2.1, the action values, $q _ { * } ( a )$ , $a = 1 , \ldots , 1 0$ ,

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/0aba7f6c7bfaacfc7e385110295a4f82e3aaa923a4a24d13da7c224435c1a06a.jpg)



Figure 2.1: An example bandit problem from the 10-armed testbed. The true value $q _ { * } ( a )$ o feach of the ten actions was selected according to a normal distribution with mean zero and unitvariance, and then the actual rewards were selected according to a mean $q _ { * } ( a )$ , unit-variancenormal distribution, as suggested by these gray distributions.


were selected according to a normal (Gaussian) distribution with mean 0 and variance 1.Then, when a learning method applied to that problem selected action $A _ { t }$ at time step $t$ ,the actual reward, $R _ { t }$ , was selected from a normal distribution with mean $q _ { * } ( A _ { t } )$ andvariance 1. These distributions are shown in gray in Figure 2.1. We call this suite of testtasks the 10-armed testbed. For any learning method, we can measure its performanceand behavior as it improves with experience over 1000 time steps when applied to one ofthe bandit problems. This makes up one run. Repeating this for 2000 independent runs,each with a di↵erent bandit problem, we obtained measures of the learning algorithm’saverage behavior.

Figure 2.2 compares a greedy method with two $\varepsilon$ -greedy methods $\varepsilon = 0 . 0 1$ and $\varepsilon = 0 . 1$ ),as described above, on the 10-armed testbed. All the methods formed their action-valueestimates using the sample-average technique (with an initial estimate of 0). The uppergraph shows the increase in expected reward with experience. The greedy methodimproved slightly faster than the other methods at the very beginning, but then leveledo↵ at a lower level. It achieved a reward-per-step of only about 1, compared with the bestpossible of about 1.54 on this testbed. The greedy method performed significantly worsein the long run because it often got stuck performing suboptimal actions. The lower graph

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/97f7f2c03f6eed116007171c370a4871e31a4f2da8f75587b885ccc4f59eff77.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/788dfb88691cb3604d43b71697ab1d595268595f9294d1b693d7e18afd42c492.jpg)



Figure 2.2: Average performance of $\varepsilon$ -greedy action-value methods on the 10-armed testbed.These data are averages over 2000 runs with di↵erent bandit problems. All methods used sampleaverages as their action-value estimates.


shows that the greedy method found the optimal action in only approximately one-thirdof the tasks. In the other two-thirds, its initial samples of the optimal action weredisappointing, and it never returned to it. The $\varepsilon$ -greedy methods eventually performedbetter because they continued to explore and to improve their chances of recognizingthe optimal action. The $\varepsilon = 0 . 1$ method explored more, and usually found the optimalaction earlier, but it never selected that action more than $9 1 \%$ of the time. The $\varepsilon = 0 . 0 1$method improved more slowly, but eventually would perform better than the $\varepsilon = 0 . 1$method on both performance measures shown in the figure. It is also possible to reduce $\varepsilon$over time to try to get the best of both high and low values.

The advantage of $\varepsilon$ -greedy over greedy methods depends on the task. For example,suppose the reward variance had been larger, say 10 instead of 1. With noisier rewardsit takes more exploration to find the optimal action, and $\varepsilon$ -greedy methods should fareeven better relative to the greedy method. On the other hand, if the reward varianceswere zero, then the greedy method would know the true value of each action after tryingit once. In this case the greedy method might actually perform best because it wouldsoon find the optimal action and then never explore. But even in the deterministic casethere is a large advantage to exploring if we weaken some of the other assumptions. Forexample, suppose the bandit task were nonstationary, that is, the true values of theactions changed over time. In this case exploration is needed even in the deterministiccase to make sure one of the nongreedy actions has not changed to become better thanthe greedy one. As we shall see in the next few chapters, nonstationarity is the casemost commonly encountered in reinforcement learning. Even if the underlying task isstationary and deterministic, the learner faces a set of banditlike decision tasks each ofwhich changes over time as learning proceeds and the agent’s decision-making policychanges. Reinforcement learning requires a balance between exploration and exploitation.

Exercise 2.2: Bandit example Consider a $k$ -armed bandit problem with $k = 4$ actions,denoted 1, 2, 3, and 4. Consider applying to this problem a bandit algorithm using$\varepsilon$ -greedy action selection, sample-average action-value estimates, and initial estimatesof $Q _ { 1 } ( a ) = 0$ , for all $a$ . Suppose the initial sequence of actions and rewards is $A _ { 1 } = 1$$R _ { 1 } = - 1$ , $A _ { 2 } = 2$ , $R _ { 2 } = 1$ , $A _ { 3 } = 2$ , $R _ { 3 } = - 2$ , $A _ { 4 } = 2$ , $R _ { 4 } = 2$ , $A _ { 5 } = 3$ , $R _ { 5 } = 0$ . On someof these time steps the $\varepsilon$ case may have occurred, causing an action to be selected atrandom. On which time steps did this definitely occur? On which time steps could thispossibly have occurred? ⇤

Exercise 2.3 In the comparison shown in Figure 2.2, which method will perform best inthe long run in terms of cumulative reward and probability of selecting the best action?How much better will it be? Express your answer quantitatively. ⇤

# 2.4 Incremental Implementation

The action-value methods we have discussed so far all estimate action values as sampleaverages of observed rewards. We now turn to the question of how these averages can becomputed in a computationally e cient manner, in particular, with constant memoryand constant per-time-step computation.

To simplify notation we concentrate on a single action. Let $R _ { i }$ now denote the rewardreceived after the $i$ th selection of this action, and let $Q _ { n }$ denote the estimate of its actionvalue after it has been selected $n - 1$ times, which we can now write simply as

$$
Q _ {n} \doteq \frac {R _ {1} + R _ {2} + \cdots + R _ {n - 1}}{n - 1}.
$$

The obvious implementation would be to maintain a record of all the rewards and thenperform this computation whenever the estimated value was needed. However, if this isdone, then the memory and computational requirements would grow over time as morerewards are seen. Each additional reward would require additional memory to store itand additional computation to compute the sum in the numerator.

As you might suspect, this is not really necessary. It is easy to devise incrementalformulas for updating averages with small, constant computation required to processeach new reward. Given $Q _ { n }$ and the $n$ th reward, $R _ { n }$ , the new average of all $n$ rewardscan be computed by

$$
\begin{array}{l} Q _ {n + 1} = \frac {1}{n} \sum_ {i = 1} ^ {n} R _ {i} \\ = \frac {1}{n} \left(R _ {n} + \sum_ {i = 1} ^ {n - 1} R _ {i}\right) \\ = \frac {1}{n} \left(R _ {n} + (n - 1) \frac {1}{n - 1} \sum_ {i = 1} ^ {n - 1} R _ {i}\right) \\ = \frac {1}{n} \left(R _ {n} + (n - 1) Q _ {n}\right) \\ = \frac {1}{n} \left(R _ {n} + n Q _ {n} - Q _ {n}\right) \\ = Q _ {n} + \frac {1}{n} \left[ R _ {n} - Q _ {n} \right], \tag {2.3} \\ \end{array}
$$

which holds even for $n = 1$ , obtaining $Q _ { 2 } = R _ { 1 }$ for arbitrary $Q _ { 1 }$ . This implementationrequires memory only for $Q _ { n }$ and $n$ , and only the small computation (2.3) for each newreward.

This update rule (2.3) is of a form that occurs frequently throughout this book. Thegeneral form is

$$
\text {N e w E s t i m a t e} \leftarrow \text {O l d E s t i m a t e} + \text {S t e p S i z e} \left[ \text {T a r g e t} - \text {O l d E s t i m a t e} \right]. \tag {2.4}
$$

The expression $\lfloor T a r g e t - O l d E s t i m a t e \rfloor$ is an error in the estimate. It is reduced by takinga step toward the “Target.” The target is presumed to indicate a desirable direction inwhich to move, though it may be noisy. In the case above, for example, the target is the$n$ th reward.

Note that the step-size parameter (StepSize) used in the incremental method (2.3)changes from time step to time step. In processing the $_ { n }$ th reward for action $a$ , the

method uses the step-size parameter $\textstyle { \frac { 1 } { n } }$ . In this book we denote the step-size parameterby $\alpha$ or, more generally, by $\alpha _ { t } ( a )$ .

Pseudocode for a complete bandit algorithm using incrementally computed sampleaverages and $\varepsilon$ -greedy action selection is shown in the box below. The function bandit(a)is assumed to take an action and return a corresponding reward.

# A simple bandit algorithm

Initialize, for $a = 1$ to $k$ :

$$
Q (a) \leftarrow 0
$$

$$
N (a) \leftarrow 0
$$

Loop forever:

$$
A \leftarrow \left\{ \begin{array}{l l} \operatorname {a r g m a x} _ {a} Q (a) & \text {w i t h p r o b a b i l i t y} 1 - \varepsilon \quad (\text {b r e a k i n g t i e s r a n d o m l y}) \\ \text {a r a n d o m a c t i o n} & \text {w i t h p r o b a b i l i t y} \varepsilon \end{array} \right.
$$

$$
R \leftarrow b a n d i t (A)
$$

$$
N (A) \leftarrow N (A) + 1
$$

$$
Q (A) \leftarrow Q (A) + \frac {1}{N (A)} \left[ R - Q (A) \right]
$$

# 2.5 Tracking a Nonstationary Problem

The averaging methods discussed so far are appropriate for stationary bandit problems,that is, for bandit problems in which the reward probabilities do not change over time.As noted earlier, we often encounter reinforcement learning problems that are e↵ectivelynonstationary. In such cases it makes sense to give more weight to recent rewards thanto long-past rewards. One of the most popular ways of doing this is to use a constantstep-size parameter. For example, the incremental update rule (2.3) for updating anaverage $Q _ { n }$ of the $n - 1$ past rewards is modified to be

$$
Q _ {n + 1} \doteq Q _ {n} + \alpha \left[ R _ {n} - Q _ {n} \right], \tag {2.5}
$$

where the step-size parameter $\alpha \in ( 0 , 1 ]$ is constant. This results in $Q _ { n + 1 }$ being a weightedaverage of past rewards and the initial estimate $Q _ { 1 }$ :

$$
\begin{array}{l} Q _ {n + 1} = Q _ {n} + \alpha \left[ R _ {n} - Q _ {n} \right] \\ = \alpha R _ {n} + (1 - \alpha) Q _ {n} \\ = \alpha R _ {n} + (1 - \alpha) [ \alpha R _ {n - 1} + (1 - \alpha) Q _ {n - 1} ] \\ = \alpha R _ {n} + (1 - \alpha) \alpha R _ {n - 1} + (1 - \alpha) ^ {2} Q _ {n - 1} \\ = \alpha R _ {n} + (1 - \alpha) \alpha R _ {n - 1} + (1 - \alpha) ^ {2} \alpha R _ {n - 2} + \\ \dots + (1 - \alpha) ^ {n - 1} \alpha R _ {1} + (1 - \alpha) ^ {n} Q _ {1} \\ = (1 - \alpha) ^ {n} Q _ {1} + \sum_ {i = 1} ^ {n} \alpha (1 - \alpha) ^ {n - i} R _ {i}. \tag {2.6} \\ \end{array}
$$

We call this a weighted average because the sum of the weights is $\begin{array} { r } { ( 1 - \alpha ) ^ { n } + \sum _ { i = 1 } ^ { n } \alpha ( 1 - } \end{array}$$\alpha ) ^ { n - i } = 1$ , as you can check for yourself. Note that the weight, $\alpha ( 1 - \alpha ) ^ { n - i }$ , given to thereward $R _ { i }$ depends on how many rewards ago, $n - i$ , it was observed. The quantity $1 - \alpha$is less than 1, and thus the weight given to $R _ { i }$ decreases as the number of interveningrewards increases. In fact, the weight decays exponentially according to the exponenton $1 - \alpha$ . (If $1 - \alpha = 0$ , then all the weight goes on the very last reward, $R _ { n }$ , becauseof the convention that $0 ^ { 0 } = 1$ .) Accordingly, this is sometimes called an exponentialrecency-weighted average.

Sometimes it is convenient to vary the step-size parameter from step to step. Let $\alpha _ { n } ( a )$denote the step-size parameter used to process the reward received after the $n$ th selectionof action $a$ . As we have noted, the choice $\textstyle \alpha _ { n } ( a ) = { \frac { 1 } { n } }$ results in the sample-average method,which is guaranteed to converge to the true action values by the law of large numbers.But of course convergence is not guaranteed for all choices of the sequence $\{ \alpha _ { n } ( a ) \}$ . Awell-known result in stochastic approximation theory gives us the conditions required toassure convergence with probability 1:

$$
\sum_ {n = 1} ^ {\infty} \alpha_ {n} (a) = \infty \quad \text {a n d} \quad \sum_ {n = 1} ^ {\infty} \alpha_ {n} ^ {2} (a) <   \infty . \tag {2.7}
$$

The first condition is required to guarantee that the steps are large enough to eventuallyovercome any initial conditions or random fluctuations. The second condition guaranteesthat eventually the steps become small enough to assure convergence.

Note that both convergence conditions are met for the sample-average case, $\textstyle \alpha _ { n } ( a ) = { \frac { 1 } { n } }$but not for the case of constant step-size parameter, $\alpha _ { n } ( a ) = \alpha$ . In the latter case, thesecond condition is not met, indicating that the estimates never completely converge butcontinue to vary in response to the most recently received rewards. As we mentionedabove, this is actually desirable in a nonstationary environment, and problems that aree↵ectively nonstationary are the most common in reinforcement learning. In addition,sequences of step-size parameters that meet the conditions (2.7) often converge very slowlyor need considerable tuning in order to obtain a satisfactory convergence rate. Althoughsequences of step-size parameters that meet these convergence conditions are often usedin theoretical work, they are seldom used in applications and empirical research.

Exercise 2.4 If the step-size parameters, $\alpha _ { n }$ , are not constant, then the estimate $Q _ { n }$ isa weighted average of previously received rewards with a weighting di↵erent from thatgiven by (2.6). What is the weighting on each prior reward for the general case, analogousto (2.6), in terms of the sequence of step-size parameters? ⇤

Exercise 2.5 (programming) Design and conduct an experiment to demonstrate thedi culties that sample-average methods have for nonstationary problems. Use a modifiedversion of the 10-armed testbed in which all the $q _ { * } ( a )$ start out equal and then takeindependent random walks (say by adding a normally distributed increment with mean 0and standard deviation 0.01 to all the $q _ { * } ( a )$ on each step). Prepare plots like Figure 2.2for an action-value method using sample averages, incrementally computed, and anotheraction-value method using a constant step-size parameter, $\alpha = 0 . 1$ . Use $\varepsilon = 0 . 1$ andlonger runs, say of 10,000 steps. ⇤

# 2.6 Optimistic Initial Values

All the methods we have discussed so far are dependent to some extent on the initialaction-value estimates, $Q _ { 1 } ( a )$ . In the language of statistics, these methods are biasedby their initial estimates. For the sample-average methods, the bias disappears once allactions have been selected at least once, but for methods with constant $\alpha$ , the bias ispermanent, though decreasing over time as given by (2.6). In practice, this kind of biasis usually not a problem and can sometimes be very helpful. The downside is that theinitial estimates become, in e↵ect, a set of parameters that must be picked by the user, ifonly to set them all to zero. The upside is that they provide an easy way to supply someprior knowledge about what level of rewards can be expected.

Initial action values can also be used as a simple way to encourage exploration. Supposethat instead of setting the initial action values to zero, as we did in the 10-armed testbed,we set them all to +5. Recall that the $q _ { * } ( a )$ in this problem are selected from a normaldistribution with mean 0 and variance 1. An initial estimate of $+ 5$ is thus wildly optimistic.But this optimism encourages action-value methods to explore. Whichever actions areinitially selected, the reward is less than the starting estimates; the learner switches toother actions, being “disappointed” with the rewards it is receiving. The result is that allactions are tried several times before the value estimates converge. The system does afair amount of exploration even if greedy actions are selected all the time.

Figure 2.3 shows the performance on the 10-armed bandit testbed of a greedy methodusing $Q _ { 1 } ( a ) = + 5$ , for all $a$ . For comparison, also shown is an $\varepsilon$ -greedy method with$Q _ { 1 } ( a ) = 0$ . Initially, the optimistic method performs worse because it explores more,but eventually it performs better because its exploration decreases with time. We callthis technique for encouraging exploration optimistic initial values. We regard it asa simple trick that can be quite e↵ective on stationary problems, but it is far frombeing a generally useful approach to encouraging exploration. For example, it is notwell suited to nonstationary problems because its drive for exploration is inherently

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/4da0efcbb178500a2b640df32850d39cdf66d4ee5697a3f58a979bd5aa72e05e.jpg)



Figure 2.3: The e↵ect of optimistic initial action-value estimates on the 10-armed testbed.Both methods used a constant step-size parameter, $\alpha = 0 . 1$ .


temporary. If the task changes, creating a renewed need for exploration, this methodcannot help. Indeed, any method that focuses on the initial conditions in any special wayis unlikely to help with the general nonstationary case. The beginning of time occursonly once, and thus we should not focus on it too much. This criticism applies as well tothe sample-average methods, which also treat the beginning of time as a special event,averaging all subsequent rewards with equal weights. Nevertheless, all of these methodsare very simple, and one of them—or some simple combination of them—is often adequatein practice. In the rest of this book we make frequent use of several of these simpleexploration techniques.

Exercise 2.6: Mysterious Spikes The results shown in Figure 2.3 should be quite reliablebecause they are averages over 2000 individual, randomly chosen 10-armed bandit tasks.Why, then, are there oscillations and spikes in the early part of the curve for the optimisticmethod? In other words, what might make this method perform particularly better orworse, on average, on particular early steps? ⇤

Exercise 2.7: Unbiased Constant-Step-Size Trick In most of this chapter we have usedsample averages to estimate action values because sample averages do not produce theinitial bias that constant step sizes do (see the analysis leading to (2.6)). However, sampleaverages are not a completely satisfactory solution because they may perform poorlyon nonstationary problems. Is it possible to avoid the bias of constant step sizes whileretaining their advantages on nonstationary problems? One way is to use a step size of

$$
\beta_ {n} \doteq \alpha / \bar {\sigma} _ {n}, \tag {2.8}
$$

to process the $n$ th reward for a particular action, where $\alpha > 0$ is a conventional constantstep size, and $o _ { n }$ is a trace of one that starts at 0:

$$
\bar {o} _ {n} \doteq \bar {o} _ {n - 1} + \alpha (1 - \bar {o} _ {n - 1}), \quad \text {f o r} n > 0, \quad \text {w i t h} \bar {o} _ {0} \doteq 0. \tag {2.9}
$$

Carry out an analysis like that in (2.6) to show that $Q _ { n }$ is an exponential recency-weightedaverage without initial bias. ⇤

# 2.7 Upper-Confidence-Bound Action Selection

Exploration is needed because there is always uncertainty about the accuracy of theaction-value estimates. The greedy actions are those that look best at present, but some ofthe other actions may actually be better. $\varepsilon$ -greedy action selection forces the non-greedyactions to be tried, but indiscriminately, with no preference for those that are nearlygreedy or particularly uncertain. It would be better to select among the non-greedyactions according to their potential for actually being optimal, taking into account bothhow close their estimates are to being maximal and the uncertainties in those estimates.One e↵ective way of doing this is to select actions according to

$$
A _ {t} \doteq \underset {a} {\arg \max } \left[ Q _ {t} (a) + c \sqrt {\frac {\ln t}{N _ {t} (a)}} \right], \tag {2.10}
$$

where $\ln t$ denotes the natural logarithm of $t$ (the number that $e \approx 2 . 7 1 8 2 8$ would haveto be raised to in order to equal $t$ ), $N _ { t } ( a )$ denotes the number of times that action $a$ has

been selected prior to time $t$ (the denominator in (2.1)), and the number $c > 0$ controlsthe degree of exploration. If $N _ { t } ( a ) = 0$ , then $a$ is considered to be a maximizing action.

The idea of this upper confidence bound (UCB) action selection is that the square-rootterm is a measure of the uncertainty or variance in the estimate of $a$ ’s value. The quantitybeing max’ed over is thus a sort of upper bound on the possible true value of action $a$ , with$c$ determining the confidence level. Each time $a$ is selected the uncertainty is presumablyreduced: $N _ { t } ( a )$ increments, and, as it appears in the denominator, the uncertainty termdecreases. On the other hand, each time an action other than $a$ is selected, $t$ increases but$N _ { t } ( a )$ does not; because $t$ appears in the numerator, the uncertainty estimate increases.The use of the natural logarithm means that the increases get smaller over time, but areunbounded; all actions will eventually be selected, but actions with lower value estimates,or that have already been selected frequently, will be selected with decreasing frequencyover time.

Results with UCB on the 10-armed testbed are shown in Figure 2.4. UCB oftenperforms well, as shown here, but is more di cult than $\varepsilon$ -greedy to extend beyond banditsto the more general reinforcement learning settings considered in the rest of this book.One di culty is in dealing with nonstationary problems; methods more complex thanthose presented in Section 2.5 would be needed. Another di culty is dealing with largestate spaces, particularly when using function approximation as developed in Part II ofthis book. In these more advanced settings the idea of UCB action selection is usuallynot practical.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/87874a8fa9c9550d54680200393e7972104aafea3123d3992ee6cb665e6c5995.jpg)



Figure 2.4: Average performance of UCB action selection on the 10-armed testbed. As shown,UCB generally performs better than $\varepsilon$ -greedy action selection, except in the first $k$ steps, whenit selects randomly among the as-yet-untried actions.


Exercise 2.8: UCB Spikes In Figure 2.4 the UCB algorithm shows a distinct spikein performance on the 11th step. Why is this? Note that for your answer to be fullysatisfactory it must explain both why the reward increases on the 11th step and why itdecreases on the subsequent steps. Hint: If $c = 1$ , then the spike is less prominent. ⇤

# 2.8 Gradient Bandit Algorithms

So far in this chapter we have considered methods that estimate action values and usethose estimates to select actions. This is often a good approach, but it is not the onlyone possible. In this section we consider learning a numerical preference for each action$a$ , which we denote $H _ { t } ( a ) \in \mathbb { R }$ . The larger the preference, the more often that action istaken, but the preference has no interpretation in terms of reward. Only the relativepreference of one action over another is important; if we add 1000 to all the actionpreferences there is no e↵ect on the action probabilities, which are determined accordingto a soft-max distribution (i.e., Gibbs or Boltzmann distribution) as follows:

$$
\Pr \left\{A _ {t} = a \right\} \doteq \frac {e ^ {H _ {t} (a)}}{\sum_ {b = 1} ^ {k} e ^ {H _ {t} (b)}} \doteq \pi_ {t} (a), \tag {2.11}
$$

where here we have also introduced a useful new notation, $\pi _ { t } ( a )$ , for the probability oftaking action $a$ at time $t$ . Initially all action preferences are the same (e.g., $H _ { 1 } ( a ) = 0$ ,for all $a$ ) so that all actions have an equal probability of being selected.

Exercise 2.9 Show that in the case of two actions, the soft-max distribution is the sameas that given by the logistic, or sigmoid, function often used in statistics and artificialneural networks. ⇤

There is a natural learning algorithm for soft-max action preferences based on the ideaof stochastic gradient ascent. On each step, after selecting action $A _ { t }$ and receiving thereward $R _ { t }$ , the action preferences are updated by:

$$
H _ {t + 1} \left(A _ {t}\right) \doteq H _ {t} \left(A _ {t}\right) + \alpha \left(R _ {t} - \bar {R} _ {t}\right) \left(1 - \pi_ {t} \left(A _ {t}\right)\right), \quad \text {a n d} \tag {2.12}
$$

$$
H _ {t + 1} (a) \doteq H _ {t} (a) - \alpha \left(R _ {t} - \bar {R} _ {t}\right) \pi_ {t} (a), \quad \text {f o r a l l} a \neq A _ {t},
$$

where $\alpha > 0$ is a step-size parameter, and $R _ { t } \in \mathbb { R }$ is the average of the rewards up to butnot including time $t$ (with $R _ { 1 } \doteq R _ { 1 }$ ), which can be computed incrementally as describedin Section 2.4 (or Section 2.5 if the problem is nonstationary).1 The $R _ { t }$ term serves as abaseline with which the reward is compared. If the reward is higher than the baseline,then the probability of taking $A _ { t }$ in the future is increased, and if the reward is belowbaseline, then the probability is decreased. The non-selected actions move in the oppositedirection.

Figure 2.5 shows results with the gradient bandit algorithm on a variant of the 10-armed testbed in which the true expected rewards were selected according to a normaldistribution with a mean of $+ 4$ instead of zero (and with unit variance as before). Thisshifting up of all the rewards has absolutely no e↵ect on the gradient bandit algorithmbecause of the reward baseline term, which instantaneously adapts to the new level. Butif the baseline were omitted (that is, if $R _ { t }$ was taken to be constant zero in (2.12)), thenperformance would be significantly degraded, as shown in the figure.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/1b3ea2fb0dfb038a9cb1ff6fc438bcf05b0bf981f919a65e5e13e2a1b9a9efbc.jpg)



Figure 2.5: Average performance of the gradient bandit algorithm with and without a rewardbaseline on the 10-armed testbed when the $q _ { * } ( a )$ are chosen to be near +4 rather than near zero.


# The Bandit Gradient Algorithm as Stochastic Gradient Ascent

One can gain a deeper insight into the gradient bandit algorithm by understandingit as a stochastic approximation to gradient ascent. In exact gradient ascent, eachaction preference $H _ { t } ( a )$ would be incremented in proportion to the increment’se↵ect on performance:

$$
H _ {t + 1} (a) \doteq H _ {t} (a) + \alpha \frac {\partial \mathbb {E} [ R _ {t} ]}{\partial H _ {t} (a)}, \tag {2.13}
$$

where the measure of performance here is the expected reward:

$$
\mathbb {E} [ R _ {t} ] = \sum_ {x} \pi_ {t} (x) q _ {*} (x),
$$

and the measure of the increment’s e↵ect is the partial derivative of this performancemeasure with respect to the action preference. Of course, it is not possible toimplement gradient ascent exactly in our case because by assumption we do notknow the $q _ { * } ( x )$ , but in fact the updates of our algorithm (2.12) are equal to (2.13)in expected value, making the algorithm an instance of stochastic gradient ascent.The calculations showing this require only beginning calculus, but take several

steps. First we take a closer look at the exact performance gradient:

$$
\begin{array}{l} \frac {\partial \mathbb {E} [ R _ {t} ]}{\partial H _ {t} (a)} = \frac {\partial}{\partial H _ {t} (a)} \left[ \sum_ {x} \pi_ {t} (x) q _ {*} (x) \right] \\ = \sum_ {x} q _ {*} (x) \frac {\partial \pi_ {t} (x)}{\partial H _ {t} (a)} \\ = \sum_ {x} \left(q _ {*} (x) - B _ {t}\right) \frac {\partial \pi_ {t} (x)}{\partial H _ {t} (a)}, \\ \end{array}
$$

where $B _ { t }$ , called the baseline, can be any scalar that does not depend on $x$ . We caninclude a baseline here withoto zero over all the actions, $\begin{array} { r } { \sum _ { x } \frac { \partial \pi _ { t } ( x ) } { \partial H _ { t } ( a ) } = 0 } \end{array}$ e equ. As $H _ { t } ( a )$ because the gradient sums is changed, some actions’probabilities go up and some go down, but the sum of the changes must be zerobecause the sum of the probabilities is always one.

Next we multiply each term of the sum by $\pi _ { t } ( x ) / \pi _ { t } ( x )$ :

$$
\frac {\partial \mathbb {E} [ R _ {t} ]}{\partial H _ {t} (a)} = \sum_ {x} \pi_ {t} (x) \big (q _ {*} (x) - B _ {t} \big) \frac {\partial \pi_ {t} (x)}{\partial H _ {t} (a)} / \pi_ {t} (x).
$$

The equation is now in the form of an expectation, summing over all possible values$x$ of the random variable $A _ { t }$ , then multiplying by the probability of taking thosevalues. Thus:

$$
\begin{array}{l} = \mathbb {E} \left[ \left(q _ {*} (A _ {t}) - B _ {t}\right) \frac {\partial \pi_ {t} (A _ {t})}{\partial H _ {t} (a)} / \pi_ {t} (A _ {t}) \right] \\ = \mathbb {E} \left[ \left(R _ {t} - \bar {R} _ {t}\right) \frac {\partial \pi_ {t} (A _ {t})}{\partial H _ {t} (a)} / \pi_ {t} (A _ {t}) \right], \\ \end{array}
$$

where here we have chosen the baseline $B _ { t } = R _ { t }$ and substituted $R _ { t }$ for $q _ { * } ( A _ { t } )$ ,$\mathbb { E } [ R _ { t } | A _ { t } ] = q _ { * } ( A _ { t } )$ . Shortly we will  defined to be 1 if ish that, else 0.$\begin{array} { r } { \frac { \partial \pi _ { t } ( x ) } { \partial H _ { t } ( a ) } \ : = \ : \pi _ { t } ( x ) \big ( \mathbb { 1 } _ { a = x } - \pi _ { t } ( a ) \big ) } \end{array}$ $\mathbb { 1 } _ { a = x }$ $a = x$

$$
\begin{array}{l} = \mathbb {E} \left[ \left(R _ {t} - \bar {R} _ {t}\right) \pi_ {t} (A _ {t}) \left(\mathbb {1} _ {a = A _ {t}} - \pi_ {t} (a)\right) / \pi_ {t} (A _ {t}) \right] \\ = \mathbb {E} \left[ \left(R _ {t} - \bar {R} _ {t}\right) \left(\mathbb {1} _ {a = A _ {t}} - \pi_ {t} (a)\right) \right]. \\ \end{array}
$$

Recall that our plan has been to write the performance gradient as an expectationof something that we can sample on each step, as we have just done, and thenupdate on each step in proportion to the sample. Substituting a sample of theexpectation above for the performance gradient in (2.13) yields:

$$
H _ {t + 1} (a) = H _ {t} (a) + \alpha \bigl (R _ {t} - \bar {R} _ {t} \bigr) \bigl (\mathbb {1} _ {a = A _ {t}} - \pi_ {t} (a) \bigr), \qquad \mathrm {f o r a l l} a,
$$

which you may recognize as being equivalent to our original algorithm (2.12).

Thus it remains only to show that $\begin{array} { r } { \frac { \partial \pi _ { t } ( x ) } { \partial H _ { t } ( a ) } = \pi _ { t } ( x ) \big ( \mathbb { 1 } _ { a = x } - \pi _ { t } ( a ) \big ) } \end{array}$ , as we assumed.Recall the standard quotient rule for derivatives:

$$
\frac {\partial}{\partial x} \left[ \frac {f (x)}{g (x)} \right] = \frac {\frac {\partial f (x)}{\partial x} g (x) - f (x) \frac {\partial g (x)}{\partial x}}{g (x) ^ {2}}.
$$

Using this, we can write

$$
\begin{array}{l} \frac {\partial \pi_ {t} (x)}{\partial H _ {t} (a)} = \frac {\partial}{\partial H _ {t} (a)} \pi_ {t} (x) \\ = \frac {\partial}{\partial H _ {t} (a)} \left[ \frac {e ^ {H _ {t} (x)}}{\sum_ {y = 1} ^ {k} e ^ {H _ {t} (y)}} \right] \\ = \frac {\frac {\partial e ^ {H _ {t} (x)}}{\partial H _ {t} (a)} \sum_ {y = 1} ^ {k} e ^ {H _ {t} (y)} - e ^ {H _ {t} (x)} \frac {\partial \sum_ {y = 1} ^ {k} e ^ {H _ {t} (y)}}{\partial H _ {t} (a)}}{\left(\sum_ {y = 1} ^ {k} e ^ {H _ {t} (y)}\right) ^ {2}} \quad (\text {b y}) \\ = \frac {\mathbb {1} _ {a = x} e ^ {H _ {t} (x)} \sum_ {y = 1} ^ {k} e ^ {H _ {t} (y)} - e ^ {H _ {t} (x)} e ^ {H _ {t} (a)}}{\left(\sum_ {y = 1} ^ {k} e ^ {H _ {t} (y)}\right) ^ {2}} \quad \text {(b e c a u s e} \frac {\partial e ^ {x}}{\partial x} = e ^ {x}) \\ = \frac {\mathbb {1} _ {a = x} e ^ {H _ {t} (x)}}{\sum_ {y = 1} ^ {k} e ^ {H _ {t} (y)}} - \frac {e ^ {H _ {t} (x)} e ^ {H _ {t} (a)}}{\left(\sum_ {y = 1} ^ {k} e ^ {H _ {t} (y)}\right) ^ {2}} \\ = \mathbb {1} _ {a = x} \pi_ {t} (x) - \pi_ {t} (x) \pi_ {t} (a) \\ = \pi_ {t} (x) \left(\mathbb {1} _ {a = x} - \pi_ {t} (a)\right). \tag {Q.E.D.} \\ \end{array}
$$

We have just shown that the expected update of the gradient bandit algorithmis equal to the gradient of expected reward, and thus that the algorithm is aninstance of stochastic gradient ascent. This assures us that the algorithm has robustconvergence properties.

Note that we did not require any properties of the reward baseline other thanthat it does not depend on the selected action. For example, we could have setit to zero, or to 1000, and the algorithm would still be an instance of stochasticgradient ascent. The choice of the baseline does not a↵ect the expected updateof the algorithm, but it does a↵ect the variance of the update and thus the rate ofconvergence (as shown, for example, in Figure 2.5). Choosing it as the average ofthe rewards may not be the very best, but it is simple and works well in practice.

# 2.9 Associative Search (Contextual Bandits)

So far in this chapter we have considered only nonassociative tasks, that is, tasks in whichthere is no need to associate di↵erent actions with di↵erent situations. In these tasksthe learner either tries to find a single best action when the task is stationary, or tries totrack the best action as it changes over time when the task is nonstationary. However,in a general reinforcement learning task there is more than one situation, and the goalis to learn a policy: a mapping from situations to the actions that are best in thosesituations. To set the stage for the full problem, we briefly discuss the simplest way inwhich nonassociative tasks extend to the associative setting.

As an example, suppose there are several di↵erent $k$ -armed bandit tasks, and that oneach step you confront one of these chosen at random. Thus, the bandit task changesrandomly from step to step. If the probabilities with which each task is selected for youdo not change over time, this would appear as a single stationary $k$ -armed bandit task,and you could use one of the methods described in this chapter. Now suppose, however,that when a bandit task is selected for you, you are given some distinctive clue about itsidentity (but not its action values). Maybe you are facing an actual slot machine thatchanges the color of its display as it changes its action values. Now you can learn a policyassociating each task, signaled by the color you see, with the best action to take whenfacing that task—for instance, if red, select arm 1; if green, select arm 2. With the rightpolicy you can usually do much better than you could in the absence of any informationdistinguishing one bandit task from another.

This is an example of an associative search task, so called because it involves bothtrial-and-error learning to search for the best actions, and association of these actionswith the situations in which they are best. Associative search tasks are often now calledcontextual bandits in the literature. Associative search tasks are intermediate betweenthe $k$ -armed bandit problem and the full reinforcement learning problem. They are likethe full reinforcement learning problem in that they involve learning a policy, but theyare also like our version of the $k$ -armed bandit problem in that each action a↵ects onlythe immediate reward. If actions are allowed to a↵ect the next situation as well as thereward, then we have the full reinforcement learning problem. We present this problemin the next chapter and consider its ramifications throughout the rest of the book.

Exercise 2.10 Suppose you face a 2-armed bandit task whose true action values changerandomly from time step to time step. Specifically, suppose that, for any time step,the true values of actions 1 and 2 are respectively 10 and 20 with probability 0.5 (caseA), and 90 and 80 with probability 0.5 (case B). If you are not able to tell which caseyou face at any step, what is the best expected reward you can achieve and how shouldyou behave to achieve it? Now suppose that on each step you are told whether you arefacing case A or case B (although you still don’t know the true action values). This is anassociative search task. What is the best expected reward you can achieve in this task,and how should you behave to achieve it? ⇤

# 2.10 Summary

We have presented in this chapter several simple ways of balancing exploration andexploitation. The $\varepsilon$ -greedy methods choose randomly a small fraction of the time, whereasUCB methods choose deterministically but achieve exploration by subtly favoring at eachstep the actions that have so far received fewer samples. Gradient bandit algorithmsestimate not action values, but action preferences, and favor the more preferred actionsin a graded, probabilistic manner using a soft-max distribution. The simple expedient ofinitializing estimates optimistically causes even greedy methods to explore significantly.

It is natural to ask which of these methods is best. Although this is a di cult questionto answer in general, we can certainly run them all on the 10-armed testbed that wehave used throughout this chapter and compare their performances. A complication isthat they all have a parameter; to get a meaningful comparison we have to considertheir performance as a function of their parameter. Our graphs so far have shown thecourse of learning over time for each algorithm and parameter setting, to produce alearning curve for that algorithm and parameter setting. If we plotted learning curvesfor all algorithms and all parameter settings, then the graph would be too complex andcrowded to make clear comparisons. Instead we summarize a complete learning curveby its average value over the 1000 steps; this value is proportional to the area under thelearning curve. Figure 2.6 shows this measure for the various bandit algorithms fromthis chapter, each as a function of its own parameter shown on a single scale on thex-axis. This kind of graph is called a parameter study. Note that the parameter valuesare varied by factors of two and presented on a log scale. Note also the characteristicinverted-U shapes of each algorithm’s performance; all the algorithms perform best atan intermediate value of their parameter, neither too large nor too small. In assessing

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/58b88764b991bd7c11004b8fb152510029e57b6072c4eb13e92d547240d38c89.jpg)



Figure 2.6: A parameter study of the various bandit algorithms presented in this chapter.Each point is the average reward obtained over 1000 steps with a particular algorithm at aparticular setting of its parameter.


a method, we should attend not just to how well it does at its best parameter setting,but also to how sensitive it is to its parameter value. All of these algorithms are fairlyinsensitive, performing well over a range of parameter values varying by about an orderof magnitude. Overall, on this problem, UCB seems to perform best.

Despite their simplicity, in our opinion the methods presented in this chapter canfairly be considered the state of the art. There are more sophisticated methods, but theircomplexity and assumptions make them impractical for the full reinforcement learningproblem that is our real focus. Starting in Chapter 5 we present learning methods forsolving the full reinforcement learning problem that use in part the simple methodsexplored in this chapter.

Although the simple methods explored in this chapter may be the best we can doat present, they are far from a fully satisfactory solution to the problem of balancingexploration and exploitation.

One well-studied approach to balancing exploration and exploitation in $k$ -armed banditproblems is to compute a special kind of action value called a Gittins index. In certainimportant special cases, this computation is tractable and leads directly to optimalsolutions, although it does require complete knowledge of the prior distribution of possibleproblems, which we generally assume is not available. In addition, neither the theorynor the computational tractability of this approach appear to generalize to the fullreinforcement learning problem that we consider in the rest of the book.

The Gittins-index approach is an instance of Bayesian methods, which assume a knowninitial distribution over the action values and then update the distribution exactly aftereach step (assuming that the true action values are stationary). In general, the updatecomputations can be very complex, but for certain special distributions (called conjugatepriors) they are easy. One possibility is to then select actions at each step accordingto their posterior probability of being the best action. This method, sometimes calledposterior sampling or Thompson sampling, often performs similarly to the best of thedistribution-free methods we have presented in this chapter.

In the Bayesian setting it is even conceivable to compute the optimal balance betweenexploration and exploitation. One can compute for any possible action the probabilityof each possible immediate reward and the resultant posterior distributions over actionvalues. This evolving distribution becomes the information state of the problem. Givena horizon, say of 1000 steps, one can consider all possible actions, all possible resultingrewards, all possible next actions, all next rewards, and so on for all 1000 steps. Giventhe assumptions, the rewards and probabilities of each possible chain of events can bedetermined; one need only pick the best. But the tree of possibilities grows extremelyrapidly; even if there were only two actions and two rewards, the tree would have $2 ^ { 2 0 0 0 }$leaves. It is generally not feasible to perform this immense computation exactly, butperhaps it could be approximated e ciently. This approach would e↵ectively turn thebandit problem into an instance of the full reinforcement learning problem. In the end, wemay be able to use approximate reinforcement learning methods such as those presentedin Part II of this book to approach this optimal solution. But that is a topic for researchand beyond the scope of this introductory book.

Exercise 2.11 (programming) Make a figure analogous to Figure 2.6 for the nonstationarycase outlined in Exercise 2.5. Include the constant-step-size $\varepsilon$ -greedy algorithm with$\alpha { = } 0 . 1$ . Use runs of 200,000 steps and, as a performance measure for each algorithm andparameter setting, use the average reward over the last 100,000 steps. ⇤

# Bibliographical and Historical Remarks

2.1 Bandit problems have been studied in statistics, engineering, and psychology. Instatistics, bandit problems fall under the heading “sequential design of experi-ments,” introduced by Thompson (1933, 1934) and Robbins (1952), and studiedby Bellman (1956). Berry and Fristedt (1985) provide an extensive treatment ofbandit problems from the perspective of statistics. Narendra and Thathachar(1989) treat bandit problems from the engineering perspective, providing a gooddiscussion of the various theoretical traditions that have focused on them. Inpsychology, bandit problems have played roles in statistical learning theory (e.g.,Bush and Mosteller, 1955; Estes, 1950).

The term greedy is often used in the heuristic search literature (e.g., Pearl, 1984).The conflict between exploration and exploitation is known in control engineeringas the conflict between identification (or estimation) and control (e.g., Witten,1976b). Feldbaum (1965) called it the dual control problem, referring to theneed to solve the two problems of identification and control simultaneously whentrying to control a system under uncertainty. In discussing aspects of geneticalgorithms, Holland (1975) emphasized the importance of this conflict, referringto it as the conflict between the need to exploit and the need for new information.

2.2 Action-value methods for our $k$ -armed bandit problem were first proposed byThathachar and Sastry (1985). These are often called estimator algorithms in thelearning automata literature. The term action value is due to Watkins (1989).The first to use $\varepsilon$ -greedy methods may also have been Watkins (1989, p. 187),but the idea is so simple that some earlier use seems likely.

2.4–5 This material falls under the general heading of stochastic iterative algorithms,which is well covered by Bertsekas and Tsitsiklis (1996).

2.6 Optimistic initialization was used in reinforcement learning by Sutton (1996).

2.7 Early work on using estimates of the upper confidence bound to select actionswas done by Lai and Robbins (1985), Kaelbling (1993b), and Agrawal (1995).The UCB algorithm we present here is called UCB1 in the literature and wasfirst developed by Auer, Cesa-Bianchi and Fischer (2002).

2.8 Gradient bandit algorithms are a special case of the gradient-based reinforcementlearning algorithms introduced by Williams (1992) that later developed into theactor–critic and policy-gradient algorithms that we treat later in this book. Ourdevelopment here was influenced by that by Balaraman Ravindran (personal

communication). Further discussion of the choice of baseline is provided byGreensmith, Bartlett, and Baxter (2002, 2004) and by Dick (2015). Earlysystematic studies of algorithms like this were done by Sutton (1984).

The term soft-max for the action selection rule (2.11) is due to Bridle (1990).This rule appears to have been first proposed by Luce (1959).

2.9 The term associative search and the corresponding problem were introduced byBarto, Sutton, and Brouwer (1981). The term associative reinforcement learninghas also been used for associative search (Barto and Anandan, 1985), but weprefer to reserve that term as a synonym for the full reinforcement learningproblem (as in Sutton, 1984). (And, as we noted, the modern literature alsouses the term “contextual bandits” for this problem.) We note that Thorndike’sLaw of E↵ect (quoted in Chapter 1) describes associative search by referringto the formation of associative links between situations (states) and actions.According to the terminology of operant, or instrumental, conditioning (e.g.,Skinner, 1938), a discriminative stimulus is a stimulus that signals the presenceof a particular reinforcement contingency. In our terms, di↵erent discriminativestimuli correspond to di↵erent states.

2.10 Bellman (1956) was the first to show how dynamic programming could be usedto compute the optimal balance between exploration and exploitation within aBayesian formulation of the problem. The Gittins index approach is due to Gittinsand Jones (1974). Du↵ (1995) showed how it is possible to learn Gittins indicesfor bandit problems through reinforcement learning. The survey by Kumar (1985)provides a good discussion of Bayesian and non-Bayesian approaches to theseproblems. The term information state comes from the literature on partiallyobservable MDPs; see, for example, Lovejoy (1991).

Other theoretical research focuses on the e ciency of exploration, usually ex-pressed as how quickly an algorithm can approach an optimal decision-makingpolicy. One way to formalize exploration e ciency is by adapting to reinforcementlearning the notion of sample complexity for a supervised learning algorithm,which is the number of training examples the algorithm needs to attain a desireddegree of accuracy in learning the target function. A definition of the samplecomplexity of exploration for a reinforcement learning algorithm is the number oftime steps in which the algorithm does not select near-optimal actions (Kakade,2003). Li (2012) discusses this and several other approaches in a survey of theo-retical approaches to exploration e ciency in reinforcement learning. A thoroughmodern treatment of Thompson sampling is provided by Russo et al. (2018).

