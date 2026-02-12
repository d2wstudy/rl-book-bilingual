# Chapter 11

# *O↵-policy Methods withApproximation

This book has treated on-policy and o↵-policy learning methods since Chapter 5 primarilyas two alternative ways of handling the conflict between exploitation and explorationinherent in learning forms of generalized policy iteration. The two chapters preceding thishave treated the $o n$ -policy case with function approximation, and in this chapter we treatthe o↵ -policy case with function approximation. The extension to function approximationturns out to be significantly di↵erent and harder for o↵-policy learning than it is foron-policy learning. The tabular o↵-policy methods developed in Chapters 6 and 7 readilyextend to semi-gradient algorithms, but these algorithms do not converge as robustly asthey do under on-policy training. In this chapter we explore the convergence problems,take a closer look at the theory of linear function approximation, introduce a notion oflearnability, and then discuss new algorithms with stronger convergence guarantees for theo↵-policy case. In the end we will have improved methods, but the theoretical results willnot be as strong, nor the empirical results as satisfying, as they are for on-policy learning.Along the way, we will gain a deeper understanding of approximation in reinforcementlearning for on-policy learning as well as o↵-policy learning.

Recall that in o↵-policy learning we seek to learn a value function for a target policy$\pi$ , given data due to a di↵erent behavior policy $b$ . In the prediction case, both policiesare static and given, and we seek to learn either state values $\hat { v } \approx v _ { \pi }$ or action values$\hat { q } \approx q _ { \pi }$ . In the control case, action values are learned, and both policies typically changeduring learning— $\pi$ being the greedy policy with respect to $\hat { q }$ , and $b$ being somethingmore exploratory such as the $\varepsilon$ -greedy policy with respect to $\hat { q }$ .

The challenge of o↵-policy learning can be divided into two parts, one that arises inthe tabular case and one that arises only with function approximation. The first partof the challenge has to do with the target of the update (not to be confused with thetarget policy), and the second part has to do with the distribution of the updates. Thetechniques related to importance sampling developed in Chapters 5 and 7 deal withthe first part; these may increase variance but are needed in all successful algorithms,

tabular and approximate. The extension of these techniques to function approximationare quickly dealt with in the first section of this chapter.

Something more is needed for the second part of the challenge of o↵-policy learningwith function approximation because the distribution of updates in the o↵-policy case isnot according to the on-policy distribution. The on-policy distribution is important tothe stability of semi-gradient methods. Two general approaches have been explored todeal with this. One is to use importance sampling methods again, this time to warp theupdate distribution back to the on-policy distribution, so that semi-gradient methodsare guaranteed to converge (in the linear case). The other is to develop true gradientmethods that do not rely on any special distribution for stability. We present methodsbased on both approaches. This is a cutting-edge research area, and it is not clear whichof these approaches is most e↵ective in practice.

# 11.1 Semi-gradient Methods

We begin by describing how the methods developed in earlier chapters for the o↵-policy case extend readily to function approximation as semi-gradient methods. Thesemethods address the first part of the challenge of o↵-policy learning (changing the updatetargets) but not the second part (changing the update distribution). Accordingly, thesemethods may diverge in some cases, and in that sense are not sound, but still theyare often successfully used. Remember that these methods are guaranteed stable andasymptotically unbiased for the tabular case, which corresponds to a special case offunction approximation. So it may still be possible to combine them with feature selectionmethods in such a way that the combined system could be assured stable. In any event,these methods are simple and thus a good place to start.

In Chapter 7 we described a variety of tabular o↵-policy algorithms. To convert themto semi-gradient form, we simply replace the update to an array ( $V$ or $Q$ ) to an updateto a weight vector (w), using the approximate value function ( $\hat { v }$ or $\hat { q }$ ) and its gradient.Many of these algorithms use the per-step importance sampling ratio:

$$
\rho_ {t} \doteq \rho_ {t: t} = \frac {\pi \left(A _ {t} \mid S _ {t}\right)}{b \left(A _ {t} \mid S _ {t}\right)}. \tag {11.1}
$$

For example, the one-step, state-value algorithm is semi-gradient o↵-policy TD(0), whichis just like the corresponding on-policy algorithm (page 203) except for the addition of$\rho _ { t }$ :

$$
\mathbf {w} _ {t + 1} \doteq \mathbf {w} _ {t} + \alpha \rho_ {t} \delta_ {t} \nabla \hat {v} \left(S _ {t}, \mathbf {w} _ {t}\right), \tag {11.2}
$$

where $\delta _ { t }$ is defined appropriately depending on whether the problem is episodic anddiscounted, or continuing and undiscounted using average reward:

$$
\delta_ {t} \dot {=} R _ {t + 1} + \gamma \hat {v} \left(S _ {t + 1}, \mathbf {w} _ {t}\right) - \hat {v} \left(S _ {t}, \mathbf {w} _ {t}\right), \text {o r} \tag {11.3}
$$

$$
\delta_ {t} \doteq R _ {t + 1} - \bar {R} _ {t} + \hat {v} \left(S _ {t + 1}, \mathbf {w} _ {t}\right) - \hat {v} \left(S _ {t}, \mathbf {w} _ {t}\right). \tag {11.4}
$$

For action values, the one-step algorithm is semi-gradient Expected Sarsa:

$$
\mathbf {w} _ {t + 1} \doteq \mathbf {w} _ {t} + \alpha \delta_ {t} \nabla \hat {q} \left(S _ {t}, A _ {t}, \mathbf {w} _ {t}\right), \text {w i t h} \tag {11.5}
$$

$$
\delta_ {t} \doteq R _ {t + 1} + \gamma \sum_ {a} \pi (a | S _ {t + 1}) \hat {q} (S _ {t + 1}, a, \mathbf {w} _ {t}) - \hat {q} (S _ {t}, A _ {t}, \mathbf {w} _ {t}), \text {o r} \tag {episodic}
$$

$$
\delta_ {t} \doteq R _ {t + 1} - \bar {R} _ {t} + \sum_ {a} \pi (a | S _ {t + 1}) \hat {q} \big (S _ {t + 1}, a, \mathbf {w} _ {t} \big) - \hat {q} \big (S _ {t}, A _ {t}, \mathbf {w} _ {t} \big). \quad \mathrm {(c o n t i n u i n g)}
$$

Note that this algorithm does not use importance sampling. In the tabular case it is clearthat this is appropriate because the only sample action is $A _ { t }$ , and in learning its value wedo not have to consider any other actions. With function approximation it is less clearbecause we might want to weight di↵erent state–action pairs di↵erently once they allcontribute to the same overall approximation. Proper resolution of this issue awaits amore thorough understanding of the theory of function approximation in reinforcementlearning.

In the multi-step generalizations of these algorithms, both the state-value and action-value algorithms involve importance sampling. The $n$ -step version of semi-gradient Sarsais

$$
\mathbf {w} _ {t + n} \doteq \mathbf {w} _ {t + n - 1} + \alpha \rho_ {t + 1} \dots \rho_ {t + n} \left[ G _ {t: t + n} - \hat {q} \left(S _ {t}, A _ {t}, \mathbf {w} _ {t + n - 1}\right) \right] \nabla \hat {q} \left(S _ {t}, A _ {t}, \mathbf {w} _ {t + n - 1}\right) \tag {11.6}
$$

with

$$
G _ {t: t + n} \doteq R _ {t + 1} + \dots + \gamma^ {n - 1} R _ {t + n} + \gamma^ {n} \hat {q} \left(S _ {t + n}, A _ {t + n}, \mathbf {w} _ {t + n - 1}\right), \text {o r} \tag {episodic}
$$

$$
G _ {t: t + n} \doteq R _ {t + 1} - \bar {R} _ {t} + \dots + R _ {t + n} - \bar {R} _ {t + n - 1} + \hat {q} (S _ {t + n}, A _ {t + n}, \mathbf {w} _ {t + n - 1}), (\mathrm {c o n t i n u i n g})
$$

where here we are being slightly informal in our treatment of the ends of episodes. In thefirst equation, the s for $k \geq T$ (where $T$ is the last time step of the episode) should be$\rho _ { k }$taken to be 1, and $G _ { t : t + n }$ should be taken to be $G _ { t }$ if $t + n \geq T$ .

Recall that we also presented in Chapter 7 an o↵-policy algorithm that does not involveimportance sampling at all: the $n$ -step tree-backup algorithm. Here is its semi-gradientversion:

$$
\mathbf {w} _ {t + n} \doteq \mathbf {w} _ {t + n - 1} + \alpha \left[ G _ {t: t + n} - \hat {q} \left(S _ {t}, A _ {t}, \mathbf {w} _ {t + n - 1}\right) \right] \nabla \hat {q} \left(S _ {t}, A _ {t}, \mathbf {w} _ {t + n - 1}\right), \tag {11.7}
$$

$$
G _ {t: t + n} \doteq \hat {q} \left(S _ {t}, A _ {t}, \mathbf {w} _ {t + n - 1}\right) + \sum_ {k = t} ^ {t + n - 1} \delta_ {k} \prod_ {i = t + 1} ^ {k} \gamma \pi \left(A _ {i} \mid S _ {i}\right), \tag {11.8}
$$

with $\delta _ { t }$ as defined at the top of this page for Expected Sarsa. We also defined in Chapter 7an algorithm that unifies all action-value algorithms: $n$ -step $Q ( \sigma )$ . We leave the semi-gradient form of that algorithm, and also of the $n$ -step state-value algorithm, as exercisesfor the reader.

Exercise 11.1 Convert the equation of $\boldsymbol { n }$ -step o↵-policy TD (7.9) to semi-gradient form.Give accompanying definitions of the return for both the episodic and continuing cases. $\boxed { \begin{array} { r l } \end{array} }$

⇤ Exercise 11.2 Convert the equations of $n$ -step $Q ( \sigma )$ (7.11 and 7.17) to semi-gradientform. Give definitions that cover both the episodic and continuing cases. ⇤

# 11.2 Examples of O↵-policy Divergence

In this section we begin to discuss the second part of the challenge of o↵-policy learningwith function approximation—that the distribution of updates does not match the on-policy distribution. We describe some instructive counterexamples to o↵-policy learning—cases where semi-gradient and other simple algorithms are unstable and diverge.

To establish intuitions, it is best to consider first a very simple example. Suppose,perhaps as part of a larger MDP, there are two states whose estimated values are ofthe functional form $w$ and $2 w$ , where the parameter vector w consists of only a singlecomponent $w$ . This occurs under linear function approximation if the feature vectorsfor the two states are each simple numbers (single-component vectors), in this case 1and 2. In the first state, only one action is available, and it results deterministically in atransition to the second state with a reward of 0:

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/557a70e74d607ce25590282b742cba1556908f879ee9432b727169afb4b555d6.jpg)


where the expressions inside the two circles indicate the two state’s values.

Suppose initially $w = 1 0$ . The transition will then be from a state of estimated value10 to a state of estimated value 20. It will look like a good transition, and $w$ will beincreased to raise the first state’s estimated value. If $\gamma$ is nearly 1, then the TD error willbe nearly 10, and, if $\alpha = 0 . 1$ , then $w$ will be increased to nearly 11 in trying to reduce theTD error. However, the second state’s estimated value will also be increased, to nearly22. If the transition occurs again, then it will be from a state of estimated value ⇡11 toa state of estimated value ⇡22, for a TD error of ⇡11—larger, not smaller, than before.It will look even more like the first state is undervalued, and its value will be increasedagain, this time to ${ \approx } 1 2 . 1$ . This looks bad, and in fact with further updates $w$ will divergeto infinity.

To see this definitively we have to look more carefully at the sequence of updates. TheTD error on a transition between the two states is

$$
\delta_ {t} = R _ {t + 1} + \gamma \hat {v} (S _ {t + 1}, \mathbf {w} _ {t}) - \hat {v} (S _ {t}, \mathbf {w} _ {t}) = 0 + \gamma 2 w _ {t} - w _ {t} = (2 \gamma - 1) w _ {t},
$$

and the o↵-policy semi-gradient TD(0) update (from (11.2)) is

$$
w _ {t + 1} = w _ {t} + \alpha \rho_ {t} \delta_ {t} \nabla \hat {v} (S _ {t}, w _ {t}) = w _ {t} + \alpha \cdot 1 \cdot (2 \gamma - 1) w _ {t} \cdot 1 = (1 + \alpha (2 \gamma - 1)) w _ {t}.
$$

Note that the importance sampling ratio, $\rho _ { t }$ , is 1 on this transition because there isonly one action available from the first state, so its probabilities of being taken underthe target and behavior policies must both be 1. In the final update above, the newparameter is the old parameter times a scalar constant, $1 + \alpha ( 2 \gamma - 1 )$ . If this constant isgreater than 1, then the system is unstable and $w$ will go to positive or negative infinitydepending on its initial value. Here this constant is greater than 1 whenever $\gamma > 0 . 5$ .Note that stability does not depend on the specific step size, as long as $\alpha > 0$ . Smaller orlarger step sizes would a↵ect the rate at which $w$ goes to infinity, but not whether it goesthere or not.

Key to this example is that the one transition occurs repeatedly without $w$ beingupdated on other transitions. This is possible under o↵-policy training because the

behavior policy might select actions on those other transitions which the target policynever would. For these transitions, $\rho _ { t }$ would be zero and no update would be made.Under on-policy training, however, $\rho _ { t }$ is always one. Each time there is a transition fromthe $w$ state to the $2 w$ state, increasing $w$ , there would also have to be a transition outof the $2 w$ state. That transition would reduce $w$ , unless it were to a state whose valuewas higher (because $\gamma < 1$ ) than $2 w$ , and then that state would have to be followed by astate of even higher value, or else again $w$ would be reduced. Each state can support theone before only by creating a higher expectation. Eventually the piper must be paid. Inthe on-policy case the promise of future reward must be kept and the system is kept incheck. But in the o↵-policy case, a promise can be made and then, after taking an actionthat the target policy never would, forgotten and forgiven.

This simple example communicates much of the reason why o↵-policy training can leadto divergence, but it is not completely convincing because it is not complete—it is just afragment of a complete MDP. Can there really be a complete system with instability? Asimple complete example of divergence is Baird’s counterexample. Consider the episodicseven-state, two-action MDP shown in Figure 11.1. The dashed action takes the systemto one of the six upper states with equal probability, whereas the solid action takes thesystem to the seventh state. The behavior policy $b$ selects the dashed and solid actionswith probabilities $\frac { 6 } { 7 }$ and $\frac { 1 } { 7 }$ , so that the next-state distribution under it is uniform (thesame for all nonterminal states), which is also the starting distribution for each episode.The target policy $\pi$ always takes the solid action, and so the on-policy distribution (for $\pi$ )is concentrated in the seventh state. The reward is zero on all transitions. The discountrate is $\gamma = 0 . 9 9$ .

Consider estimating the state-value under the linear parameterization indicated bythe expression shown in each state circle. For example, the estimated value of theleftmost state is $2 w _ { 1 } + w _ { 8 }$ , where the subscript corresponds to the component of the

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/85b009fc44c455083f7b43761872c42df9918e112777326189291bfed62723f0.jpg)



Figure 11.1: Baird’s counterexample. The approximate state-value function for this Markovprocess is of the form shown by the linear expressions inside each state. The solid action usuallyresults in the seventh state, and the dashed action usually results in one of the other six states,each with equal probability. The reward is always zero.


overall weight vector $\mathbf { w } \in \mathbb { R } ^ { 8 }$ ; this corresponds to a feature vector for the first statebeing $\mathbf { x } ( 1 ) = ( 2 , 0 , 0 , 0 , 0 , 0 , 0 , 1 ) ^ { \mid }$ . The reward is zero on all transitions, so the true valuefunction is $v _ { \pi } ( s ) = 0$ , for all $s$ , which can be exactly approximated if $\mathbf { w } = \mathbf { 0 }$ . In fact,there are many solutions, as there are more components to the weight vector (8) thanthere are nonterminal states (7). Moreover, the set of feature vectors, $\{ \mathbf { x } ( s ) : s \in \mathcal { S } \}$ , isa linearly independent set. In all these ways this task seems a favorable case for linearfunction approximation.

If we apply semi-gradient TD(0) to this problem (11.2), then the weights divergeto infinity, as shown in Figure 11.2 (left). The instability occurs for any positive stepsize, no matter how small. In fact, it even occurs if an expected update is done as indynamic programming (DP), as shown in Figure 11.2 (right). That is, if the weightvector, $\mathbf { w } _ { k }$ , is updated for all states at the same time in a semi-gradient way, using theDP (expectation-based) target:

$$
\mathbf {w} _ {k + 1} \doteq \mathbf {w} _ {k} + \frac {\alpha}{| \mathcal {S} |} \sum_ {s} \left(\mathbb {E} _ {\pi} \left[ R _ {t + 1} + \gamma \hat {v} \left(S _ {t + 1}, \mathbf {w} _ {k}\right) \mid S _ {t} = s \right] - \hat {v} (s, \mathbf {w} _ {k})\right) \nabla \hat {v} (s, \mathbf {w} _ {k}). \tag {11.9}
$$

In this case, there is no randomness and no asynchrony, just as in a classical DP update.The method is conventional except in its use of semi-gradient function approximation.Yet still the system is unstable.

If we alter just the distribution of DP updates in Baird’s counterexample, from theuniform distribution to the on-policy distribution (which generally requires asynchronousupdating), then convergence is guaranteed to a solution with error bounded by (9.14).This example is striking because the TD and DP methods used are arguably the simplest

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/a3f7dd8d2f8e16a9c6cb3f799c1abdf8872142be35b69ca8516dfcfbd29c76cf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/fa8e5977c7ccb2ec06c4d22abe0b420f55c98fdfb5fcd2bfcde9fb9d5a09886e.jpg)



Figure 11.2: Demonstration of instability on Baird’s counterexample. Shown are the evolutionof the components of the parameter vector $\mathbf { w }$ of the two semi-gradient algorithms. The step sizewas $\alpha = 0 . 0 1$ , and the initial weights were $\mathbf { w } = ( 1 , 1 , 1 , 1 , 1 , 1 , 1 0 , 1 ) ^ { \top }$ .


and best-understood bootstrapping methods, and the linear, semi-descent method used isarguably the simplest and best-understood kind of function approximation. The exampleshows that even the simplest combination of bootstrapping and function approximationcan be unstable if the updates are not done according to the on-policy distribution.

There are also counterexamples similar to Baird’s showing divergence for Q-learning.This is cause for concern because otherwise Q-learning has the best convergence guaranteesof all control methods. Considerable e↵ort has gone into trying to find a remedy tothis problem or to obtain some weaker, but still workable, guarantee. For example, itmay be possible to guarantee convergence of Q-learning as long as the behavior policy issu ciently close to the target policy, for example, when it is the $\varepsilon$ -greedy policy. To thebest of our knowledge, Q-learning has never been found to diverge in this case, but therehas been no theoretical analysis. In the rest of this section we present several other ideasthat have been explored.

Suppose that instead of taking just a step toward the expected one-step return on eachiteration, as in Baird’s counterexample, we actually change the value function all the wayto the best, least-squares approximation. Would this solve the instability problem? Ofcourse it would if the feature vectors, $\{ \mathbf { x } ( s ) : s \in \mathcal { S } \}$ , formed a linearly independent set,as they do in Baird’s counterexample, because then exact approximation is possible oneach iteration and the method reduces to standard tabular DP. But of course the pointhere is to consider the case when an exact solution is not possible. In this case stabilityis not guaranteed even when forming the best approximation at each iteration, as shownin the example.

Example 11.1: Tsitsiklis and Van Roy’s Counterexample This example showsthat linear function approximation would not work with DP even if the least-squares

solution was found at each step. The counterexample is formedby extending the $w$ -to- $2 w$ example (from earlier in this section)with a terminal state, as shown to the right. As before, theestimated value of the first state is $w$ , and the estimated valueof the second state is $2 w$ . The reward is zero on all transitions,so the true values are zero at both states, which is exactlyrepresentable with $w = 0$ . If we set $w _ { k + 1 }$ at each step soas to minimize the $\overline { { \mathrm { V E } } }$ between the estimated value and theexpected one-step return, then we have

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/f4db198c575d74bb9a9311d6a10216a64a03ded23b072f67f3ce223f9253176d.jpg)


$$
\begin{array}{l} w _ {k + 1} = \underset {w \in \mathbb {R}} {\arg \min } \sum_ {s \in \mathcal {S}} \left(\hat {v} (s, w) - \mathbb {E} _ {\pi} \left[ R _ {t + 1} + \gamma \hat {v} \left(S _ {t + 1}, w _ {k}\right) \mid S _ {t} = s \right]\right) ^ {2} \\ = \underset {w \in \mathbb {R}} {\arg \min } \left(w - \gamma 2 w _ {k}\right) ^ {2} + \left(2 w - (1 - \varepsilon) \gamma 2 w _ {k}\right) ^ {2} \\ = \frac {6 - 4 \varepsilon}{5} \gamma w _ {k}. \tag {11.10} \\ \end{array}
$$

$\{ w _ { k } \}$ diverges when $\begin{array} { r } { \gamma > \frac { 5 } { 6 - 4 \mathcal { E } } } \end{array}$ and $w _ { 0 } \neq 0$

Another way to try to prevent instability is to use special methods for functionapproximation. In particular, stability is guaranteed for function approximation methodsthat do not extrapolate from the observed targets. These methods, called averagers,include nearest neighbor methods and locally weighted regression, but not popularmethods such as tile coding and artificial neural networks (ANNs).

Exercise 11.3 (programming) Apply one-step semi-gradient Q-learning to Baird’s coun-terexample and show empirically that its weights diverge. ⇤

# 11.3 The Deadly Triad

Our discussion so far can be summarized by saying that the danger of instability anddivergence arises whenever we combine all of the following three elements, making upwhat we call the deadly triad:

Function approximation A powerful, scalable way of generalizing from a state spacemuch larger than the memory and computational resources (e.g., linear functionapproximation or ANNs).

Bootstrapping Update targets that include existing estimates (as in dynamic pro-gramming or TD methods) rather than relying exclusively on actual rewards andcomplete returns (as in MC methods).

O↵-policy training Training on a distribution of transitions other than that producedby the target policy. Sweeping through the state space and updating all statesuniformly, as in dynamic programming, does not respect the target policy and isan example of o↵-policy training.

In particular, note that the danger is not due to control or to generalized policy iteration.Those cases are more complex to analyze, but the instability arises in the simpler predictioncase whenever it includes all three elements of the deadly triad. The danger is also notdue to learning or to uncertainties about the environment, because it occurs just asstrongly in planning methods, such as dynamic programming, in which the environmentis completely known.

If any two elements of the deadly triad are present, but not all three, then instabilitycan be avoided. It is natural, then, to go through the three and see if there is any onethat can be given up.

Of the three, function approximation most clearly cannot be given up. We needmethods that scale to large problems and to great expressive power. We need at leastlinear function approximation with many features and parameters. State aggregation ornonparametric methods whose complexity grows with data are too weak or too expensive.Least-squares methods such as LSTD are of quadratic complexity and are therefore tooexpensive for large problems.

Doing without bootstrapping is possible, at the cost of computational and data e ciency.Perhaps most important are the losses in computational e ciency. Monte Carlo (non-bootstrapping) methods require memory to save everything that happens between making

each prediction and obtaining the final return, and all their computation is done once thefinal return is obtained. The cost of these computational issues is not apparent on serialvon Neumann computers, but would be on specialized hardware. With bootstrapping andeligibility traces (Chapter 12), data can be dealt with when and where it is generated,then need never be used again. The savings in communication and memory made possibleby bootstrapping are great.

The losses in data e ciency by giving up bootstrapping are also significant. Wehave seen this repeatedly, such as in Chapters 7 (Figure 7.2) and 9 (Figure 9.2), wheresome degree of bootstrapping performed much better than Monte Carlo methods onthe random-walk prediction task, and in Chapter 10 where the same was seen on theMountain-Car control task (Figure 10.4). Many other problems show much faster learningwith bootstrapping (e.g., see Figure 12.14). Bootstrapping often results in faster learningbecause it allows learning to take advantage of the state property, the ability to recognizea state upon returning to it. On the other hand, bootstrapping can impair learning onproblems where the state representation is poor and causes poor generalization (e.g.,this seems to be the case on Tetris, see S¸im¸sek, Alg´orta, and Kothiyal, 2016). A poorstate representation can also result in bias; this is the reason for the poorer bound onthe asymptotic approximation quality of bootstrapping methods (Equation 9.14). Onbalance, the ability to bootstrap has to be considered extremely valuable. One maysometimes choose not to use it by selecting long $n$ -step updates (or a large bootstrappingparameter, $\lambda \approx 1$ ; see Chapter 12) but often bootstrapping greatly increases e ciency. Itis an ability that we would very much like to keep in our toolkit.

Finally, there is o↵-policy learning; can we give that up? On-policy methods are oftenadequate. For model-free reinforcement learning, one can simply use Sarsa rather thanQ-learning. O↵-policy methods free behavior from the target policy. This could beconsidered an appealing convenience but not a necessity. However, o↵-policy learningis essential to other anticipated use cases, cases that we have not yet mentioned in thisbook but may be important to the larger goal of creating a powerful intelligent agent.

In these use cases, the agent learns not just a single value function and single policy,but large numbers of them in parallel. There is extensive psychological evidence thatpeople and animals learn to predict many di↵erent sensory events, not just rewards. Wecan be surprised by unusual events, and correct our predictions about them, even ifthey are of neutral valence (neither good nor bad). This kind of prediction presumablyunderlies predictive models of the world such as are used in planning. We predict whatwe will see after eye movements, how long it will take to walk home, the probability ofmaking a jump shot in basketball, and the satisfaction we will get from taking on a newproject. In all these cases, the events we would like to predict depend on our acting ina certain way. To learn them all, in parallel, requires learning from the one stream ofexperience. There are many target policies, and thus the one behavior policy cannotequal all of them. Yet parallel learning is conceptually possible because the behaviorpolicy may overlap in part with many of the target policies. To take full advantage ofthis requires o↵-policy learning.

# 11.4 Linear Value-function Geometry

To better understand the stability challenge of o↵-policy learning, it is helpful to thinkabout value function approximation more abstractly and independently of how learningis done. We can imagine the space of all possible state-value functions—all functionsfrom states to real numbers $v : \mathcal { S }  \mathbb { R }$ . Most of these value functions do not correspondto any policy. More important for our purposes is that most are not representable by thefunction approximator, which by design has far fewer parameters than there are states.

Given an enumeration of the state space $\mathcal { S } = \{ s _ { 1 } , s _ { 2 } , . . . , s _ { | \mathcal { S } | } \}$ , any value function $\boldsymbol { v }$corresponds to a vector listing the value of each state in order $[ v ( s _ { 1 } ) , v ( s _ { 2 } ) , \ldots , v ( s _ { | \mathcal { S } | } ) ] ^ { \top }$ .This vector representation of a value function has as many components as there arestates. In most cases where we want to use function approximation, this would be fartoo many components to represent the vector explicitly. Nevertheless, the idea of thisvector is conceptually useful. In the following, we treat a value function and its vectorrepresentation interchangeably.

To develop intuitions, consider the case with three states $\boldsymbol { \mathcal { S } } = \{ s _ { 1 } , s _ { 2 } , s _ { 3 } \}$ and twoparameters ${ \bf w } = ( w _ { 1 } , w _ { 2 } ) ^ { \top }$ . We can then view all value functions/vectors as points ina three-dimensional space. The parameters provide an alternative coordinate systemover a two-dimensional subspace. Any weight vector ${ \bf w } = ( w _ { 1 } , w _ { 2 } ) ^ { \top }$ is a point in thetwo-dimensional subspace and thus also a complete value function $v _ { \bf w }$ that assigns valuesto all three states. With general function approximation the relationship between thefull space and the subspace of representable functions could be complex, but in the caseof linear value-function approximation the subspace is a simple plane, as suggested byFigure 11.3.

Now consider a single fixed policy $\pi$ . We assume that its true value function, $v _ { \pi }$ , is toocomplex to be represented exactly as an approximation. Thus $v _ { \pi }$ is not in the subspace;in the figure it is depicted as being above the planar subspace of representable functions.

If $v _ { \pi }$ cannot be represented exactly, what representable value function is closest toit? This turns out to be a subtle question with multiple answers. To begin, we needa measure of the distance between two value functions. Given two value functions $v _ { 1 }$and $v _ { 2 }$ , we can talk about the vector di↵erence between them, $v = v _ { 1 } - v _ { 2 }$ . If $v$ is small,then the two value functions are close to each other. But how are we to measure the sizeof this di↵erence vector? The conventional Euclidean norm is not appropriate because,as discussed in Section 9.2, some states are more important than others because theyoccur more frequently or because we are more interested in them (Section 9.11). Asin Section 9.2, let us use the distribution $\mu : \mathcal { S }  [ 0 , 1 ]$ to specify the degree to whichwe care about di↵erent states being accurately valued (often taken to be the on-policydistribution). We can then define the distance between value functions using the norm

$$
\| v \| _ {\mu} ^ {2} \doteq \sum_ {s \in \mathcal {S}} \mu (s) v (s) ^ {2}. \tag {11.11}
$$

Note that the $\overline { { \mathrm { V E } } }$ from Section 9.2 can be written simply using this norm as $\overline { { \mathrm { V E } } } ( \mathbf { w } ) =$$\| v _ { \mathbf { w } } - v _ { \pi } \| _ { \mu } ^ { 2 }$ . For any value function $\boldsymbol { v }$ , the operation of finding its closest value functionin the subspace of representable value functions is a projection operation. We define a

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/2414e6af2a5ff1f9b027ee23ff13913d5ef3bb88629635aabf9c66846f0eb185.jpg)



Figure 11.3: The geometry of linear value-function approximation. Shown is the three-ones, through discretization, stdimensional space of all value functions over three states, while shown as a plane is the subspace ofof the state spaceall value functions representable by a linear function approximator with parameter ${ \bf w } = ( w _ { 1 } , w _ { 2 } ) ^ { \top }$ .The true value function $v _ { \pi }$ is in the larger space and can be projected down (into the subspace,using a projection operator $1 1$ ) to its best approximation in the value error (VE) sense. TheA more general and flexiblbest approximators in the Bellman error (BE), projected Bellman error (PBE), and temporaldi↵erence error (TDE) senses are all potentially di↵erent and are shown in the lower right. (VE,BE, and PBE are all treated as the corresponding vectors in this figure.) The Bellman operatorare then changed to reshape thtakes a value function in the plane to one outside, which can then be projected back. If youfunction. We denote the paramiteratively applied the Bellman operator outside the space (shown in gray above) you wouldreach the true value function, as in conventional dynamic programming. If instead you keptprojecting back into the subspace at each step, as in the lower step shown in gray, then the fixedpoint would be the point of vector-zero PBE.


projection operator $\mathrm { I I }$ where ✓ 2 R , with n ⌧ |S|that takes an arbitrary value function to the representable functionthat is closest in our norm:

$$
\Pi v \doteq v _ {\mathbf {w}} \text {w h e r e} \mathbf {w} = \underset {\mathbf {w} \in \mathbb {R} ^ {d}} {\arg \min } \| v - v _ {\mathbf {w}} \| _ {\mu} ^ {2}. \tag {11.12}
$$

The representable value function that is closest to the true value function $v _ { \pi }$ is thus itsprojection, $\scriptstyle { \mathrm { I I } } v _ { \pi }$ for things like the discount-rat, as suggested in Figure 11.3. This is the solution asymptotically foundAn important special caseby Monte Carlo methods, albeit often very slowly. The projection operation is discussedmore fully in the box on the next page.

TD methods find di↵erent solutions. To understand their rationale, recall that theBellman equation for value function $v _ { \pi }$ is

$$
v _ {\pi} (s) = \sum_ {a} \pi (a | s) \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r \mid s, a\right) \left[ r + \gamma v _ {\pi} \left(s ^ {\prime}\right) \right], \quad \text {f o r a l l} s \in \mathcal {S}. \tag {11.16}
$$

# The projection matrix

For a linear function approximator, the projection operation is linear, which impliesthat it can be represented as an $| \mathcal { S } | \times | \mathcal { S } |$ matrix:

$$
\Pi \doteq \mathbf {X} \left(\mathbf {X} ^ {\top} \mathbf {D} \mathbf {X}\right) ^ {- 1} \mathbf {X} ^ {\top} \mathbf {D}, \tag {11.13}
$$

where, as in Section 9.4, $\mathbf { D }$ denotes the $| \mathcal { S } | \times | \mathcal { S } |$ diagonal matrix with the $\mu ( s )$on the diagonal, and $\mathbf { X }$ denotes the $| \mathcal { S } | \times d$ matrix whose rows are the featurevectors $\mathbf { x } ( s ) ^ { \top }$ , one for each state $s$ . If the inverse in (11.13) does not exist, then thepseudoinverse is substituted. Using these matrices, the squared norm of a vectorcan be written

$$
\left\| v \right\| _ {\mu} ^ {2} = v ^ {\top} \mathbf {D} v, \tag {11.14}
$$

and the approximate linear value function can be written

$$
v _ {\mathbf {w}} = \mathbf {X} \mathbf {w}. \tag {11.15}
$$

The true value function $v _ { \pi }$ is the only value function that solves (11.16) exactly. If anapproximate value function $v _ { \bf w }$ were substituted for $v _ { \pi }$ , the di↵erence between the rightand left sides of the modified equation could be used as a measure of how far o↵ $v _ { \bf w }$ isfrom $v _ { \pi }$ . We call this the Bellman error at state $s$ :

$$
\begin{array}{l} \bar {\delta} _ {\mathbf {w}} (s) \doteq \left(\sum_ {a} \pi (a | s) \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r \mid s, a\right) [ r + \gamma v _ {\mathbf {w}} \left(s ^ {\prime}\right) ]\right) - v _ {\mathbf {w}} (s) (11.17) \\ = \mathbb {E} _ {\pi} \left[ R _ {t + 1} + \gamma v _ {\mathbf {w}} \left(S _ {t + 1}\right) - v _ {\mathbf {w}} \left(S _ {t}\right) \mid S _ {t} = s, A _ {t} \sim \pi \right], (11.18) \\ \end{array}
$$

which shows clearly the relationship of the Bellman error to the TD error (11.3). TheBellman error is the expectation of the TD error.

The vector of all the Bellman errors, at all states, $\delta _ { \mathbf { w } } \in \mathbb { R } ^ { | \mathcal { S } | }$ , is called the Bellmanerror vector (shown as BE in Figure 11.3). The overall size of this vector, in the norm, isan overall measure of the error in the value function, called the mean square Bellmanerror :

$$
\overline {{\mathrm {B E}}} (\mathbf {w}) = \left\| \bar {\delta} _ {\mathbf {w}} \right\| _ {\mu} ^ {2}. \tag {11.19}
$$

It is not possible in general to reduce the $\overline { { \mathrm { B E } } }$ to zero (at which point $v _ { \mathbf { w } } = v _ { \pi }$ ), but forlinear function approximation there is a unique value of w for which the $\overline { { \mathrm { B E } } }$ is minimized.This point in the representable-function subspace (labeled $\operatorname* { m i n } \overline { { \mathrm { B E } } }$ in Figure 11.3) isdi↵erent in general from that which minimizes the $\overline { { \mathrm { V E } } }$ (shown as $\boldsymbol { \mathrm { l I } } \boldsymbol { v } _ { \pi }$ ). Methods thatseek to minimize the $\mathrm { B E }$ are discussed in the next two sections.

The Bellman error vector is shown in Figure 11.3 as the result of applying the Bellmanoperator $B _ { \pi } : \mathbb { R } ^ { | \mathcal { S } | }  \mathbb { R } ^ { | \mathcal { S } | }$ to the approximate value function. The Bellman operator is

defined by

$$
\left(B _ {\pi} v\right) (s) \doteq \sum_ {a} \pi (a | s) \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r \mid s, a\right) [ r + \gamma v \left(s ^ {\prime}\right) ], \tag {11.20}
$$

for all $s \in \mathcal { S }$ , $v : \mathcal { S }  \mathbb { R }$ . The Bellman error vector for $v _ { \bf w }$ can be written $\delta _ { \mathbf { w } } = B _ { \pi } v _ { \mathbf { w } } - v _ { \mathbf { w } }$

If the Bellman operator is applied to a value function in the representable subspace,then, in general, it will produce a new value function that is outside the subspace, assuggested in the figure. In dynamic programming (without function approximation), thisoperator is applied repeatedly to the points outside the representable space, as suggestedby the gray arrows in the top of Figure 11.3. Eventually that process converges to thetrue value function $v _ { \pi }$ , the only fixed point for the Bellman operator, the only valuefunction for which

$$
v _ {\pi} = B _ {\pi} v _ {\pi}, \tag {11.21}
$$

which is just another way of writing the Bellman equation for $\pi$ (11.16).

With function approximation, however, the intermediate value functions lying outsidethe subspace cannot be represented. The gray arrows in the upper part of Figure 11.3cannot be followed because after the first update (dark line) the value function mustbe projected back into something representable. The next iteration then begins withinthe subspace; the value function is again taken outside of the subspace by the Bellmanoperator and then mapped back by the projection operator, as suggested by the lowergray arrow and line. Following these arrows is a DP-like process with approximation.

In this case we are interested in the projection of the Bellman error vector back intothe representable space. This is the projected Bellman error vector $\Pi \delta _ { \mathbf { w } }$ , shown inFigure 11.3 as PBE. The size of this vector, in the norm, is another measure of error inthe approximate value function. For any approximate value function $v _ { \mathbf { w } }$ , we define themean square Projected Bellman error, denoted $\overline { { \mathrm { P B E } } }$ , as

$$
\overline {{\mathrm {P B E}}} (\mathbf {w}) = \left\| \Pi \bar {\delta} _ {\mathbf {w}} \right\| _ {\mu} ^ {2}. \tag {11.22}
$$

With linear function approximation there always exists an approximate value function(within the subspace) with zero $\overline { { \mathrm { P B E } } }$ ; this is the TD fixed point, wTD, introduced inSection 9.4. As we have seen, this point is not always stable under semi-gradient TDmethods and o↵-policy training. As shown in the figure, this value function is generallydi↵erent from those minimizing $\mathrm { V E }$ or $\mathrm { B E }$ . Methods that are guaranteed to converge toit are discussed in Sections 11.7 and 11.8.

# 11.5 Gradient Descent in the Bellman Error

Armed with a better understanding of value function approximation and its variousobjectives, we return now to the challenge of stability in o↵-policy learning. We wouldlike to apply the approach of stochastic gradient descent (SGD, Section 9.3), in whichupdates are made that in expectation are equal to the negative gradient of an objective

function. These methods always go downhill (in expectation) in the objective and becauseof this are typically stable with excellent convergence properties. Among the algorithmsinvestigated so far in this book, only the Monte Carlo methods are true SGD methods.These methods converge robustly under both on-policy and o↵-policy training as wellas for general nonlinear (di↵erentiable) function approximators, though they are oftenslower than semi-gradient methods with bootstrapping, which are not SGD methods.Semi-gradient methods may diverge under o↵-policy training, as we have seen earlier inthis chapter, and under contrived cases of nonlinear function approximation (Tsitsiklisand Van Roy, 1997). With a true SGD method such divergence would not be possible.

The appeal of SGD is so strong that great e↵ort has gone into finding a practicalway of harnessing it for reinforcement learning. The starting place of all such e↵orts isthe choice of an error or objective function to optimize. In this and the next sectionwe explore the origins and limits of the most popular proposed objective function, thatbased on the Bellman error introduced in the previous section. Although this has been apopular and influential approach, the conclusion that we reach here is that it is a misstepand yields no good learning algorithms. On the other hand, this approach fails in aninteresting way that o↵ers insight into what might constitute a good approach.

To begin, let us consider not the Bellman error, but something more immediateand naive. Temporal di↵erence learning is driven by the TD error. Why not take theminimization of the expected square of the TD error as the objective? In the generalfunction-approximation case, the one-step TD error with discounting is

$$
\delta_ {t} = R _ {t + 1} + \gamma \hat {v} (S _ {t + 1}, \mathbf {w} _ {t}) - \hat {v} (S _ {t}, \mathbf {w} _ {t}).
$$

A possible objective function then is what one might call the mean square TD error :

$$
\begin{array}{l} \overline {{\mathrm {T D E}}} (\mathbf {w}) = \sum_ {s \in \mathcal {S}} \mu (s) \mathbb {E} \left[ \delta_ {t} ^ {2} \mid S _ {t} = s, A _ {t} \sim \pi \right] \\ = \sum_ {s \in \mathbb {S}} \mu (s) \mathbb {E} \left[ \rho_ {t} \delta_ {t} ^ {2} \mid S _ {t} = s, A _ {t} \sim b \right] \\ = \mathbb {E} _ {b} \left[ \rho_ {t} \delta_ {t} ^ {2} \right]. \quad (\text {i f} \mu \text {i s t h e d i s t r i b u t i o n e n c o u n t e r e d u n d e r} b) \\ \end{array}
$$

The last equation is of the form needed for SGD; it gives the objective as an expectationthat can be sampled from experience (remember the experience is due to the behaviorpolicy $b$ ). Thus, following the standard SGD approach, one can derive the per-step updatebased on a sample of this expected value:

$$
\begin{array}{l} \mathbf {w} _ {t + 1} = \mathbf {w} _ {t} - \frac {1}{2} \alpha \nabla \left(\rho_ {t} \delta_ {t} ^ {2}\right) \\ = \mathbf {w} _ {t} - \alpha \rho_ {t} \delta_ {t} \nabla \delta_ {t} \\ = \mathbf {w} _ {t} + \alpha \rho_ {t} \delta_ {t} \left(\nabla \hat {v} \left(S _ {t}, \mathbf {w} _ {t}\right) - \gamma \nabla \hat {v} \left(S _ {t + 1}, \mathbf {w} _ {t}\right)\right), \tag {11.23} \\ \end{array}
$$

which you will recognize as the same as the semi-gradient TD algorithm (11.2) except forthe additional final term. This term completes the gradient and makes this a true SGDalgorithm with excellent convergence guarantees. Let us call this algorithm the naive

residual-gradient algorithm (after Baird, 1995). Although the naive residual-gradientalgorithm converges robustly, it does not necessarily converge to a desirable place.

# Example 11.2: A-split example,showing the naivet´e of the naive residual-gradient algorithm

Consider the three-state episodic MRP shown to the right.Episodes begin in state A and then ‘split’ stochastically, halfthe time going to $\textsf { B }$ (and then invariably going on to terminatewith a reward of 1) and half the time going to state C (andthen invariably terminating with a reward of zero). Reward forthe first transition, out of A, is always zero whichever way theepisode goes. As this is an episodic problem, we can take $\gamma$ tobe 1. We also assume on-policy training, so that $\rho _ { t }$ is always

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/331b95c120b0cead2ccccff0e9067a3438d328c18fc757ee3521153548f4953b.jpg)


1, and tabular function approximation, so that the learning algorithms are free togive arbitrary, independent values to all three states. Thus, this should be an easyproblem.

What should the values be? From A, half the time the return is 1, and half thetime the return is 0; A should have value $\frac { \mathrm { 1 } } { \mathrm { 2 } }$ . From $\textsf { B }$ the return is always 1, so itsvalue should be 1, and similarly from $\mathsf { C }$ the return is always 0, so its value shouldbe 0. These are the true values and, as this is a tabular problem, all the methodspresented previously converge to them exactly.

However, the naive residual-gradient algorithm finds di↵erent values for $\textsf { B }$ andC. It converges with $\textsf { B }$ having a value of $\frac 3 4$ and $\mathsf { C }$ having a value of $\frac { 1 } { 4 }$ (A convergescorrectly to $\frac { 1 } { 2 }$ ). These are in fact the values that minimize the $\overline { { \mathrm { T D E } } }$ .

Let us compute the $\overline { { \mathrm { T D E } } }$ for these values. The first transition of each episode iseither up from A’s $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ to $\textsf { B }$ ’s $\frac { 3 } { 4 }$ , a change of $\frac { 1 } { 4 }$ , or down from A’s $\frac { 1 } { 2 }$ to $\mathsf { C }$ ’s $\textstyle { \frac { 1 } { 4 } }$ , a changeof $- { \frac { 1 } { 4 } }$ . Because the reward is zero on these transitions, and $\gamma = 1$ , these changes arethe TD errors, and thus the squared TD error is always $\textstyle { \frac { 1 } { 1 6 } }$ on the first transition.The second transition is similar; it is either up from $\textsf { B }$ ’s $\frac 3 4$ to a reward of 1 (and aterminal state of value 0), or down from $\mathsf { C }$ ’s $\textstyle { \frac { 1 } { 4 } }$ to a reward of 0 (again with a terminalstate of value 0). Thus, the TD error is alwasecond step. Thus, for this set of values, the $\pm \frac { 1 } { 4 }$ , for a squared e on both steps is r of . $\textstyle { \frac { 1 } { 1 6 } }$ on the$\overline { { \mathrm { T D E } } }$ $\textstyle { \frac { 1 } { 1 6 } }$

Now let’s compute the $\overline { { \mathrm { T D E } } }$ for the true values ( $\textsf { B }$ at 1, $\mathsf { C }$ at $0$ , and $\mathsf { A }$ at $\begin{array} { l } { { \frac { 1 } { 2 } } } \end{array}$ ). In thiscase the first transition is either from $\frac { 1 } { 2 }$ up to 1, at $\textsf { B }$ , or from $\frac { \mathrm { 1 } } { \mathrm { 2 } }$ down to 0, at $\mathsf { C }$ ; ineither case the absolute error is $\frac { \mathrm { 1 } } { \mathrm { 2 } }$ and the squared error is $\frac { 1 } { 4 }$ . The second transitionhas zero error because the starting value, either 1 or 0 depending on whether thetransition is from $\textsf { B }$ or $\mathsf { C }$ , always exactly matches the immediate reward and return.Thus the squared TD error is $\frac { 1 } { 4 }$ on the first transition and 0 on the second, for amean reward over the two transitions of $\frac { 1 } { 8 }$ . As $\frac { 1 } { 8 }$ is bigger that $\textstyle { \frac { 1 } { 1 6 } }$ , this solution isworse according to the $\overline { { \mathrm { T D E } } }$ . On this simple problem, the true values do not havethe smallest $\overline { { \mathrm { T D E } } }$ .

A tabular representation is used in the A-split example, so the true state values canbe exactly represented, yet the naive residual-gradient algorithm finds di↵erent values,and these values have lower $\overline { { \mathrm { T D E } } }$ than do the true values. Minimizing the $\overline { { \mathrm { T D E } } }$ is naive;by penalizing all TD errors it achieves something more like temporal smoothing thanaccurate prediction.

A better idea would seem to be minimizing the mean square Bellman error ( $\mathrm { B E }$ ). Ifthe exact values are learned, the Bellman error is zero everywhere. Thus, a Bellman-error-minimizing algorithm should have no trouble with the A-split example. We cannotexpect to achieve zero Bellman error in general, as it would involve finding the true valuefunction, which we presume is outside the space of representable value functions. Butgetting close to this ideal is a natural-seeming goal. As we have seen, the Bellman erroris also closely related to the TD error. The Bellman error for a state is the expected TDerror in that state. So let’s repeat the derivation above with the expected TD error (allexpectations here are implicitly conditional on $S _ { t }$ ):

$$
\begin{array}{l} \mathbf {w} _ {t + 1} = \mathbf {w} _ {t} - \frac {1}{2} \alpha \nabla \left(\mathbb {E} _ {\pi} [ \delta_ {t} ] ^ {2}\right) \\ \mathbf {\alpha} = \mathbf {w} _ {t} - \frac {1}{2} \alpha \nabla (\mathbb {E} _ {b} [ \rho_ {t} \delta_ {t} ] ^ {2}) \\ = \mathbf {w} _ {t} - \alpha \mathbb {E} _ {b} [ \rho_ {t} \delta_ {t} ] \nabla \mathbb {E} _ {b} [ \rho_ {t} \delta_ {t} ] \\ = \mathbf {w} _ {t} - \alpha \mathbb {E} _ {b} \left[ \rho_ {t} \left(R _ {t + 1} + \gamma \hat {v} \left(S _ {t + 1}, \mathbf {w}\right) - \hat {v} \left(S _ {t}, \mathbf {w}\right)\right) \right] \mathbb {E} _ {b} \left[ \rho_ {t} \nabla \delta_ {t} \right] \\ = \mathbf {w} _ {t} + \alpha \Big [ \mathbb {E} _ {b} \big [ \rho_ {t} (R _ {t + 1} + \gamma \hat {v} (S _ {t + 1}, \mathbf {w})) \big ] - \hat {v} (S _ {t}, \mathbf {w}) \Big ] \Big [ \nabla \hat {v} (S _ {t}, \mathbf {w}) - \gamma \mathbb {E} _ {b} \big [ \rho_ {t} \nabla \hat {v} (S _ {t + 1}, \mathbf {w}) \big ] \Big ]. \\ \end{array}
$$

This update and various ways of sampling it are referred to as the residual-gradientalgorithm. If you simply used the sample values in all the expectations, then the equationabove reduces almost exactly to (11.23), the naive residual-gradient algorithm.1 Butthis is naive, because the equation above involves the next state, $S _ { t + 1 }$ , appearing in twoexpectations that are multiplied together. To get an unbiased sample of the product,two independent samples of the next state are required, but during normal interactionwith an external environment only one is obtained. One expectation or the other can besampled, but not both.

There are two ways to make the residual-gradient algorithm work. One is in the caseof deterministic environments. If the transition to the next state is deterministic, thenthe two samples will necessarily be the same, and the naive algorithm is valid. Theother way is to obtain two independent samples of the next state, $S _ { t + 1 }$ , from $S _ { t }$ , one forthe first expectation and another for the second expectation. In real interaction withan environment, this would not seem possible, but when interacting with a simulatedenvironment, it is. One simply rolls back to the previous state and obtains an alternatenext state before proceeding forward from the first next state. In either of these cases theresidual-gradient algorithm is guaranteed to converge to a minimum of the $\mathrm { B E }$ under theusual conditions on the step-size parameter. As a true SGD method, this convergence is

robust, applying to both linear and nonlinear function approximators. In the linear case,convergence is always to the unique w that minimizes the $\mathrm { B E }$ .

However, there remain at least three ways in which the convergence of the residual-gradient method is unsatisfactory. The first of these is that empirically it is slow, muchslower that semi-gradient methods. Indeed, proponents of this method have proposedincreasing its speed by combining it with faster semi-gradient methods initially, thengradually switching over to residual gradient for the convergence guarantee (Baird andMoore, 1999). The second way in which the residual-gradient algorithm is unsatisfactoryis that it still seems to converge to the wrong values. It does get the right values in alltabular cases, such as the A-split example, as for those an exact solution to the Bellman

# Example 11.3: A-presplit example, a counterexample for the BE

Consider the three-state episodic MRP shown to theright: Episodes start in either A1 or A2, with equalprobability. These two states look exactly the same tothe function approximator, like a single state A whosefeature representation is distinct from and unrelated tothe feature representation of the other two states, B andC, which are also distinct from each other. Specifically,the parameter of the function approximator has three

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/5276ef0bd589bd2fc9822a77262dd120ca4212d1ffa829f6ab98f6de1e1495c0.jpg)


components, one giving the value of state B, one giving the value of state C, and onegiving the value of both states A1 and A2. Other than the selection of the initialstate, the system is deterministic. If it starts in A1, then it transitions to B with areward of 0 and then on to termination with a reward of 1. If it starts in A2, then ittransitions to $\mathsf { C }$ , and then to termination, with both rewards zero.

To a learning algorithm, seeing only the features, the system looks identical tothe A-split example. The system seems to always start in A, followed by either$\textsf { B }$ or $\mathsf { C }$ with equal probability, and then terminating with a 1 or a 0 dependingdeterministically on the previous state. As in the A-split example, the true valuesof $\textsf { B }$ and $\mathsf { C }$ are 1 and 0, and the best shared value of A1 and A2 is $\frac { 1 } { 2 }$ , by symmetry.

Because this problem appears externally identical to the A-split example, wealready know what values will be found by the algorithms. Semi-gradient TDconverges to the ideal values just mentioned, while the naive residual-gradientalgorithm converges to values of $\frac 3 4$ and $\frac { 1 } { 4 }$ for $\textsf { B }$ and $\mathsf { C }$ respectively. All statetransitions are deterministic, so the non-naive residual-gradient algorithm will alsoconverge to these values (it is the same algorithm in this case). It follows then thatthis ‘naive’ solution must also be the one that minimizes the $\overline { { \mathrm { B E } } }$ , and so it is. On adeterministic problem, the Bellman errors and TD errors are all the same, so the$\mathrm { B E }$ is always the same as the $\mathrm { T D E }$ . Optimizing the $\mathrm { B E }$ on this example gives rise tothe same failure mode as with the naive residual-gradient algorithm on the A-splitexample.

equation is possible. But if we examine examples with genuine function approximation,then the residual-gradient algorithm, and indeed the $\overline { { \mathrm { B E } } }$ objective, seem to find thewrong value functions. One of the most telling such examples is the variation on theA-split example known as the A-presplit example, shown on the preceding page, in whichthe residual-gradient algorithm finds the same poor solution as its naive version. Thisexample shows intuitively that minimizing the $\overline { { \mathrm { B E } } }$ (which the residual-gradient algorithmsurely does) may not be a desirable goal.

The third way in which the convergence of the residual-gradient algorithm is notsatisfactory is explained in the next section. Like the second way, the third way is alsoa problem with the $\overline { { \mathrm { B E } } }$ objective itself rather than with any particular algorithm forachieving it.

# 11.6 The Bellman Error is Not Learnable

The concept of learnability that we introduce in this section is di↵erent from thatcommonly used in machine learning. There, a hypothesis is said to be “learnable” ifit is e ciently learnable, meaning that it can be learned within a polynomial ratherthan an exponential number of examples. Here we use the term in a more basic way,to mean learnable at all, with any amount of experience. It turns out many quantitiesof apparent interest in reinforcement learning cannot be learned even from an infiniteamount of experiential data. These quantities are well defined and can be computedgiven knowledge of the internal structure of the environment, but cannot be computedor estimated from the observed sequence of feature vectors, actions, and rewards.2 Wesay that they are not learnable. It will turn out that the Bellman error objective ( $\mathrm { B E }$ )introduced in the last two sections is not learnable in this sense. That the Bellman errorobjective cannot be learned from the observable data is probably the strongest reasonnot to seek it.

To make the concept of learnability clear, let’s start with some simple examples.Consider the two Markov reward processes $^ { 3 }$ (MRPs) diagrammed below:

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/a717240df15ce10fac272678dacaf1a509d0c1c10a61d0f9945a9b6b382200e7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/7f35fe75d9a0b61c02a4e7e01a73ba37af85da6e7dacddacba8bfad8be83e100.jpg)


Where two edges leave a state, both transitions are assumed to occur with equal probability,and the numbers indicate the reward received. All the states appear the same; they allproduce the same single-component feature vector $x = 1$ and have approximated value$w$ . Thus, the only varying part of the data trajectory is the reward sequence. The leftMRP stays in the same state and emits an endless stream of 0s and 2s at random, eachwith 0.5 probability. The right MRP, on every step, either stays in its current state or

switches to the other, with equal probability. The reward is deterministic in this MRP,always a 0 from one state and always a 2 from the other, but because the each stateis equally likely on each step, the observable data is again an endless stream of 0s and2s at random, identical to that produced by the left MRP. (We can assume the rightMRP starts in one of two states at random with equal probability.) Thus, even givenan infinite amount of data, it would not be possible to tell which of these two MRPswas generating it. In particular, we could not tell if the MRP has one state or two, isstochastic or deterministic. These things are not learnable.

This pair of MRPs also illustrates that the $\overline { { \mathrm { V E } } }$ objective (9.1) is not learnable. If$\gamma = 0$ , then the true values of the three states (in both MRPs), left to right, are 1, 0,and 2. Suppose $w = 1$ . Then the $\overline { { \mathrm { V E } } }$ is 0 for the left MRP and 1 for the right MRP.Because the $\overline { { \mathrm { V E } } }$ is di↵erent in the two problems, yet the data generated has the samedistribution, the $\overline { { \mathrm { V E } } }$ cannot be learned. The $\overline { { \mathrm { V E } } }$ is not a unique function of the datadistribution. And if it cannot be learned, then how could the $\mathrm { V E }$ possibly be useful asan objective for learning?

If an objective cannot be learned, it does indeed draw its utility into question. Inthe case of the $\overline { { \mathrm { V E } } }$ , however, there is a way out. Note that the same solution, $w = 1$ ,is optimal for both MRPs above (assuming $\mu$ is the same for the two indistinguishablestates in the right MRP). Is this a coincidence, or could it be generally true that allMDPs with the same data distribution also have the same optimal parameter vector? Ifthis is true—and we will show next that it is—then the $\overline { { \mathrm { V E } } }$ remains a usable objective.The $\overline { { \mathrm { V E } } }$ is not learnable, but the parameter that optimizes it is!

To understand this, it is useful to bring in another natural objective function, thistime one that is clearly learnable. One error that is always observable is that between thevalue estimate at each time and the return from that time. The mean square return error,denoted $\mathrm { R E }$ , is the expectation, under , of the square of this error. In the on-policy case$\mu$the $\overline { { \mathrm { R E } } }$ can be written

$$
\begin{array}{l} \overline {{\mathrm {R E}}} (\mathbf {w}) = \mathbb {E} \left[ \left(G _ {t} - \hat {v} (S _ {t}, \mathbf {w})\right) ^ {2} \right] \\ = \overline {{\mathrm {V E}}} (\mathbf {w}) + \mathbb {E} \left[ \left(G _ {t} - v _ {\pi} (S _ {t})\right) ^ {2} \right]. \tag {11.24} \\ \end{array}
$$

Thus, the two objectives are the same except for a variance term that does not depend onthe parameter vector. The two objectives must therefore have the same optimal parametervalue w⇤. The overall relationships are summarized in the left side of Figure 11.4.

⇤ Exercise 11.4 Prove (11.24). Hint: Write the $\overline { { \mathrm { R E } } }$ as an expectation over possible states$s$ of the expectation of the squared error given that $S _ { t } = s$ . Then add and subtract thetrue value of state $s$ from the error (before squaring), grouping the subtracted true valuewith the return and the added true value with the estimated value. Then, if you expandthe square, the most complex term will end up being zero, leaving you with (11.24). $\boxed { \begin{array} { r l } \end{array} }$ 1

Now let us return to the $\mathrm { B E }$ . The $\overline { { \mathrm { B E } } }$ is like the $\overline { { \mathrm { V E } } }$ in that it can be computed fromknowledge of the MDP but is not learnable from data. But it is not like the $\mathrm { V E }$ in that itsminimum solution is not learnable. The box on the next page presents a counterexample—two MRPs that generate the same data distribution but whose minimizing parametervector is di↵erent, proving that the optimal parameter vector is not a function of the

# Example 11.4: Counterexample to the learnability of the Bellman error

To show the full range of possibilities we need a slightly more complex pair of Markovreward processes (MRPs) than those considered earlier. Consider the following twoMRPs:

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/d7487087010c7ed57063d7e2432cf61ffbedd356fc9cb387752b0527536266ce.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/ec3ba6428cfe6a53ffccfcb3cb4e4454063f0606a1f284c589ed67ae588d7dff.jpg)


Where two edges leave a state, both transitions are assumed to occur with equalprobability, and the numbers indicate the reward received. The MRP on the left hastwo states that are represented distinctly. The MRP on the right has three states,two of which, $\textsf { B }$ and $\mathsf { B } ^ { \prime }$ , appear the same and must be given the same approximatevalue. Specifically, w has two components and the value of state A is given by the firstcomponent and the value of $\textsf { B }$ and $\mathsf { B } ^ { \prime }$ is given by the second. The second MRP hasbeen designed so that equal time is spent in all three states, so we can take $\textstyle \mu ( s ) = { \frac { 1 } { 3 } }$ ,for all $s$ .

Note that the observable data distribution is identical for the two MRPs. In bothcases the agent will see single occurrences of $\mathsf { A }$ followed by a 0, then some numberof apparent Bs, each followed by a $^ { - 1 }$ except the last, which is followed by a 1, thenwe start all over again with a single A and a 0, etc. All the statistical details are thesame as well; in both MRPs, the probability of a string of $k$ Bs is $2 ^ { - k }$ .

Now suppose $\mathbf { w } = \mathbf { 0 }$ . In the first MRP, this is an exact solution, and the $\overline { { \mathrm { B E } } }$ iszero. In the second MRP, this solution produces a squared error in both $\textsf { B }$ and $\mathsf { B } ^ { \prime }$ of1, such that $\begin{array} { r } { \overline { { \mathrm { B E } } } = \mu ( \mathsf { B } ) \mathbb { 1 } + \mu ( \mathsf { B } ^ { \prime } ) \mathbb { 1 } = \frac { 2 } { 3 } } \end{array}$ . These two MRPs, which generate the samedata distribution, have di↵erent $\overline { { \mathrm { B E s } } }$ ; the $\overline { { \mathrm { B E } } }$ is not learnable.

Moreover (and unlike the earlier example for the $\overline { { \mathrm { V E } } }$ ) the minimizing value of $\mathbf { w }$is di↵erent for the two MRPs. For the first MRP, $\mathbf { w } = \mathbf { 0 }$ minimizes the $\overline { { \mathrm { B E } } }$ for any$\gamma$ . For the second MRP, the minimizing w is a complicated function of $\gamma$ , but inthe limit, as $\gamma  1$ , it is $( - \frac { 1 } { 2 } , 0 ) ^ { \top }$ . Thus the solution that minimizes $\overline { { \mathrm { B E } } }$ cannot beestimated from data alone; knowledge of the MRP beyond what is revealed in thedata is required. In this sense, it is impossible in principle to pursue the $\overline { { \mathrm { B E } } }$ as anobjective for learning.

It may be surprising that in the second MRP the $\mathrm { B E }$ -minimizing value of $\mathsf { A }$ is so farfrom zero. Recall that A has a dedicated weight and thus its value is unconstrainedby function approximation. A is followed by a reward of 0 and transition to a statewith a value of nearly 0, which suggests $v _ { \mathbf { w } } ( \mathsf { A } )$ should be 0; why is its optimalvalue substantially negative rather than 0? The answer is that making $v _ { \mathbf { w } } ( \mathsf { A } )$ negativereduces the error upon arriving in A from B. The reward on this deterministic transitionis 1, which implies that B should have a value 1 more than A. Because $\textsf { B }$ ’s value isapproximately zero, A’s value is driven toward $^ { - 1 }$ . The $\overline { { \mathrm { B E } } }$ -minimizing value of $\approx - { \frac { 1 } { 2 } }$for $\mathsf { A }$ is a compromise between reducing the errors on leaving and on entering A.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/5918f6c164b9f5d6b1633c93834d9642127e655dfcaa7d737bf4664d9e93f49f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/018d925297e05134507355462eb042e82b9ded2b98331706f944dd3b93ec563c.jpg)



data produced by two di↵erent MDPs is identicdata produced by two di↵erent MDPs1 2Figure 11.4: Causal relationships among the data distribution, MDPs, and various objectives.In such a case the BE is literally not a functioIn such a case the BE is literally notLeft, Monte Carlo objectives: Two di↵erent MDPs can produce the same data distributiondata produced by two data produceP-like examples, in which the observableyet also produce di↵erent VEs, proving that the $\overline { { \mathrm { V E } } }$ erent MDPs is identical in every respecty two di↵erent MDPs is identical in eveestimate it from data. One of the simpobjective cannot be determined from dataIn such a case the BE is literally not a function of the data, andIn such a case the BE is literally not a function of the        and is not learnable. However, all such VEs must have the same optimal parameter vector, w⇤!Moreover, this same $\mathbf { w } ^ { * }$ estimate it from data. One of the simplest eestimate it from data. One of the    can be determined from another objective, the $\overline { { \mathrm { R E } } }$ ples is the pair ofplest examples is t, which is uniquely1 1 f the data, and thus there is no way todetermined from the data distribution. Thus w⇤ and the RE are learnable even though the VEs1 -1 1 1 -1A BA Bes is the pair of MDPs shown below:are not. Right, Bootstrapping objectives: Two di↵erent MDPs can produce the same data-1distribution yet also produce di↵erent BE A -1 these are not learnable from the data d BA 0 B  istribution. The PBE and TDE objectives and their A  0and have di↵erent minimizing parameter vectors;0 0A B B(di↵erent) minima can be directly determined from data and thus are learnable.


probability. The numbers on the edges indicate probability. The numbers on the edges0data and thus cannot be learned from it. The other bootstrapping objectives that weno actions), so they ahave considered, the $\mathrm { P B E }$ e↵ecand $\mathrm { T D E }$ The MDP on the left has two states that are The MDP on the left has two states rkov, can be determined from data (are learnable) andweight so that they can take on any value.       bilities are assumed to occur with equaldetermine optimal solutions that are in general di↵erent from each other and the $\overline { { \mathrm { B E } } }$of which, B and B  , are represente      reward emitted if that edge is traversed.minimums. The general case is summarized in the right side of Figure 11.4.

ented distiThus, the $\overline { { \mathrm { B E } } }$            value. We can imagine that the value of state value. We can imagine that the valuely; each has a separateis not learnable; it cannot be estimated from feature vectors and otherobservable data. This limits the $\overline { { \mathrm { B E } } }$ the value of B and B  is given by the second. the value of B and B  is given by the to model-based settings. There can be no algorithmthat minimizes the $\mathrm { B E }$ for the two MDPs. In both cases the agent wifor the two MDPs. In both cases thewithout access to the underlying MDP states beyond the feature0, then some number   vectors. The residual-gradient algorithm is only able to minimize $\overline { { \mathrm { B E } } }$ s each followed by a     because it is allowed1, then we start all over again with a single A 1, then we start all over again with a        to double sample from the same state—not a state that has the same feature vector,                as well; in both MDPs, the probability of a str            but one that is guaranteed to be the same underlying state. We can see now that there1, then        is no way around this. Minimizing the $\overline { { \mathrm { B E } } }$ tart all over again with a single A and a 0,ction v  =  0. In the first MDP, this is an exfunction v  = 0. In the first MDP, thisrequires some such access to the nominal,as well; in both MDPs, the probas well; in both MDPs,the second Mthe s except the last which is followed by aunderlying MDP. This is an important limitation of the $\mathrm { B E }$ ity of a string of k Bs is 2 .e probability of a string of k  this solution produces an errnd MDP, this solution producbeyond that identified in thefunction v  = 0. In the first MDP, this is an exact solfunction v  = 0. In the first MDP, this is aof pd(B) + d(B  ), or p2/3 if the tof d(B) + d(B  ), or 2/ a 0, etc. All the details are the sameA-presplit example on page 273. All this directs more attention toward the $\overline { { \mathrm { P B E } } }$ n,c .

# 11.7 Gradient-TD Methods

We now consider SGD methods for minimizing the $\overline { { \mathrm { P B E } } }$ . As true SGD methods, theseGradient-TD methods have robust convergence properties even under o↵-policy trainingand nonlinear function approximation. Remember that in the linear case there is alwaysan exact solution, the TD fixed point wTD, at which the $\overline { { \mathrm { P B E } } }$ is zero. This solution couldbe found by least-squares methods (Section 9.8), but only by methods of quadratic $O ( d ^ { 2 } )$complexity in the number of parameters. We seek instead an SGD method, which shouldbe $O ( d )$ and have robust convergence properties. Gradient-TD methods come close toachieving these goals, at the cost of a rough doubling of computational complexity.

To derive an SGD method for the $\overline { { \mathrm { P B E } } }$ (assuming linear function approximation) webegin by expanding and rewriting the objective (11.22) in matrix terms:

$$
\begin{array}{l} \overline {{\mathrm {P B E}}} (\mathbf {w}) = \left\| \Pi \bar {\delta} _ {\mathbf {w}} \right\| _ {\mu} ^ {2} \\ = \left(\Pi \bar {\delta} _ {\mathbf {w}}\right) ^ {\top} \mathbf {D} \Pi \bar {\delta} _ {\mathbf {w}} \quad (\text {f r o m (1 1 . 1 4)}) \\ = \bar {\delta} _ {\mathbf {w}} ^ {\top} \Pi^ {\top} \mathbf {D} \Pi \bar {\delta} _ {\mathbf {w}} \\ = \bar {\delta} _ {\mathbf {w}} ^ {\top} \mathbf {D} \mathbf {X} \left(\mathbf {X} ^ {\top} \mathbf {D} \mathbf {X}\right) ^ {- 1} \mathbf {X} ^ {\top} \mathbf {D} \bar {\delta} _ {\mathbf {w}} \tag {11.25} \\ \end{array}
$$

(using (11.13) and the identity $\Pi ^ { \top } \mathbf { D } \Pi = \mathbf { D } \mathbf { X } \left( \mathbf { X } ^ { \top } \mathbf { D } \mathbf { X } \right) ^ { - 1 } \mathbf { X } ^ { \top } \mathbf { D } )$

$$
= \left(\mathbf {X} ^ {\top} \mathbf {D} \bar {\delta} _ {\mathbf {w}}\right) ^ {\top} \left(\mathbf {X} ^ {\top} \mathbf {D} \mathbf {X}\right) ^ {- 1} \left(\mathbf {X} ^ {\top} \mathbf {D} \bar {\delta} _ {\mathbf {w}}\right). \tag {11.26}
$$

The gradient with respect to w is

$$
\nabla \overline {{\mathrm {P B E}}} (\mathbf {w}) = 2 \nabla \left[ \mathbf {X} ^ {\top} \mathbf {D} \bar {\delta} _ {\mathbf {w}} \right] ^ {\top} \left(\mathbf {X} ^ {\top} \mathbf {D} \mathbf {X}\right) ^ {- 1} \left(\mathbf {X} ^ {\top} \mathbf {D} \bar {\delta} _ {\mathbf {w}}\right).
$$

To turn this into an SGD method, we have to sample something on every time step thathas this quantity as its expected value. Let us take $\mu$ to be the distribution of statesvisited under the behavior policy. All three of the factors above can then be written interms of expectations under this distribution. For example, the last factor can be written

$$
\mathbf {X} ^ {\top} \mathbf {D} \bar {\delta} _ {\mathbf {w}} = \sum_ {s} \mu (s) \mathbf {x} (s) \bar {\delta} _ {\mathbf {w}} (s) = \mathbb {E} [ \rho_ {t} \delta_ {t} \mathbf {x} _ {t} ],
$$

which is just the expectation of the semi-gradient TD(0) update (11.2). The first factoris the transpose of the gradient of this update:

$$
\begin{array}{l} \nabla \mathbb {E} \left[ \rho_ {t} \delta_ {t} \mathbf {x} _ {t} \right] ^ {\top} = \mathbb {E} \left[ \rho_ {t} \nabla \delta_ {t} ^ {\top} \mathbf {x} _ {t} ^ {\top} \right] \\ = \mathbb {E} \left[ \rho_ {t} \nabla \left(R _ {t + 1} + \gamma \mathbf {w} ^ {\top} \mathbf {x} _ {t + 1} - \mathbf {w} ^ {\top} \mathbf {x} _ {t}\right) ^ {\top} \mathbf {x} _ {t} ^ {\top} \right] \quad (\text {u s i n g}) \\ = \mathbb {E} \left[ \rho_ {t} \left(\gamma \mathbf {x} _ {t + 1} - \mathbf {x} _ {t}\right) \mathbf {x} _ {t} ^ {\top} \right]. \\ \end{array}
$$

Finally, the middle factor is the inverse of the expected outer-product matrix of thefeature vectors:

$$
\mathbf {X} ^ {\top} \mathbf {D} \mathbf {X} = \sum_ {s} \mu (s) \mathbf {x} (s) \mathbf {x} (s) ^ {\top} = \mathbb {E} \left[ \mathbf {x} _ {t} \mathbf {x} _ {t} ^ {\top} \right].
$$

Substituting these expectations for the three factors in our expression for the gradient ofthe PBE, we get

$$
\nabla \overline {{\mathrm {P B E}}} (\mathbf {w}) = 2 \mathbb {E} \left[ \rho_ {t} \left(\gamma \mathbf {x} _ {t + 1} - \mathbf {x} _ {t}\right) \mathbf {x} _ {t} ^ {\top} \right] \mathbb {E} \left[ \mathbf {x} _ {t} \mathbf {x} _ {t} ^ {\top} \right] ^ {- 1} \mathbb {E} \left[ \rho_ {t} \delta_ {t} \mathbf {x} _ {t} \right]. \tag {11.27}
$$

It might not be obvious that we have made any progress by writing the gradient in thisform. It is a product of three expressions and the first and last are not independent.They both depend on the next feature vector $\mathbf { x } _ { t + 1 }$ ; we cannot simply sample both ofthese expectations and then multiply the samples. This would give us a biased estimateof the gradient just as in the residual-gradient algorithm.

Another idea would be to estimate the three expectations separately and then combinethem to produce an unbiased estimate of the gradient. This would work, but wouldrequire a lot of computational resources, particularly to store the first two expectations,which are $d \times d$ matrices, and to compute the inverse of the second. This idea can beimproved. If two of the three expectations are estimated and stored, then the third couldbe sampled and used in conjunction with the two stored quantities. For example, youcould store estimates of the second two quantities (using the increment inverse-updatingtechniques in Section 9.8) and then sample the first expression. Unfortunately, the overallalgorithm would still be of quadratic complexity (of order $O ( d ^ { 2 } )$ ).

The idea of storing some estimates separately and then combining them with samplesis a good one and is also used in Gradient-TD methods. Gradient-TD methods estimateand store the product of the second two factors in (11.27). These factors are a $d \times d$ matrix and a $d$ -vector, so their product is just a $d$ -vector, like w itself. We denote thissecond learned vector as v:

$$
\mathbf {v} \approx \mathbb {E} \left[ \mathbf {x} _ {t} \mathbf {x} _ {t} ^ {\top} \right] ^ {- 1} \mathbb {E} \left[ \rho_ {t} \delta_ {t} \mathbf {x} _ {t} \right]. \tag {11.28}
$$

This form is familiar to students of linear supervised learning. It is the solution to a linearleast-squares problem that tries to approximate $\rho _ { t } \delta _ { t }$ from the features. The standardSGD method for incrementally finding the vector $\mathbf { v }$ that minimizes the expected squarederror $\left( \mathbf { v } ^ { \top } \mathbf { x } _ { t } - \rho _ { t } \delta _ { t } \right) ^ { 2 }$ is known as the Least Mean Square (LMS) rule (here augmentedwith an importance sampling ratio):

$$
\mathbf {v} _ {t + 1} \doteq \mathbf {v} _ {t} + \beta \rho_ {t} \left(\delta_ {t} - \mathbf {v} _ {t} ^ {\top} \mathbf {x} _ {t}\right) \mathbf {x} _ {t},
$$

where $\beta > 0$ is another step-size parameter. We can use this method to e↵ectively achieve(11.28) with $O ( d )$ storage and per-step computation.

Given a stored estimate $\mathbf { v } _ { t }$ approximating (11.28), we can update our main parametervector $\mathbf { w } _ { t }$ using SGD methods based on (11.27). The simplest such rule is

$$
\begin{array}{l} \mathbf {w} _ {t + 1} = \mathbf {w} _ {t} - \frac {1}{2} \alpha \nabla \overline {{\mathrm {P B E}}} (\mathbf {w} _ {t}) \quad \text {(t h e g e n e r a l S G D r u l e)} \\ = \mathbf {w} _ {t} - \frac {1}{2} \alpha 2 \mathbb {E} \left[ \rho_ {t} \left(\gamma \mathbf {x} _ {t + 1} - \mathbf {x} _ {t}\right) \mathbf {x} _ {t} ^ {\top} \right] \mathbb {E} \left[ \mathbf {x} _ {t} \mathbf {x} _ {t} ^ {\top} \right] ^ {- 1} \mathbb {E} \left[ \rho_ {t} \delta_ {t} \mathbf {x} _ {t} \right] \quad (\text {f r o m (1 1 . 2 7)}) \\ = \mathbf {w} _ {t} + \alpha \mathbb {E} \left[ \rho_ {t} \left(\mathbf {x} _ {t} - \gamma \mathbf {x} _ {t + 1}\right) \mathbf {x} _ {t} ^ {\top} \right] \mathbb {E} \left[ \mathbf {x} _ {t} \mathbf {x} _ {t} ^ {\top} \right] ^ {- 1} \mathbb {E} \left[ \rho_ {t} \delta_ {t} \mathbf {x} _ {t} \right] (11.29) \\ \approx \mathbf {w} _ {t} + \alpha \mathbb {E} \left[ \rho_ {t} \left(\mathbf {x} _ {t} - \gamma \mathbf {x} _ {t + 1}\right) \mathbf {x} _ {t} ^ {\top} \right] \mathbf {v} _ {t} \quad (\text {b a s e d o n (1 1 . 2 8)}) \\ \approx \mathbf {w} _ {t} + \alpha \rho_ {t} \left(\mathbf {x} _ {t} - \gamma \mathbf {x} _ {t + 1}\right) \mathbf {x} _ {t} ^ {\top} \mathbf {v} _ {t}. (sampling) \\ \end{array}
$$

This algorithm is called $G T D \mathcal { Q }$ . Note that if the final inner product $\left( \mathbf { x } _ { t } ^ { \mid } \mathbf { v } _ { t } \right)$ is done first,then the entire algorithm is of $O ( d )$ complexity.

A slightly better algorithm can be derived by doing a few more analytic steps beforesubstituting in $\mathbf { v } _ { t }$ . Continuing from (11.29):

$$
\begin{array}{l} \mathbf {w} _ {t + 1} = \mathbf {w} _ {t} + \alpha \mathbb {E} \big [ \rho_ {t} (\mathbf {x} _ {t} - \gamma \mathbf {x} _ {t + 1}) \mathbf {x} _ {t} ^ {\top} \big ] \mathbb {E} \big [ \mathbf {x} _ {t} \mathbf {x} _ {t} ^ {\top} \big ] ^ {- 1} \mathbb {E} \big [ \rho_ {t} \delta_ {t} \mathbf {x} _ {t} \big ] \\ = \mathbf {w} _ {t} + \alpha \left(\mathbb {E} \left[ \rho_ {t} \mathbf {x} _ {t} \mathbf {x} _ {t} ^ {\top} \right] - \gamma \mathbb {E} \left[ \rho_ {t} \mathbf {x} _ {t + 1} \mathbf {x} _ {t} ^ {\top} \right]\right) \mathbb {E} \left[ \mathbf {x} _ {t} \mathbf {x} _ {t} ^ {\top} \right] ^ {- 1} \mathbb {E} [ \rho_ {t} \delta_ {t} \mathbf {x} _ {t} ] \\ = \mathbf {w} _ {t} + \alpha \left(\mathbb {E} \left[ \mathbf {x} _ {t} \mathbf {x} _ {t} ^ {\top} \right] - \gamma \mathbb {E} \left[ \rho_ {t} \mathbf {x} _ {t + 1} \mathbf {x} _ {t} ^ {\top} \right]\right) \mathbb {E} \left[ \mathbf {x} _ {t} \mathbf {x} _ {t} ^ {\top} \right] ^ {- 1} \mathbb {E} [ \rho_ {t} \delta_ {t} \mathbf {x} _ {t} ] \\ = \mathbf {w} _ {t} + \alpha \left(\mathbb {E} \left[ \rho_ {t} \delta_ {t} \mathbf {x} _ {t} \right] - \gamma \mathbb {E} \left[ \rho_ {t} \mathbf {x} _ {t + 1} \mathbf {x} _ {t} ^ {\top} \right] \mathbb {E} \left[ \mathbf {x} _ {t} \mathbf {x} _ {t} ^ {\top} \right] ^ {- 1} \mathbb {E} \left[ \rho_ {t} \delta_ {t} \mathbf {x} _ {t} \right]\right) \\ \approx \mathbf {w} _ {t} + \alpha \left(\mathbb {E} \left[ \rho_ {t} \delta_ {t} \mathbf {x} _ {t} \right] - \gamma \mathbb {E} \left[ \rho_ {t} \mathbf {x} _ {t + 1} \mathbf {x} _ {t} ^ {\top} \right] \mathbf {v} _ {t}\right) \quad (\text {b a s e d o n (1 1 . 2 8)}) \\ \approx \mathbf {w} _ {t} + \alpha \rho_ {t} \left(\delta_ {t} \mathbf {x} _ {t} - \gamma \mathbf {x} _ {t + 1} \mathbf {x} _ {t} ^ {\top} \mathbf {v} _ {t}\right), \quad (\text {s a m p l i n g}) \\ \end{array}
$$

which again is $O ( d )$ if the final product $( \mathbf { x } _ { t } ^ { \top } \mathbf { v } _ { t } )$ is done first. This algorithm is known aseither TD(0) with gradient correction (TDC) or, alternatively, as GTD(0).

Figure 11.5 shows a sample and the expected behavior of TDC on Baird’s counterex-ample. As intended, the $\overline { { \mathrm { P B E } } }$ falls to zero, but note that the individual componentsof the parameter vector do not approach zero. In fact, these values are still far from

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/c39b6dc7ea024d2bd36a236725b6d6ad4a9e476d9e6e545a111c4987ff4c4b44.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/36f9bade93c5dc5ca10f8c969a4c5fa7488d8449a287357ce10f7cb2aca8ed24.jpg)



Figure 11.5: The behavior of the TDC algorithm on Baird’s counterexample. On the left isshown a typical single run, and on the right is shown the expected behavior of this algorithm ifthe updates are done synchronously (analogous to (11.9), except for the two TDC parametervectors). The step sizes were $\alpha = 0 . 0 0 5$ and $\beta = 0 . 0 5$ .


an optimal solution, $\hat { v } ( s ) = 0$ , for all $s$ , for which w would have to be proportional to$( 1 , 1 , 1 , 1 , 1 , 1 , 4 , - 2 ) ^ { \top }$ . After 1000 iterations we are still far from an optimal solution, aswe can see from the $\mathrm { V E }$ , which remains almost 2. The system is actually converging toan optimal solution, but progress is extremely slow because the $\mathrm { P B E }$ is already so closeto zero.

GTD2 and TDC both involve two learning processes, a primary one for w and asecondary one for $\mathbf { v }$ . The logic of the primary learning process relies on the secondarylearning process having finished, at least approximately, whereas the secondary learningprocess proceeds without being influenced by the first. We call this sort of asymmetricaldependence a cascade. In cascades we often assume that the secondary learning processis proceeding faster and thus is always at its asymptotic value, ready and accurate toassist the primary learning process. The convergence proofs for these methods often makethis assumption explicitly. These are called two-time-scale proofs. The fast time scale isthat of the secondary learning process, and the slower time scale is that of the primarylearning process. If $\alpha$ is the step size of the primary learning process, and $\beta$ is the stepsize of the secondary learning process, then these convergence proofs will typically requirethat in the limit $\beta  0$ and $\begin{array} { r } { \frac { \alpha } { \beta } \to 0 } \end{array}$ .

Gradient-TD methods are currently the most well understood and widely used stableo↵-policy methods. There are extensions to action values and control (GQ, Maei et al.,2010), to eligibility traces (GTD( ) and GQ( $\lambda$ ), Maei, 2011; Maei and Sutton, 2010), andto nonlinear function approximation (Maei et al., 2009). There have also been proposedhybrid algorithms midway between semi-gradient TD and gradient TD (Hackman, 2012;White and White, 2016). Hybrid-TD algorithms behave like Gradient-TD algorithms instates where the target and behavior policies are very di↵erent, and behave like semi-gradient algorithms in states where the target and behavior policies are the same. Finally,the Gradient-TD idea has been combined with the ideas of proximal methods and controlvariates to produce more e cient methods (Mahadevan et al., 2014; Du et al., 2017).

# 11.8 Emphatic-TD Methods

We turn now to the second major strategy that has been extensively explored for obtaininga cheap and e cient o↵-policy learning method with function approximation. Recallthat linear semi-gradient TD methods are e cient and stable when trained under theon-policy distribution, and that we showed in Section 9.4 that this has to do with thepositive definiteness of the matrix A (9.11)4 and the match between the on-policy statedistribution $\mu _ { \pi }$ and the state-transition probabilities $p ( s | s , a )$ under the target policy. Ino↵-policy learning, we reweight the state transitions using importance sampling so thatthey become appropriate for learning about the target policy, but the state distributionis still that of the behavior policy. There is a mismatch. A natural idea is to somehowreweight the states, emphasizing some and de-emphasizing others, so as to return thedistribution of updates to the on-policy distribution. There would then be a match,and stability and convergence would follow from existing results. This is the idea of

Emphatic-TD methods, first introduced for on-policy training in Section 9.11.

Actually, the notion of “the on-policy distribution” is not quite right, as there are manyon-policy distributions, and any one of these is su cient to guarantee stability. Consideran undiscounted episodic problem. The way episodes terminate is fully determined by thetransition probabilities, but there may be several di↵erent ways the episodes might begin.However the episodes start, if all state transitions are due to the target policy, then thestate distribution that results is an on-policy distribution. You might start close to theterminal state and visit only a few states with high probability before ending the episode.Or you might start far away and pass through many states before terminating. Both areon-policy distributions, and training on both with a linear semi-gradient method wouldbe guaranteed to be stable. However the process starts, an on-policy distribution resultsas long as all states encountered are updated up until termination.

If there is discounting, it can be treated as partial or probabilistic termination for thesepurposes. If $\gamma = 0 . 9$ , then we can consider that with probability 0.1 the process terminateson every time step and then immediately restarts in the state that is transitioned to. Adiscounted problem is one that is continually terminating and restarting with probability$1 - \gamma$ on every step. This way of thinking about discounting is an example of a moregeneral notion of pseudo termination—termination that does not a↵ect the sequence ofstate transitions, but does a↵ect the learning process and the quantities being learned.This kind of pseudo termination is important to o↵-policy learning because the restartingis optional—remember we can start any way we want to—and the termination relievesthe need to keep including encountered states within the on-policy distribution. That is,if we don’t consider the new states as restarts, then discounting quickly gives us a limitedon-policy distribution.

The one-step Emphatic-TD algorithm for learning episodic state values is defined by:

$$
\begin{array}{l} \delta_ {t} = R _ {t + 1} + \gamma \hat {v} \left(S _ {t + 1}, \mathbf {w} _ {t}\right) - \hat {v} \left(S _ {t}, \mathbf {w} _ {t}\right), \\ \mathbf {w} _ {t + 1} = \mathbf {w} _ {t} + \alpha M _ {t} \rho_ {t} \delta_ {t} \nabla \hat {v} (S _ {t}, \mathbf {w} _ {t}), \\ M _ {t} = \gamma \rho_ {t - 1} M _ {t - 1} + I _ {t}, \\ \end{array}
$$

with $I _ { t }$ , the interest, being arbitrary and $M _ { t }$ , the emphasis, being initialized to $M _ { - 1 } = 0$ .How does this algorithm perform on Baird’s counterexample? Figure 11.6 shows thetrajectory in expectation of the components of the parameter vector (for the case inwhich $I _ { t } = 1$ , for all $t$ ). There are some oscillations but eventually everything convergesand the $\mathrm { V E }$ goes to zero. These trajectories are obtained by iteratively computing theexpectation of the parameter vector trajectory without any of the variance due to samplingof transitions and rewards. We do not show the results of applying the Emphatic-TDalgorithm directly because its variance on Baird’s counterexample is so high that it isnigh impossible to get consistent results in computational experiments. The algorithmconverges to the optimal solution in theory on this problem, but in practice it doesnot. We turn to the topic of reducing the variance of all these algorithms in the nextsection.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/cd05ddba7ca8120ae12f7d6cf868b03fd3bfe449c9174a3bea7e137db4ea04f3.jpg)



Figure 11.6: The behavior of the one-step Emphatic-TD algorithm in expectation on Baird’scounterexample. The step size was $\alpha = 0 . 0 3$ .


# 11.9 Reducing Variance

O↵-policy learning is inherently of greater variance than on-policy learning. This is notsurprising; if you receive data less closely related to a policy, you should expect to learnless about the policy’s values. In the extreme, one may be able to learn nothing. Youcan’t expect to learn how to drive by cooking dinner, for example. Only if the target andbehavior policies are related, if they visit similar states and take similar actions, shouldone be able to make significant progress in o↵-policy training.

On the other hand, any policy has many neighbors, many similar policies with con-siderable overlap in states visited and actions chosen, and yet which are not identical.The raison d’ˆetre of o↵-policy learning is to enable generalization to this vast numberof related-but-not-identical policies. The problem remains of how to make the best useof the experience. Now that we have some methods that are stable in expected value(if the step sizes are set right), attention naturally turns to reducing the variance of theestimates. There are many possible ideas, and we can just touch on a few of them in thisintroductory text.

Why is controlling variance especially critical in o↵-policy methods based on importancesampling? As we have seen, importance sampling often involves products of policy ratios.The ratios are always one in expectation (5.13), but their actual values may be very highor as low as zero. Successive ratios are uncorrelated, so their products are also always onein expected value, but they can be of very high variance. Recall that these ratios multiplythe step size in SGD methods, so high variance means taking steps that vary greatly intheir sizes. This is problematic for SGD because of the occasional very large steps. Theymust not be so large as to take the parameter to a part of the space with a very di↵erentgradient. SGD methods rely on averaging over multiple steps to get a good sense ofthe gradient, and if they make large moves from single samples they become unreliable.If the step-size parameter is set small enough to prevent this, then the expected step

can end up being very small, resulting in very slow learning. The notions of momentum(Derthick, 1984), of Polyak-Ruppert averaging (Polyak, 1990; Ruppert, 1988; Polyak andJuditsky, 1992), or further extensions of these ideas may significantly help. Methods foradaptively setting separate step sizes for di↵erent components of the parameter vectorare also pertinent (e.g., Jacobs, 1988; Sutton, 1992b, c), as are the “importance weightaware” updates of Karampatziakis and Langford (2010).

In Chapter 5 we saw how weighted importance sampling is significantly better behaved,with lower variance updates, than ordinary importance sampling. However, adaptingweighted importance sampling to function approximation is challenging and can probablyonly be done approximately with $O ( d )$ complexity (Mahmood and Sutton, 2015).

The Tree Backup algorithm (Section 7.5) shows that it is possible to perform someo↵-policy learning without using importance sampling. This idea has been extended tothe o↵-policy case to produce stable and more e cient methods by Munos, Stepleton,Harutyunyan, and Bellemare (2016) and by Mahmood, Yu and Sutton (2017).

Another, complementary strategy is to allow the target policy to be determined inpart by the behavior policy, in such a way that it never can be so di↵erent from it tocreate large importance sampling ratios. For example, the target policy can be defined byreference to the behavior policy, as in the “recognizers” proposed by Precup et al. (2006).

# 11.10 Summary

O↵-policy learning is a tempting challenge, testing our ingenuity in designing stable ande cient learning algorithms. Tabular Q-learning makes o↵-policy learning seem easy,and it has natural generalizations to Expected Sarsa and to the Tree Backup algorithm.But as we have seen in this chapter, the extension of these ideas to significant functionapproximation, even linear function approximation, involves new challenges and forces usto deepen our understanding of reinforcement learning algorithms.

Why go to such lengths? One reason to seek o↵-policy algorithms is to give flexibilityin dealing with the tradeo↵ between exploration and exploitation. Another is to freebehavior from learning, and avoid the tyranny of the target policy. TD learning appearsto hold out the possibility of learning about multiple things in parallel, of using onestream of experience to solve many tasks simultaneously. We can certainly do this inspecial cases, just not in every case that we would like to or as e ciently as we wouldlike to.

In this chapter we divided the challenge of o↵-policy learning into two parts. Thefirst part, correcting the targets of learning for the behavior policy, is straightforwardlydealt with using the techniques devised earlier for the tabular case, albeit at the cost ofincreasing the variance of the updates and thereby slowing learning. High variance willprobably always remains a challenge for o↵-policy learning.

The second part of the challenge of o↵-policy learning emerges as the instabilityof semi-gradient TD methods that involve bootstrapping. We seek powerful functionapproximation, o↵-policy learning, and the e ciency and flexibility of bootstrapping

TD methods, but it is challenging to combine all three aspects of this deadly triad inone algorithm without introducing the potential for instability. There have been severalattempts. The most popular has been to seek to perform true stochastic gradient descent(SGD) in the Bellman error (a.k.a. the Bellman residual). However, our analysis concludesthat this is not an appealing goal in many cases, and that anyway it is impossible toachieve with a learning algorithm—the gradient of the $\overline { { \mathrm { B E } } }$ is not learnable from experiencethat reveals only feature vectors and not underlying states. Another approach, Gradient-TD methods, performs SGD in the projected Bellman error. The gradient of the $\mathrm { P B E }$is learnable with $O ( d )$ complexity, but at the cost of a second parameter vector with asecond step size. The newest family of methods, Emphatic-TD methods, refine an old ideafor reweighting updates, emphasizing some and de-emphasizing others. In this way theyrestore the special properties that make on-policy learning stable with computationallysimple semi-gradient methods.

The whole area of o↵-policy learning is relatively new and unsettled. Which methodsare best or even adequate is not yet clear. Are the complexities of the new methodsintroduced at the end of this chapter really necessary? Which of them can be combinede↵ectively with variance reduction methods? The potential for o↵-policy learning remainstantalizing, the best way to achieve it still a mystery.

# Bibliographical and Historical Remarks

11.1 The first semi-gradient method was linear TD( ) (Sutton, 1988). The name“semi-gradient” is more recent (Sutton, 2015a). Semi-gradient o↵-policy TD(0)with general importance-sampling ratio may not have been explicitly stated untilSutton, Mahmood, and White (2016), but the action-value forms were introducedby Precup, Sutton, and Singh (2000), who also did eligibility trace forms of thesealgorithms (see Chapter 12). Their continuing, undiscounted forms have notbeen significantly explored. The $\boldsymbol { n }$ -step forms given here are new.

11.2 The earliest $w$ -to- $2 w$ example was given by Tsitsiklis and Van Roy (1996), whoalso introduced the specific counterexample in the box on page 263. Baird’scounterexample is due to Baird (1995), though the version we present here isslightly modified. Averaging methods for function approximation were developedby Gordon (1995, 1996b). Other examples of instability with o↵-policy DPmethods and more complex methods of function approximation are given byBoyan and Moore (1995). Bradtke (1993) gives an example in which Q-learningusing linear function approximation in a linear quadratic regulation problemconverges to a destabilizing policy.

11.3 The deadly triad was first identified by Sutton (1995b) and thoroughly analyzedby Tsitsiklis and Van Roy (1997). The name “deadly triad” is due to Sutton(2015a).

11.4 This kind of linear analysis was pioneered by Tsitsiklis and Van Roy (1996; 1997),including the dynamic programming operator. Diagrams like Figure 11.3 were

introduced by Lagoudakis and Parr (2003).

What we have called the Bellman operator, and denoted $B _ { \pi }$ , is more commonlydenoted $T ^ { \pi }$ and called a “dynamic programming operator,” while a generalizedform, denoted $T ^ { ( \lambda ) }$ , is called the “TD( $\lambda$ ) operator” (Tsitsiklis and Van Roy, 1996,1997).

11.5 The $\mathrm { B E }$ was first proposed as an objective function for dynamic programming bySchweitzer and Seidmann (1985). Baird (1995, 1999) extended it to TD learningbased on stochastic gradient descent. In the literature, $\mathrm { B E }$ minimization is oftenreferred to as Bellman residual minimization.

The earliest A-split example is due to Dayan (1992). The two forms given herewere introduced by Sutton et al. (2009a).

11.6 The contents of this section are new to this text.

11.7 Gradient-TD methods were introduced by Sutton, Szepesv´ari, and Maei (2009b).The methods highlighted in this section were introduced by Sutton et al. (2009a)and Mahmood et al. (2014). A major extension to proximal TD methodswas developed by Mahadevan et al. (2014). The most sensitive empiricalinvestigations to date of Gradient-TD and related methods are given by Geistand Scherrer (2014), Dann, Neumann, and Peters (2014), White (2015), andGhiassian, Patterson, White, Sutton, and White (2018). Recent developments inthe theory of Gradient-TD methods are presented by Yu (2017).

11.8 Emphatic-TD methods were introduced by Sutton, Mahmood, and White (2016).Full convergence proofs and other theory were later established by Yu (2015;2016; Yu, Mahmood, and Sutton, 2017), Hallak, Tamar, and Mannor (2015), andHallak, Tamar, Munos, and Mannor (2016).

