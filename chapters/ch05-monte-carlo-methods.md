# Chapter 5

# Monte Carlo Methods

In this chapter we consider our first learning methods for estimating value functions anddiscovering optimal policies. Unlike the previous chapter, here we do not assume completeknowledge of the environment. Monte Carlo methods require only experience—samplesequences of states, actions, and rewards from actual or simulated interaction with anenvironment. Learning from actual experience is striking because it requires no priorknowledge of the environment’s dynamics, yet can still attain optimal behavior. Learningfrom simulated experience is also powerful. Although a model is required, the model needonly generate sample transitions, not the complete probability distributions of all possibletransitions that is required for dynamic programming (DP). In surprisingly many cases itis easy to generate experience sampled according to the desired probability distributions,but infeasible to obtain the distributions in explicit form.

Monte Carlo methods are ways of solving the reinforcement learning problem based onaveraging sample returns. To ensure that well-defined returns are available, here we defineMonte Carlo methods only for episodic tasks. That is, we assume experience is dividedinto episodes, and that all episodes eventually terminate no matter what actions areselected. Only on the completion of an episode are value estimates and policies changed.Monte Carlo methods can thus be incremental in an episode-by-episode sense, but not ina step-by-step (online) sense. The term “Monte Carlo” is often used more broadly forany estimation method whose operation involves a significant random component. Herewe use it specifically for methods based on averaging complete returns (as opposed tomethods that learn from partial returns, considered in the next chapter).

Monte Carlo methods sample and average returns for each state–action pair much likethe bandit methods we explored in Chapter 2 sample and average rewards for each action.The main di↵erence is that now there are multiple states, each acting like a di↵erentbandit problem (like an associative-search or contextual bandit) and the di↵erent banditproblems are interrelated. That is, the return after taking an action in one state dependson the actions taken in later states in the same episode. Because all the action selectionsare undergoing learning, the problem becomes nonstationary from the point of view ofthe earlier state.

To handle the nonstationarity, we adapt the idea of general policy iteration (GPI)developed in Chapter 4 for DP. Whereas there we computed value functions from knowledgeof the MDP, here we learn value functions from sample returns with the MDP. The valuefunctions and corresponding policies still interact to attain optimality in essentially thesame way (GPI). As in the DP chapter, first we consider the prediction problem (thecomputation of $v _ { \pi }$ and $q _ { \pi }$ for a fixed arbitrary policy $\pi$ ) then policy improvement, and,finally, the control problem and its solution by GPI. Each of these ideas taken from DPis extended to the Monte Carlo case in which only sample experience is available.

# 5.1 Monte Carlo Prediction

We begin by considering Monte Carlo methods for learning the state-value function for agiven policy. Recall that the value of a state is the expected return—expected cumulativefuture discounted reward—starting from that state. An obvious way to estimate it fromexperience, then, is simply to average the returns observed after visits to that state. Asmore returns are observed, the average should converge to the expected value. This ideaunderlies all Monte Carlo methods.

In particular, suppose we wish to estimate $v _ { \pi } ( s )$ , the value of a state $s$ under policy $\pi$ ,given a set of episodes obtained by following $\pi$ and passing through $s$ . Each occurrenceof state $s$ in an episode is called a visit to $s$ . Of course, $s$ may be visited multiple timesin the same episode; let us call the first time it is visited in an episode the first visitto $s$ . The first-visit MC method estimates $v _ { \pi } ( s )$ as the average of the returns followingfirst visits to $s$ , whereas the every-visit MC method averages the returns following allvisits to $s$ . These two Monte Carlo (MC) methods are very similar but have slightlydi↵erent theoretical properties. First-visit MC has been most widely studied, dating backto the 1940s, and is the one we focus on in this chapter. Every-visit MC extends morenaturally to function approximation and eligibility traces, as discussed in Chapters 9 and12. First-visit MC is shown in procedural form in the box. Every-visit MC would be thesame except without the check for $S _ { t }$ having occurred earlier in the episode.

First-visit MC prediction, for estimating  $V\approx v_{\pi}$    
Input: a policy  $\pi$  to be evaluated. Initialize:  $V(s)\in \mathbb{R}$  , arbitrarily, for all  $s\in S$  Returns(s)  $\leftarrow$  an empty list, for all  $s\in S$    
Loop forever (for each episode): Generate an episode following  $\pi$  ..  $S_0,A_0,R_1,S_1,A_1,R_2,\ldots ,S_{T - 1},A_{T - 1},RT$ $G\gets 0$  Loop for each step of episode,  $t = T - 1,T - 2,\dots ,0$  .  $G\gets \gamma G + R_{t + 1}$  Unless  $S_{t}$  appears in  $S_0,S_1,\ldots ,S_{t - 1}$  : Append  $G$  to Returns  $(S_{t})$ $V(S_{t})\gets$  average(Returns  $(S_{t}))$

Both first-visit MC and every-visit MC converge to $v _ { \pi } ( s )$ as the number of visits (orfirst visits) to $s$ goes to infinity. This is easy to see for the case of first-visit MC. Inthis case each return is an independent, identically distributed estimate of $v _ { \pi } ( s )$ withfinite variance. By the law of large numbers the sequence of averages of these estimatesconverges to their expected value. Each average is itself an unbiased estimate, and thestandard deviation of its error falls as $1 / \sqrt { n }$ , where $n$ is the number of returns averaged.Every-visit MC is less straightforward, but its estimates also converge quadratically to$v _ { \pi } ( s )$ (Singh and Sutton, 1996).

The use of Monte Carlo methods is best illustrated through an example.

Example 5.1: Blackjack The object of the popular casino card game of blackjack is toobtain cards the sum of whose numerical values is as great as possible without exceeding21. All face cards count as 10, and an ace can count either as 1 or as 11. We considerthe version in which each player competes independently against the dealer. The gamebegins with two cards dealt to both dealer and player. One of the dealer’s cards is faceup and the other is face down. If the player has 21 immediately (an ace and a 10-card),it is called a natural. He then wins unless the dealer also has a natural, in which case thegame is a draw. If the player does not have a natural, then he can request additionalcards, one by one (hits), until he either stops (sticks) or exceeds 21 (goes bust). If he goesbust, he loses; if he sticks, then it becomes the dealer’s turn. The dealer hits or sticksaccording to a fixed strategy without choice: he sticks on any sum of 17 or greater, andhits otherwise. If the dealer goes bust, then the player wins; otherwise, the outcome—win,lose, or draw—is determined by whose final sum is closer to 21.

Playing blackjack is naturally formulated as an episodic finite MDP. Each game ofblackjack is an episode. Rewards of $+ 1$ , $^ { - 1 }$ , and 0 are given for winning, losing, anddrawing, respectively. All rewards within a game are zero, and we do not discount $\gamma = 1$ );therefore these terminal rewards are also the returns. The player’s actions are to hit orto stick. The states depend on the player’s cards and the dealer’s showing card. Weassume that cards are dealt from an infinite deck (i.e., with replacement) so that there isno advantage to keeping track of the cards already dealt. If the player holds an ace thathe could count as 11 without going bust, then the ace is said to be usable. In this caseit is always counted as 11 because counting it as 1 would make the sum 11 or less, inwhich case there is no decision to be made because, obviously, the player should alwayshit. Thus, the player makes decisions on the basis of three variables: his current sum(12–21), the dealer’s one showing card (ace–10), and whether or not he holds a usableace. This makes for a total of 200 states.

Consider the policy that sticks if the player’s sum is 20 or 21, and otherwise hits. Tofind the state-value function for this policy by a Monte Carlo approach, one simulatesmany blackjack games using the policy and averages the returns following each state.In this way, we obtained the estimates of the state-value function shown in Figure 5.1.The estimates for states with a usable ace are less certain and less regular because thesestates are less common. In any event, after 500,000 games the value function is very wellapproximated.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/d244835adf18b8e5873fe93d2dc69ada724f3050258fb85e38f8c26a01128cb7.jpg)



Figure 5.1: Approximate state-value functions for the blackjack policy that sticks only on 20or 21, computed by Monte Carlo policy evaluation.


Exercise 5.1 Consider the diagrams on the right in Figure 5.1. Why does the estimatedvalue function jump up for the last two rows in the rear? Why does it drop o↵ for thewhole last row on the left? Why are the frontmost values higher in the upper diagramsthan in the lower? ⇤

Exercise 5.2 Suppose every-visit MC was used instead of first-visit MC on the blackjacktask. Would you expect the results to be very di↵erent? Why or why not? ⇤

Although we have complete knowledge of the environment in the blackjack task, itwould not be easy to apply DP methods to compute the value function. DP methodsrequire the distribution of next events—in particular, they require the environmentsdynamics as given by the four-argument function $p$ —and it is not easy to determinethis for blackjack. For example, suppose the player’s sum is 14 and he chooses to stick.What is his probability of terminating with a reward of $+ 1$ as a function of the dealer’sshowing card? All of the probabilities must be computed before DP can be applied, andsuch computations are often complex and error-prone. In contrast, generating the samplegames required by Monte Carlo methods is easy. This is the case surprisingly often; theability of Monte Carlo methods to work with sample episodes alone can be a significantadvantage even when one has complete knowledge of the environment’s dynamics.

Can we generalize the idea of backup diagrams to Monte Carlo algorithms? Thegeneral idea of a backup diagram is to show at the top the root node to be updated andto show below all the transitions and leaf nodes whose rewards and estimated valuescontribute to the update. For Monte Carlo estimation of $v _ { \pi }$ , the root is a state node, andbelow it is the entire trajectory of transitions along a particular single episode, ending

at the terminal state, as shown to the right. Whereas the DP diagram (page 59)shows all possible transitions, the Monte Carlo diagram shows only those sampledon the one episode. Whereas the DP diagram includes only one-step transitions,the Monte Carlo diagram goes all the way to the end of the episode. Thesedi↵erences in the diagrams accurately reflect the fundamental di↵erences betweenthe algorithms.

An important fact about Monte Carlo methods is that the estimates for eachstate are independent. The estimate for one state does not build upon the estimateof any other state, as is the case in DP. In other words, Monte Carlo methods donot bootstrap as we defined it in the previous chapter.

In particular, note that the computational expense of estimating the value ofa single state is independent of the number of states. This can make Monte Carlomethods particularly attractive when one requires the value of only one or a subset

of states. One can generate many sample episodes starting from the states of interest,averaging returns from only these states, ignoring all others. This is a third advantageMonte Carlo methods can have over DP methods (after the ability to learn from actualexperience and from simulated experience).

Example 5.2: Soap Bubble Suppose a wireframe forming a closed loop is dunked in soapywater to form a soap surface or bubble conform-ing at its edges to the wire frame. If the geom-etry of the wire frame is irregular but known,how can you compute the shape of the surface?The shape has the property that the total forceon each point exerted by neighboring points iszero (or else the shape would change). Thismeans that the surface’s height at any point isthe average of its heights at points in a smallcircle around that point. In addition, the sur-face must meet at its boundaries with the wireframe. The usual approach to problems of thiskind is to put a grid over the area covered by

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/37118ffef5e728285a51c3d8712a30fca8404fa3f921fc03836108dd18aa1c71.jpg)



A bubble on a wire loop.



From Hersh and Griego (1969). Reproduced withpermission. $©$ 1969 Scientific American, a divi-sion of Nature America, Inc. All rights reserved.


the surface and solve for its height at the grid points by an iterative computation. Gridpoints at the boundary are forced to the wire frame, and all others are adjusted towardthe average of the heights of their four nearest neighbors. This process then iterates, muchlike DP’s iterative policy evaluation, and ultimately converges to a close approximationto the desired surface.

This is similar to the kind of problem for which Monte Carlo methods were originallydesigned. Instead of the iterative computation described above, imagine standing on thesurface and taking a random walk, stepping randomly from grid point to neighboringgrid point, with equal probability, until you reach the boundary. It turns out that theexpected value of the height at the boundary is a close approximation to the height ofthe desired surface at the starting point (in fact, it is exactly the value computed by theiterative method described above). Thus, one can closely approximate the height of the

surface at a point by simply averaging the boundary heights of many walks started atthe point. If one is interested in only the value at one point, or any fixed small set ofpoints, then this Monte Carlo method can be far more e cient than the iterative methodbased on local consistency. 

# 5.2 Monte Carlo Estimation of Action Values

If a model is not available, then it is particularly useful to estimate action values (thevalues of state–action pairs) rather than state values. With a model, state values alone aresu cient to determine a policy; one simply looks ahead one step and chooses whicheveraction leads to the best combination of reward and next state, as we did in the chapter onDP. Without a model, however, state values alone are not su cient. One must explicitlyestimate the value of each action in order for the values to be useful in suggesting a policy.Thus, one of our primary goals for Monte Carlo methods is to estimate $q _ { * }$ . To achievethis, we first consider the policy evaluation problem for action values.

The policy evaluation problem for action values is to estimate $q _ { \pi } ( s , a )$ , the expectedreturn when starting in state $s$ , taking action $a$ , and thereafter following policy $\pi$ . TheMonte Carlo methods for this are essentially the same as just presented for state values,except now we talk about visits to a state–action pair rather than to a state. A state–action pair $s , a$ is said to be visited in an episode if ever the state $s$ is visited and action$a$ is taken in it. The every-visit MC method estimates the value of a state–action pairas the average of the returns that have followed all the visits to it. The first-visit MCmethod averages the returns following the first time in each episode that the state wasvisited and the action was selected. These methods converge quadratically, as before, tothe true expected values as the number of visits to each state–action pair approachesinfinity.

The only complication is that many state–action pairs may never be visited. If $\pi$ isa deterministic policy, then in following $\pi$ one will observe returns only for one of theactions from each state. With no returns to average, the Monte Carlo estimates of theother actions will not improve with experience. This is a serious problem because thepurpose of learning action values is to help in choosing among the actions available ineach state. To compare alternatives we need to estimate the value of all the actions fromeach state, not just the one we currently favor.

This is the general problem of maintaining exploration, as discussed in the contextof the $k$ -armed bandit problem in Chapter 2. For policy evaluation to work for actionvalues, we must assure continual exploration. One way to do this is by specifying thatthe episodes start in a state–action pair, and that every pair has a nonzero probability ofbeing selected as the start. This guarantees that all state–action pairs will be visited aninfinite number of times in the limit of an infinite number of episodes. We call this theassumption of exploring starts.

The assumption of exploring starts is sometimes useful, but of course it cannot berelied upon in general, particularly when learning directly from actual interaction with anenvironment. In that case the starting conditions are unlikely to be so helpful. The mostcommon alternative approach to assuring that all state–action pairs are encountered is

to consider only policies that are stochastic with a nonzero probability of selecting allactions in each state. We discuss two important variants of this approach in later sections.For now, we retain the assumption of exploring starts and complete the presentation of afull Monte Carlo control method.

Exercise 5.3 What is the backup diagram for Monte Carlo estimation of $q _ { \pi }$ ?

# 5.3 Monte Carlo Control

We are now ready to consider how Monte Carlo estimation can be used in control, thatis, to approximate optimal policies. The overall idea is to proceed according to the samepattern as in the DP chapter, that is, according to the idea of generalized policy iteration

(GPI). In GPI one maintains both an approximate policy andan approximate value function. The value function is repeatedlyaltered to more closely approximate the value function for thecurrent policy, and the policy is repeatedly improved with respectto the current value function, as suggested by the diagram tothe right. These two kinds of changes work against each other tosome extent, as each creates a moving target for the other, buttogether they cause both policy and value function to approachoptimality.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/9d3ccda2675de9911918a0f28404dba521171986548b3ca1e77be1505e1aa70f.jpg)


To begin, let us consider a Monte Carlo version of classical policy iteration. Inthis method, we perform alternating complete steps of policy evaluation and policyimprovement, beginning with an arbitrary policy $\pi _ { 0 }$ and ending with the optimal policyand optimal action-value function:

$$
\pi_ {0} \xrightarrow {\mathrm {E}} q _ {\pi_ {0}} \xrightarrow {\mathrm {I}} \pi_ {1} \xrightarrow {\mathrm {E}} q _ {\pi_ {1}} \xrightarrow {\mathrm {I}} \pi_ {2} \xrightarrow {\mathrm {E}} \dots \xrightarrow {\mathrm {I}} \pi_ {*} \xrightarrow {\mathrm {E}} q _ {*},
$$

where $\xrightarrow { \textrm { E } }$ denotes a complete policy evaluation and $\xrightarrow { \textrm { I } }$ denotes a complete policyimprovement. Policy evaluation is done exactly as described in the preceding section.Many episodes are experienced, with the approximate action-value function approachingthe true function asymptotically. For the moment, let us assume that we do indeedobserve an infinite number of episodes and that, in addition, the episodes are generatedwith exploring starts. Under these assumptions, the Monte Carlo methods will computeeach $q _ { \pi _ { k } }$ exactly, for arbitrary $\pi _ { k }$ .

Policy improvement is done by making the policy greedy with respect to the currentvalue function. In this case we have an action-value function, and therefore no model isneeded to construct the greedy policy. For any action-value function $q$ , the correspondinggreedy policy is the one that, for each $s \in \mathcal { S }$ , deterministically chooses an action withmaximal action-value:

$$
\pi (s) \doteq \arg \max  _ {a} q (s, a). \tag {5.1}
$$

Policy improvement then can be done by constructing each $\pi _ { k + 1 }$ as the greedy policywith respect to $q _ { \pi _ { k } }$ . The policy improvement theorem (Section 4.2) then applies to $\pi _ { k }$

and $\pi _ { k + 1 }$ because, for all $s \in \mathcal { S }$ ,

$$
\begin{array}{l} q _ {\pi_ {k}} (s, \pi_ {k + 1} (s)) = q _ {\pi_ {k}} (s, \underset {a} {\operatorname {a r g m a x}} q _ {\pi_ {k}} (s, a)) \\ = \max  _ {a} q _ {\pi_ {k}} (s, a) \\ \geq q _ {\pi_ {k}} (s, \pi_ {k} (s)) \\ \geq v _ {\pi_ {k}} (s). \\ \end{array}
$$

As we discussed in the previous chapter, the theorem assures us that each $\pi _ { k + 1 }$ is uniformlybetter than $\pi _ { k }$ , or just as good as $\pi _ { k }$ , in which case they are both optimal policies. Thisin turn assures us that the overall process converges to the optimal policy and optimalvalue function. In this way Monte Carlo methods can be used to find optimal policiesgiven only sample episodes and no other knowledge of the environment’s dynamics.

We made two unlikely assumptions above in order to easily obtain this guarantee ofconvergence for the Monte Carlo method. One was that the episodes have exploringstarts, and the other was that policy evaluation could be done with an infinite number ofepisodes. To obtain a practical algorithm we will have to remove both assumptions. Wepostpone consideration of the first assumption until later in this chapter.

For now we focus on the assumption that policy evaluation operates on an infinitenumber of episodes. This assumption is relatively easy to remove. In fact, the same issuearises even in classical DP methods such as iterative policy evaluation, which also convergeonly asymptotically to the true value function. In both DP and Monte Carlo cases thereare two ways to solve the problem. One is to hold firm to the idea of approximating $q _ { \pi _ { k } }$in each policy evaluation. Measurements and assumptions are made to obtain boundson the magnitude and probability of error in the estimates, and then su cient steps aretaken during each policy evaluation to assure that these bounds are su ciently small.This approach can probably be made completely satisfactory in the sense of guaranteeingcorrect convergence up to some level of approximation. However, it is also likely to requirefar too many episodes to be useful in practice on any but the smallest problems.

There is a second approach to avoiding the infinite number of episodes nominallyrequired for policy evaluation, in which we give up trying to complete policy evaluationbefore returning to policy improvement. On each evaluation step we move the valuefunction toward $q _ { \pi _ { k } }$ , but we do not expect to actually get close except over many steps.We used this idea when we first introduced the idea of GPI in Section 4.6. One extremeform of the idea is value iteration, in which only one iteration of iterative policy evaluationis performed between each step of policy improvement. The in-place version of valueiteration is even more extreme; there we alternate between improvement and evaluationsteps for single states.

For Monte Carlo policy iteration it is natural to alternate between evaluation andimprovement on an episode-by-episode basis. After each episode, the observed returnsare used for policy evaluation, and then the policy is improved at all the states visited inthe episode. A complete simple algorithm along these lines, which we call Monte Carlo$E S$ , for Monte Carlo with Exploring Starts, is given in pseudocode in the box on the nextpage.

Monte Carlo ES (Exploring Starts), for estimating  $\pi \approx \pi_{*}$    
Initialize:  $\pi (s)\in \mathcal{A}(s)$  (arbitrarily), for all  $s\in S$ $Q(s,a)\in \mathbb{R}$  (arbitrarily), for all  $s\in S,a\in \mathcal{A}(s)$  Returns(s,a)  $\leftarrow$  empty list, for all  $s\in S,a\in \mathcal{A}(s)$    
Loop forever (for each episode): Choose  $S_0\in \mathbb{S},A_0\in \mathcal{A}(S_0)$  randomly such that all pairs have probability  $>0$  Generate an episode from  $S_0,A_0$  , following  $\pi$  ..  $S_0,A_0,R_1,\ldots ,S_{T - 1},A_{T - 1},R_T$ $G\gets 0$  Loop for each step of episode,  $t = T - 1,T - 2,\dots ,0$ $G\gets \gamma G + R_{t + 1}$  Unless the pair  $S_{t},A_{t}$  appears in  $S_0,A_0,S_1,A_1\dots ,S_{t - 1},A_{t - 1}$  : Append  $G$  to Returns  $(S_t,A_t)$ $Q(S_{t},A_{t})\gets$  average(Returns  $(S_{t},A_{t}))$ $\pi (S_t)\gets \mathrm{argmax}_aQ(S_t,a)$

Exercise 5.4 The pseudocode for Monte Carlo ES is ine cient because, for each state–action pair, it maintains a list of all returns and repeatedly calculates their mean. It wouldbe more e cient to use techniques similar to those explained in Section 2.4 to maintainjust the mean and a count (for each state–action pair) and update them incrementally.Describe how the pseudocode would be altered to achieve this. ⇤

In Monte Carlo ES, all the returns for each state–action pair are accumulated andaveraged, irrespective of what policy was in force when they were observed. It is easyto see that Monte Carlo ES cannot converge to any suboptimal policy. If it did, thenthe value function would eventually converge to the value function for that policy, andthat in turn would cause the policy to change. Stability is achieved only when boththe policy and the value function are optimal. Convergence to this optimal fixed pointseems inevitable as the changes to the action-value function decrease over time, but hasnot yet been formally proved. In our opinion, this is one of the most fundamental opentheoretical questions in reinforcement learning (for a partial solution, see Tsitsiklis, 2002).

Example 5.3: Solving Blackjack It is straightforward to apply Monte Carlo ES toblackjack. Because the episodes are all simulated games, it is easy to arrange for exploringstarts that include all possibilities. In this case one simply picks the dealer’s cards, theplayer’s sum, and whether or not the player has a usable ace, all at random with equalprobability. As the initial policy we use the policy evaluated in the previous blackjackexample, that which sticks only on 20 or 21. The initial action-value function can be zerofor all state–action pairs. Figure 5.2 shows the optimal policy for blackjack found byMonte Carlo ES. This policy is the same as the “basic” strategy of Thorp (1966) with thesole exception of the leftmost notch in the policy for a usable ace, which is not presentin Thorp’s strategy. We are uncertain of the reason for this discrepancy, but confidentthat what is shown here is indeed the optimal policy for the version of blackjack we havedescribed.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/13e4789693c93697079cd149b49d7c0bb1b78169dc973c06d92887fccd7b3d4b.jpg)



mSTICK 19Figure 5.2: The optimal policy and state-value function for blackjack, found by Monte CarloNo su1718ES. The state-value function shown was computed from the action-value function found byusaMonte Carlo ES. ■


# A 2 3 4 5 6 7 8 9 10 115.4 Monte Carlo Control without Exploring Starts

How can we avoid the unlikely assumption of exploring starts? The only general way toensure that all actions are selected infinitely often is for the agent to continue to selectthem. There are two approaches to ensuring this, resulting in what we call on-policymethods and o↵-policy methods. On-policy methods attempt to evaluate or improve thepolicy that is used to make decisions, whereas o↵-policy methods evaluate or improvea policy di↵erent from that used to generate the data. The Monte Carlo ES methoddeveloped above is an example of an on-policy method. In this section we show how anon-policy Monte Carlo control method can be designed that does not use the unrealisticassumption of exploring starts. O↵-policy methods are considered in the next section.

In on-policy control methods the policy is generally soft, meaning that $\pi ( a | s ) > 0$for all $s \in \mathcal { S }$ and all $a \in { \mathcal { A } } ( s )$ , but gradually shifted closer and closer to a deterministicoptimal policy. Many of the methods discussed in Chapter 2 provide mechanisms forthis. The on-policy method we present in this section uses $\varepsilon$ -greedy policies, meaningthat most of the time they choose an action that has maximal estimated action value,but with probability $\varepsilon$ they instead select an action at random. That is, all nongreedyactions are given the minimal probability of selection, $\frac { \varepsilon } { | \mathcal { A } ( s ) | }$ , and the remaining bulk ofthe probability, $\begin{array} { r } { 1 - \varepsilon + \frac { \varepsilon } { | \mathcal { A } ( s ) | } } \end{array}$ "A(s) , is given to the greedy action. The "-greedy policies are $\varepsilon$examples of $\varepsilon$ -soft policies, defined as policies for which $\begin{array} { r } { \pi ( { a } | { s } ) \geq \frac { \varepsilon } { | \mathcal { A } ( { s } ) | } } \end{array}$ for all states andactions, for some $\varepsilon > 0$ . Among $\varepsilon$ -soft policies, $\varepsilon$ -greedy policies are in some sense thosethat are closest to greedy.

The overall idea of on-policy Monte Carlo control is still that of GPI. As in MonteCarlo ES, we use first-visit MC methods to estimate the action-value function for thecurrent policy. Without the assumption of exploring starts, however, we cannot simplyimprove the policy by making it greedy with respect to the current value function, becausethat would prevent further exploration of nongreedy actions. Fortunately, GPI does notrequire that the policy be taken all the way to a greedy policy, only that it be movedtoward a greedy policy. In our on-policy method we will move it only to an $\varepsilon$ -greedypolicy. For any $\varepsilon$ -soft policy, $\pi$ , any $\varepsilon$ -greedy policy with respect to $q _ { \pi }$ is guaranteed tobe better than or equal to $\pi$ . The complete algorithm is given in the box below.


On-policy first-visit MC control (for "-soft policies), estimates ⇡ ⇡ ⇡⇤


Algorithm parameter: small  $\varepsilon >0$    
Initialize:  $\pi \gets$  an arbitrary  $\varepsilon$  -soft policy  $Q(s,a)\in \mathbb{R}$  (arbitrarily), for all  $s\in S$ $a\in \mathcal{A}(s)$  Returns(s,a)  $\leftarrow$  empty list, for all  $s\in S$ $a\in \mathcal{A}(s)$    
Repeat forever (for each episode): Generate an episode following  $\pi$  ..  $S_0,A_0,R_1,\ldots ,S_{T - 1},A_{T - 1},R_T$ $G\gets 0$  Loop for each step of episode,  $t = T - 1,T - 2,\dots ,0$ $G\gets \gamma G + R_{t + 1}$  Unless the pair  $S_{t},A_{t}$  appears in  $S_0,A_0,S_1,A_1\dots ,S_{t - 1},A_{t - 1}$  . Append  $G$  to Returns(St,At)  $Q(S_{t},A_{t})\gets$  average(Returns(St,At))  $A^{*}\gets \operatorname {argmax}_{a}Q(S_{t},a)$  (with ties broken arbitrarily) For all  $a\in \mathcal{A}(S_t)$ $\pi (a|S_t)\gets \left\{ \begin{array}{ll}1 - \varepsilon +\varepsilon /|\mathcal{A}(S_t)| & \text{if} a = A^*\\ \varepsilon /|\mathcal{A}(S_t)| & \text{if} a\neq A^* \end{array} \right.$

That any $\varepsilon$ -greedy policy with respect to $q _ { \pi }$ is an improvement over any $\varepsilon$ -soft policy$\pi$ is assured by the policy improvement theorem. Let $\pi ^ { \prime }$ be the $\varepsilon$ -greedy policy. Theconditions of the policy improvement theorem apply because for any $s \in \mathcal { S }$ :

$$
\begin{array}{l} q _ {\pi} (s, \pi^ {\prime} (s)) = \sum_ {a} \pi^ {\prime} (a | s) q _ {\pi} (s, a) \\ = \frac {\varepsilon}{| \mathcal {A} (s) |} \sum_ {a} q _ {\pi} (s, a) + (1 - \varepsilon) \max  _ {a} q _ {\pi} (s, a) \tag {5.2} \\ \geq \frac {\varepsilon}{| \mathcal {A} (s) |} \sum_ {a} q _ {\pi} (s, a) + (1 - \varepsilon) \sum_ {a} \frac {\pi (a | s) - \frac {\varepsilon}{| \mathcal {A} (s) |}}{1 - \varepsilon} q _ {\pi} (s, a) \\ \end{array}
$$

(the sum is a weighted average with nonnegative weights summing to 1, and as such itmust be less than or equal to the largest number averaged)

$$
\begin{array}{l} = \frac {\varepsilon}{| \mathcal {A} (s) |} \sum_ {a} q _ {\pi} (s, a) - \frac {\varepsilon}{| \mathcal {A} (s) |} \sum_ {a} q _ {\pi} (s, a) + \sum_ {a} \pi (a | s) q _ {\pi} (s, a) \\ = v _ {\pi} (s). \\ \end{array}
$$

Thus, by the policy improvement theorem, $\pi ^ { \prime } \geq \pi$ (i.e., $v _ { \pi ^ { \prime } } ( s ) \geq v _ { \pi } ( s )$ , for all $s \in \mathcal { S }$ ). Wenow prove that equality can hold only when both $\pi ^ { \prime }$ and $\pi$ are optimal among the $\varepsilon$ -softpolicies, that is, when they are better than or equal to all other $\varepsilon$ -soft policies.

Consider a new environment that is just like the original environment, except with therequirement that policies be $\varepsilon$ -soft “moved inside” the environment. The new environmenthas the same action and state set as the original and behaves as follows. If in state $s$and taking action $a$ , then with probability $1 - \varepsilon$ the new environment behaves exactlylike the old environment. With probability $\varepsilon$ it repicks the action at random, with equalprobabilities, and then behaves like the old environment with the new, random action.The best one can do in this new environment with general policies is the same as thebest one could do in the original environment with $\varepsilon$ -soft policies. Let $\widetilde { v } _ { * }$ and $\widetilde { q } _ { * }$ denotethe optimal value functions for the new environment. Then a policy $\pi$ is optimal among$\varepsilon$ -soft policies if and only if $v _ { \pi } = \widetilde { v } _ { * }$ . We know that $\widetilde { v } _ { * }$ is the unique solution to theBellman optimality equation (3.19) with altered transition probabilities:

$$
\begin{array}{l} \widetilde {v} _ {*} (s) = \max  _ {a} \sum_ {s ^ {\prime}, r} \left[ (1 - \varepsilon) p \left(s ^ {\prime}, r \mid s, a\right) + \sum_ {a ^ {\prime}} \frac {\varepsilon}{\left| \mathcal {A} (s) \right|} p \left(s ^ {\prime}, r \mid s, a ^ {\prime}\right) \right] \left[ r + \gamma \widetilde {v} _ {*} \left(s ^ {\prime}\right) \right] \\ = (1 - \varepsilon) \max  _ {a} \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r \mid s, a\right) \left[ r + \gamma \widetilde {v} _ {*} \left(s ^ {\prime}\right) \right] \\ + \frac {\varepsilon}{| \mathcal {A} (s) |} \sum_ {a} \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r \mid s, a\right) \left[ r + \gamma \widetilde {v} _ {*} \left(s ^ {\prime}\right) \right]. \\ \end{array}
$$

When equality holds and the $\varepsilon$ -soft policy $\pi$ is no longer improved, then we also know,from (5.2), that

$$
\begin{array}{l} v _ {\pi} (s) = (1 - \varepsilon) \max  _ {a} q _ {\pi} (s, a) + \frac {\varepsilon}{| \mathcal {A} (s) |} \sum_ {a} q _ {\pi} (s, a) \\ = (1 - \varepsilon) \max  _ {a} \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r \mid s, a\right) \left[ r + \gamma v _ {\pi} \left(s ^ {\prime}\right) \right] \\ + \frac {\varepsilon}{| \mathcal {A} (s) |} \sum_ {a} \sum_ {s ^ {\prime}, r} p \left(s ^ {\prime}, r \mid s, a\right) \left[ r + \gamma v _ {\pi} \left(s ^ {\prime}\right) \right]. \\ \end{array}
$$

However, this equation is the same as the previous one, except for the substitution of $v _ { \pi }$for $\widetilde { v } _ { * }$ . Because $\widetilde { v } _ { * }$ is the unique solution, it must be that $v _ { \pi } = \widetilde { v } _ { * }$ .

In essence, we have shown in the last few pages that policy iteration works for $\varepsilon$ -softpolicies. Using the natural notion of greedy policy for $\varepsilon$ -soft policies, one is assured ofimprovement on every step, except when the best policy has been found among the $\varepsilon$ -softpolicies. This analysis is independent of how the action-value functions are determined

at each stage, but it does assume that they are computed exactly. This brings us toroughly the same point as in the previous section. Now we only achieve the best policyamong the $\varepsilon$ -soft policies, but on the other hand, we have eliminated the assumption ofexploring starts.

# 5.5 O↵-policy Prediction via Importance Sampling

All learning control methods face a dilemma: They seek to learn action values conditionalon subsequent optimal behavior, but they need to behave non-optimally in order toexplore all actions (to find the optimal actions). How can they learn about the optimalpolicy while behaving according to an exploratory policy? The on-policy approach in thepreceding section is actually a compromise—it learns action values not for the optimalpolicy, but for a near-optimal policy that still explores. A more straightforward approachis to use two policies, one that is learned about and that becomes the optimal policy, andone that is more exploratory and is used to generate behavior. The policy being learnedabout is called the target policy, and the policy used to generate behavior is called thebehavior policy. In this case we say that learning is from data “o↵” the target policy, andthe overall process is termed o↵-policy learning.

Throughout the rest of this book we consider both on-policy and o↵-policy methods.On-policy methods are generally simpler and are considered first. O↵-policy methodsrequire additional concepts and notation, and because the data is due to a di↵erent policy,o↵-policy methods are often of greater variance and are slower to converge. On the otherhand, o↵-policy methods are more powerful and general. They include on-policy methodsas the special case in which the target and behavior policies are the same. O↵-policymethods also have a variety of additional uses in applications. For example, they canoften be applied to learn from data generated by a conventional non-learning controller,or from a human expert. O↵-policy learning is also seen by some as key to learningmulti-step predictive models of the world’s dynamics (see Section 17.2; Sutton, 2009;Sutton et al., 2011).

In this section we begin the study of o↵-policy methods by considering the predictionproblem, in which both target and behavior policies are fixed. That is, suppose we wishto estimate $v _ { \pi }$ or $q _ { \pi }$ , but all we have are episodes following another policy $b$ , where$b \neq \pi$ . In this case, $\pi$ is the target policy, $b$ is the behavior policy, and both policies areconsidered fixed and given.

In order to use episodes from $b$ to estimate values for $\pi$ , we require that every actiontaken under $\pi$ is also taken, at least occasionally, under $b$ . That is, we require that$\pi ( a | s ) > 0$ implies $b ( a | s ) > 0$ . This is called the assumption of coverage. It followsfrom coverage that $b$ must be stochastic in states where it is not identical to $\pi$ . Thetarget policy $\pi$ , on the other hand, may be deterministic, and, in fact, this is a caseof particular interest in control applications. In control, the target policy is typicallythe deterministic greedy policy with respect to the current estimate of the action-valuefunction. This policy becomes a deterministic optimal policy while the behavior policyremains stochastic and more exploratory, for example, an $\varepsilon$ -greedy policy. In this section,however, we consider the prediction problem, in which $\pi$ is unchanging and given.

Almost all o↵-policy methods utilize importance sampling, a general technique forestimating expected values under one distribution given samples from another. We applyimportance sampling to o↵-policy learning by weighting returns according to the relativeprobability of their trajectories occurring under the target and behavior policies, calledthe importance-sampling ratio. Given a starting state $S _ { t }$ , the probability of the subsequentstate–action trajectory, $A _ { t } , S _ { t + 1 } , A _ { t + 1 } , \ldots , S _ { T }$ , occurring under any policy $\pi$ is

$$
\begin{array}{l} \Pr \left\{A _ {t}, S _ {t + 1}, A _ {t + 1}, \dots , S _ {T} \mid S _ {t}, A _ {t: T - 1} \sim \pi \right\} \\ = \pi (A _ {t} | S _ {t}) p (S _ {t + 1} | S _ {t}, A _ {t}) \pi (A _ {t + 1} | S _ {t + 1}) \dots p (S _ {T} | S _ {T - 1}, A _ {T - 1}) \\ = \prod_ {k = t} ^ {T - 1} \pi (A _ {k} | S _ {k}) p (S _ {k + 1} | S _ {k}, A _ {k}), \\ \end{array}
$$

where $p$ here is the state-transition probability function defined by (3.4). Thus, the relativeprobability of the trajectory under the target and behavior policies (the importance-sampling ratio) is

$$
\rho_ {t: T - 1} \doteq \frac {\prod_ {k = t} ^ {T - 1} \pi \left(A _ {k} \mid S _ {k}\right) p \left(S _ {k + 1} \mid S _ {k} , A _ {k}\right)}{\prod_ {k = t} ^ {T - 1} b \left(A _ {k} \mid S _ {k}\right) p \left(S _ {k + 1} \mid S _ {k} , A _ {k}\right)} = \prod_ {k = t} ^ {T - 1} \frac {\pi \left(A _ {k} \mid S _ {k}\right)}{b \left(A _ {k} \mid S _ {k}\right)}. \tag {5.3}
$$

Although the trajectory probabilities depend on the MDP’s transition probabilities, whichare generally unknown, they appear identically in both the numerator and denominator,and thus cancel. The importance sampling ratio ends up depending only on the twopolicies and the sequence, not on the MDP.

Recall that we wish to estimate the expected returns (values) under the target policy,but all we have are returns $G _ { t }$ due to the behavior policy. These returns have the wrongexpectation $\mathbb { E } [ G _ { t } | S _ { t } = s ] = v _ { b } ( s )$ and so cannot be averaged to obtain $v _ { \pi }$ . This is whereimportance sampling comes in. The ratio $\rho _ { t : T - 1 }$ transforms the returns to have the rightexpected value:

$$
\mathbb {E} \left[ \rho_ {t: T - 1} G _ {t} \mid S _ {t} = s \right] = v _ {\pi} (s). \tag {5.4}
$$

Now we are ready to give a Monte Carlo algorithm that averages returns from a batchof observed episodes following policy $b$ to estimate $v _ { \pi } ( s )$ . It is convenient here to numbertime steps in a way that increases across episode boundaries. That is, if the first episodeof the batch ends in a terminal state at time 100, then the next episode begins at time$t = 1 0 1$ . This enables us to use time-step numbers to refer to particular steps in particularepisodes. In particular, we can define the set of all time steps in which state $s$ is visited,denoted $\mathcal { T } ( s )$ . This is for an every-visit method; for a first-visit method, $\mathcal { T } ( s )$ would onlyinclude time steps that were first visits to $s$ within their episodes. Also, let $T ( t )$ denotethe first time of termination following time $t$ , and $G _ { t }$ denote the return after $t$ up through$T ( t )$ . Then $\{ G _ { t } \} _ { t \in \mathcal { T } ( s ) }$ are the returns that pertain to state $s$ , and $\left\{ \rho _ { t : T ( t ) - 1 } \right\} _ { t \in \mathcal { T } ( s ) }$ arethe corresponding importance-sampling ratios. To estimate $v _ { \pi } ( s )$ , we simply scale thereturns by the ratios and average the results:

$$
V (s) \doteq \frac {\sum_ {t \in \mathcal {T} (s)} \rho_ {t : T (t) - 1} G _ {t}}{| \mathcal {T} (s) |}. \tag {5.5}
$$

When importance sampling is done as a simple average in this way it is called ordinaryimportance sampling.

An important alternative is weighted importance sampling, which uses a weightedaverage, defined as

$$
V (s) \doteq \frac {\sum_ {t \in \mathcal {T} (s)} \rho_ {t : T (t) - 1} G _ {t}}{\sum_ {t \in \mathcal {T} (s)} \rho_ {t : T (t) - 1}}, \tag {5.6}
$$

or zero if the denominator is zero. To understand these two varieties of importancesampling, consider the estimates of their first-visit methods after observing a single returnfrom state $s$ . In the weighted-average estimate, the ratio $\rho _ { t : T ( t ) - 1 }$ for the single returncancels in the numerator and denominator, so that the estimate is equal to the observedreturn independent of the ratio (assuming the ratio is nonzero). Given that this returnwas the only one observed, this is a reasonable estimate, but its expectation is $v _ { b } ( s )$ ratherthan $v _ { \pi } ( s )$ , and in this statistical sense it is biased. In contrast, the first-visit versionof the ordinary importance-sampling estimator (5.5) is always $v _ { \pi } ( s )$ in expectation (itis unbiased), but it can be extreme. Suppose the ratio were ten, indicating that thetrajectory observed is ten times as likely under the target policy as under the behaviorpolicy. In this case the ordinary importance-sampling estimate would be ten times theobserved return. That is, it would be quite far from the observed return even though theepisode’s trajectory is considered very representative of the target policy.

Formally, the di↵erence between the first-visit methods of the two kinds of importancesampling is expressed in their biases and variances. Ordinary importance sampling isunbiased whereas weighted importance sampling is biased (though the bias convergesasymptotically to zero). On the other hand, the variance of ordinary importance samplingis in general unbounded because the variance of the ratios can be unbounded, whereas inthe weighted estimator the largest weight on any single return is one. In fact, assumingbounded returns, the variance of the weighted importance-sampling estimator convergesto zero even if the variance of the ratios themselves is infinite (Precup, Sutton, andDasgupta 2001). In practice, the weighted estimator usually has dramatically lowervariance and is strongly preferred. Nevertheless, we will not totally abandon ordinaryimportance sampling as it is easier to extend to the approximate methods using functionapproximation that we explore in the second part of this book.

The every-visit methods for ordinary and weighed importance sampling are both biased,though, again, the bias falls asymptotically to zero as the number of samples increases.In practice, every-visit methods are often preferred because they remove the need to keeptrack of which states have been visited and because they are much easier to extend toapproximations. A complete every-visit MC algorithm for o↵-policy policy evaluationusing weighted importance sampling is given in the next section on page 110.

Exercise 5.5 Consider an MDP with a single nonterminal state and a single actionthat transitions back to the nonterminal state with probability $p$ and transitions to theterminal state with probability $1 - p$ . Let the reward be +1 on all transitions, and let$\gamma = 1$ . Suppose you observe one episode that lasts 10 steps, with a return of 10. Whatare the first-visit and every-visit estimators of the value of the nonterminal state? ⇤

Example 5.4: O↵-policy Estimation of a Blackjack State Value We appliedboth ordinary and weighted importance-sampling methods to estimate the value of a singleblackjack state (Example 5.1) from o↵-policy data. Recall that one of the advantages ofMonte Carlo methods is that they can be used to evaluate a single state without formingestimates for any other states. In this example, we evaluated the state in which the dealeris showing a deuce, the sum of the player’s cards is 13, and the player has a usable ace (thatis, the player holds an ace and a deuce, or equivalently three aces). The data was generatedby starting in this state then choosing to hit or stick at random with equal probability(the behavior policy). The target policy was to stick only on a sum of 20 or 21, as inExample 5.1. The value of this state under the target policy is approximately  0.27726(this was determined by separately generating one-hundred million episodes using thetarget policy and averaging their returns). Both o↵-policy methods closely approximatedthis value after 1000 o↵-policy episodes using the random policy. To make sure they didthis reliably, we performed 100 independent runs, each starting from estimates of zeroand learning for 10,000 episodes. Figure 5.3 shows the resultant learning curves—thesquared error of the estimates of each method as a function of number of episodes,averaged over the 100 runs. The error approaches zero for both algorithms, but theweighted importance-sampling method has much lower error at the beginning, as is typicalin practice.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/b857933e400da477fdb7a964ec1b451e6d2e3c97a8001a69814ffd4b1095e8a4.jpg)



Figure 5.3: Weighted importance sampling produces lower error estimates of the value of asingle blackjack state from o↵-policy episodes.


Example 5.5: Infinite Variance The estimates of ordinary importance sampling willtypically have infinite variance, and thus unsatisfactory convergence properties, wheneverthe scaled returns have infinite variance—and this can easily happen in o↵-policy learningwhen trajectories contain loops. A simple example is shown inset in Figure 5.4. There isonly one nonterminal state $s$ and two actions, right and left. The right action causes adeterministic transition to termination, whereas the left action transitions, with probability0.9, back to $s$ or, with probability 0.1, on to termination. The rewards are $+ 1$ on thelatter transition and otherwise zero. Consider the target policy that always selects left.All episodes under this policy consist of some number (possibly zero) of transitions back

to $s$ followed by termination with a reward and return of $+ 1$ . Thus the value of $s$ underthe target policy is 1 ( $\gamma = 1$ ). Suppose we are estimating this value from o↵-policy datausing the behavior policy that selects right and left with equal probability.

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/4f1b8ff7c3a6bb46880962e8cfec55005c229d3214f5866cd3b3b85d52d9f902.jpg)



Figure 5.4: Ordinary importance sampling produces surprisingly unstable estimates on theone-state MDP shown inset (Example 5.5). The correct estimate here is 1 $\gamma = 1$ ), and, eventhough this is the expected value of a sample return (after importance sampling), the varianceof the samples is infinite, and the estimates do not converge to this value. These results are foro↵-policy first-visit MC.


The lower part of Figure 5.4 shows ten independent runs of the first-visit MC algorithmusing ordinary importance sampling. Even after millions of episodes, the estimates failto converge to the correct value of 1. In contrast, the weighted importance-samplingalgorithm would give an estimate of exactly 1 forever after the first episode that endedwith the left action. All returns not equal to 1 (that is, ending with the right action)would be inconsistent with the target policy and thus would have a $\rho _ { t : T ( t ) - 1 }$ of zero andcontribute neither to the numerator nor denominator of (5.6). The weighted importance-sampling algorithm produces a weighted average of only the returns consistent with thetarget policy, and all of these would be exactly 1.

We can verify that the variance of the importance-sampling-scaled returns is infinitein this example by a simple calculation. The variance of any random variable $X$ is theexpected value of the deviation from its mean $X$ , which can be written

$$
\operatorname {V a r} [ X ] \doteq \mathbb {E} \left[ \left(X - \bar {X}\right) ^ {2} \right] = \mathbb {E} \left[ X ^ {2} - 2 X \bar {X} + \bar {X} ^ {2} \right] = \mathbb {E} \left[ X ^ {2} \right] - \bar {X} ^ {2}.
$$

Thus, if the mean is finite, as it is in our case, the variance is infinite if and only if theexpectation of the square of the random variable is infinite. Thus, we need only show

that the expected square of the importance-sampling-scaled return is infinite:

$$
\mathbb {E} \left[ \left(\prod_ {t = 0} ^ {T - 1} \frac {\pi (A _ {t} | S _ {t})}{b (A _ {t} | S _ {t})} G _ {0}\right) ^ {2} \right].
$$

To compute this expectation, we break it down into cases based on episode length andtermination. First note that, for any episode ending with the right action, the importancesampling ratio is zero, because the target policy would never take this action; theseepisodes thus contribute nothing to the expectation (the quantity in parenthesis will bezero) and can be ignored. We need only consider episodes that involve some number(possibly zero) of left actions that transition back to the nonterminal state, followed by aleft action transitioning to termination. All of these episodes have a return of 1, so the$G _ { 0 }$ factor can be ignored. To get the expected square we need only consider each lengthof episode, multiplying the probability of the episode’s occurrence by the square of itsimportance-sampling ratio, and add these up:

$$
\begin{array}{l} = \frac {1}{2} \cdot 0. 1 \left(\frac {1}{0 . 5}\right) ^ {2} \quad (\text {t h e l e n g t h 1 e p i s o d e}) \\ + \frac {1}{2} \cdot 0. 9 \cdot \frac {1}{2} \cdot 0. 1 \left(\frac {1}{0 . 5} \frac {1}{0 . 5}\right) ^ {2} \quad (\text {t h e l e n g t h 2 e p i s o d e}) \\ + \frac {1}{2} \cdot 0. 9 \cdot \frac {1}{2} \cdot 0. 9 \cdot \frac {1}{2} \cdot 0. 1 \left(\frac {1}{0 . 5} \frac {1}{0 . 5} \frac {1}{0 . 5}\right) ^ {2} \quad (\text {t h e l e n g t h 3 e p i s o d e}) \\ + \dots \\ \end{array}
$$

$$
= 0. 1 \sum_ {k = 0} ^ {\infty} 0. 9 ^ {k} \cdot 2 ^ {k} \cdot 2 = 0. 2 \sum_ {k = 0} ^ {\infty} 1. 8 ^ {k} = \infty .
$$

Exercise 5.6 What is the equation analogous to (5.6) for action values $Q ( s , a )$ instead ofstate values $V ( s )$ , again given returns generated using $b$ ? ⇤

Exercise 5.7 In learning curves such as those shown in Figure 5.3 error generally decreaseswith training, as indeed happened for the ordinary importance-sampling method. But forthe weighted importance-sampling method error first increased and then decreased. Whydo you think this happened? ⇤

Exercise 5.8 The results with Example 5.5 and shown in Figure 5.4 used a first-visit MCmethod. Suppose that instead an every-visit MC method was used on the same problem.Would the variance of the estimator still be infinite? Why or why not? ⇤

# 5.6 Incremental Implementation

Monte Carlo prediction methods can be implemented incrementally, on an episode-by-episode basis, using extensions of the techniques described in Chapter 2 (Section 2.4).Whereas in Chapter 2 we averaged rewards, in Monte Carlo methods we average returns.In all other respects exactly the same methods as used in Chapter 2 can be used for on-policy Monte Carlo methods. For o↵-policy Monte Carlo methods, we need to separatelyconsider those that use ordinary importance sampling and those that use weightedimportance sampling.

In ordinary importance sampling, the returns are scaled by the importance samplingratio $\rho _ { t : T ( t ) - 1 }$ (5.3), then simply averaged, as in (5.5). For these methods we can againuse the incremental methods of Chapter 2, but using the scaled returns in place ofthe rewards of that chapter. This leaves the case of o↵-policy methods using weightedimportance sampling. Here we have to form a weighted average of the returns, and aslightly di↵erent incremental algorithm is required.

Suppose we have a sequence of returns $G _ { 1 } , G _ { 2 } , \dots , G _ { n - 1 }$ , all starting in the same stateand each with a corresponding random weight $W _ { i }$ (e.g., $W _ { i } = \rho _ { t _ { i } : T ( t _ { i } ) - 1 } )$ . We wish toform the estimate

$$
V _ {n} \doteq \frac {\sum_ {k = 1} ^ {n - 1} W _ {k} G _ {k}}{\sum_ {k = 1} ^ {n - 1} W _ {k}}, \quad n \geq 2, \tag {5.7}
$$

and keep it up-to-date as we obtain a single additional return $G _ { n }$ . In addition to keepingtrack of $V _ { n }$ , we must maintain for each state the cumulative sum $C _ { n }$ of the weights givento the first $n$ returns. The update rule for $V _ { n }$ is

$$
V _ {n + 1} \doteq V _ {n} + \frac {W _ {n}}{C _ {n}} \left[ G _ {n} - V _ {n} \right], \quad n \geq 1, \tag {5.8}
$$

and

$$
C _ {n + 1} \doteq C _ {n} + W _ {n + 1},
$$

where $C _ { 0 } \doteq 0$ (and $V _ { 1 }$ is arbitrary and thus need not be specified). The box on thenext page contains a complete episode-by-episode incremental algorithm for Monte Carlopolicy evaluation. The algorithm is nominally for the o↵-policy case, using weightedimportance sampling, but applies as well to the on-policy case just by choosing thetarget and behavior policies as the same (in which case ( $\pi = b$ ), $W$ is always 1). Theapproximation $Q$ converges to $q _ { \pi }$ (for all encountered state–action pairs) while actionsare selected according to a potentially di↵erent policy, $b$ .

Exercise 5.9 Modify the algorithm for first-visit MC policy evaluation (Section 5.1) touse the incremental implementation for sample averages described in Section 2.4. ⇤

Exercise 5.10 Derive the weighted-average update rule (5.8) from (5.7). Follow thepattern of the derivation of the unweighted rule (2.3). ⇤


O↵-policy MC prediction (policy evaluation) for estimating Q ⇡ q⇡


Input: an arbitrary target policy  $\pi$   
Initialize, for all  $s \in \mathcal{S}$ ,  $a \in \mathcal{A}(s)$ :  
 $Q(s, a) \in \mathbb{R}$  (arbitrarily)  
 $C(s, a) \gets 0$   
Loop forever (for each episode):  
 $b \gets$  any policy with coverage of  $\pi$   
Generate an episode following  $b$ :  $S_0, A_0, R_1, \ldots, S_{T-1}, A_{T-1}, R_T$ $G \gets 0$ $W \gets 1$   
Loop for each step of episode,  $t = T-1, T-2, \ldots, 0$ , while  $W \neq 0$ :  
 $G \gets \gamma G + R_{t+1}$ $C(S_t, A_t) \gets C(S_t, A_t) + W$ $Q(S_t, A_t) \gets Q(S_t, A_t) + \frac{W}{C(S_t, A_t)} [G - Q(S_t, A_t)]$ $W \gets W \frac{\pi(A_t | S_t)}{b(A_t | S_t)}$

# 5.7 O↵-policy Monte Carlo Control

We are now ready to present an example of the second class of learning control methodswe consider in this book: o↵-policy methods. Recall that the distinguishing feature ofon-policy methods is that they estimate the value of a policy while using it for control.In o↵-policy methods these two functions are separated. The policy used to generatebehavior, called the behavior policy, may in fact be unrelated to the policy that isevaluated and improved, called the target policy. An advantage of this separation isthat the target policy may be deterministic (e.g., greedy), while the behavior policy cancontinue to sample all possible actions.

O↵-policy Monte Carlo control methods use one of the techniques presented in thepreceding two sections. They follow the behavior policy while learning about andimproving the target policy. These techniques require that the behavior policy has anonzero probability of selecting all actions that might be selected by the target policy(coverage). To explore all possibilities, we require that the behavior policy be soft (i.e.,that it select all actions in all states with nonzero probability).

The box on the next page shows an o↵-policy Monte Carlo control method, based onGPI and weighted importance sampling, for estimating $\pi _ { * }$ and $q _ { * }$ . The target policy$\pi \approx \pi _ { * }$ is the greedy policy with respect to $Q$ , which is an estimate of $q _ { \pi }$ . The behaviorpolicy $b$ can be anything, but in order to assure convergence of $\pi$ to the optimal policy, aninfinite number of returns must be obtained for each pair of state and action. This can beassured by choosing $b$ to be $\varepsilon$ -soft. The policy $\pi$ converges to optimal at all encounteredstates even though actions are selected according to a di↵erent soft policy $b$ , which maychange between or even within episodes.


O↵-policy MC control, for estimating ⇡ ⇡ ⇡⇤


Initialize, for all  $s\in \mathcal{S}$ $a\in \mathcal{A}(s)$ $Q(s,a)\in \mathbb{R}$  (arbitrarily)  $C(s,a)\gets 0$ $\pi (s)\leftarrow \operatorname {argmax}_aQ(s,a)\quad (\text{with ties broken consistently})$  Loop forever (for each episode):  $b\gets$  any soft policy Generate an episode using  $b$  ..  $S_0,A_0,R_1,\ldots ,S_{T - 1},A_{T - 1},RT$ $G\gets 0$ $W\gets 1$  Loop for each step of episode,  $t = T - 1,T - 2,\dots ,0$  .  $G\gets \gamma G + R_{t + 1}$ $C(S_t,A_t)\gets C(S_t,A_t) + W$ $Q(S_{t},A_{t})\gets Q(S_{t},A_{t}) + \frac{W}{C(S_{t},A_{t})} [G - Q(S_{t},A_{t})]$ $\pi (S_t)\gets \mathrm{argmax}_aQ(S_t,a)$  (with ties broken consistently) If  $A_{t}\neq \pi (S_{t})$  then exit inner Loop (proceed to next episode)  $W\gets W\frac{1}{b(A_t|S_t)}$

A potential problem is that this method learns only from the tails of episodes, whenall of the remaining actions in the episode are greedy. If nongreedy actions are common,then learning will be slow, particularly for states appearing in the early portions oflong episodes. Potentially, this could greatly slow learning. There has been insu cientexperience with o↵-policy Monte Carlo methods to assess how serious this problem is. Ifit is serious, the most important way to address it is probably by incorporating temporal-di↵erence learning, the algorithmic idea developed in the next chapter. Alternatively, if $\gamma$is less than 1, then the idea developed in the next section may also help significantly.

Exercise 5.11 In the boxed algorithm for o↵-policy MC control, you may have beenexpecting the $W$ update to have involved the importance-sampling ratio $\frac { \pi ( A _ { t } | S _ { t } ) } { b ( A _ { t } | S _ { t } ) }$ , butinstead it involves $\frac { 1 } { b ( A _ { t } | S _ { t } ) }$ . Why is this nevertheless correct? ⇤

Exercise 5.12: Racetrack (programming) Consider driving a race car around a turnlike those shown in Figure 5.5. You want to go as fast as possible, but not so fast asto run o↵ the track. In our simplified racetrack, the car is at one of a discrete set ofgrid positions, the cells in the diagram. The velocity is also discrete, a number of gridcells moved horizontally and vertically per time step. The actions are increments to thevelocity components. Each may be changed by $+ 1$ , $^ { - 1 }$ , or 0 in each step, for a total ofnine $\left( 3 \times 3 \right)$ actions. Both velocity components are restricted to be nonnegative and lessthan 5, and they cannot both be zero except at the starting line. Each episode beginsin one of the randomly selected start states with both velocity components zero andends when the car crosses the finish line. The rewards are $^ { - 1 }$ for each step until the carcrosses the finish line. If the car hits the track boundary, it is moved back to a randomposition on the starting line, both velocity components are reduced to zero, and the

![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/6b719579a6c2b4b82cf4ee018b2ca4483733640c627179bf2aa4cfbbbe8ff151.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-02-07/86bd4ca5-f410-4074-81e6-d775988a05ab/fc4735606af8bc64f8474ef6570e2b1e1152c2e2869b4acb1b9b58f7a9425743.jpg)



Figure 5.5: A couple of right turns for the racetrack task.


episode continues. Before updating the car’s location at each time step, check to see ifthe projected path of the car intersects the track boundary. If it intersects the finish line,the episode ends; if it intersects anywhere else, the car is considered to have hit the trackboundary and is sent back to the starting line. To make the task more challenging, withprobability 0.1 at each time step the velocity increments are both zero, independently ofthe intended increments. Apply a Monte Carlo control method to this task to computethe optimal policy from each starting state. Exhibit several trajectories following theoptimal policy (but turn the noise o↵ for these trajectories). ⇤

# 5.8 *Discounting-aware Importance Sampling

The o↵-policy methods that we have considered so far are based on forming importance-sampling weights for returns considered as unitary wholes, without taking into accountthe returns’ internal structures as sums of discounted rewards. We now briefly considercutting-edge research ideas for using this structure to significantly reduce the variance ofo↵-policy estimators.

For example, consider the case where episodes are long and $\gamma$ is significantly less than1. For concreteness, say that episodes last 100 steps and that $\gamma = 0$ . The return fromtime 0 will then be just $G _ { 0 } = R _ { 1 }$ , but its importance sampling ratio will be a product of100 factors, b(A0|S0) b(A1|S1) ⇡(A0|S0) ⇡(A1|S1) ${ \frac { \pi ( A _ { 0 } | S _ { 0 } ) } { b ( A _ { 0 } | S _ { 0 } ) } } { \frac { \pi ( A _ { 1 } | S _ { 1 } ) } { b ( A _ { 1 } | S _ { 1 } ) } } \cdot \cdot \cdot { \frac { \pi ( A _ { 9 9 } | S _ { 9 9 } ) } { b ( A _ { 9 9 } | S _ { 9 9 } ) } }$ In ordinary importance sampling, the returnwill be scaled by the entire product, but it is really only necessary to scale by the firstfactor, by $\frac { \pi ( A _ { 0 } | S _ { 0 } ) } { b ( A _ { 0 } | S _ { 0 } ) }$ . The other 99 factors ${ \frac { \pi ( A _ { 1 } | S _ { 1 } ) } { b ( A _ { 1 } | S _ { 1 } ) } } \cdot \cdot \cdot { \frac { \pi ( A _ { 9 9 } | S _ { 9 9 } ) } { b ( A _ { 9 9 } | S _ { 9 9 } ) } }$ are irrelevant becauseafter the first reward the return has already been determined. These later factors areall independent of the return and of expected value 1; they do not change the expectedupdate, but they add enormously to its variance. In some cases they could even make thevariance infinite. Let us now consider an idea for avoiding this large extraneous variance.

The essence of the idea is to think of discounting as determining a probability oftermination or, equivalently, a degree of partial termination. For any $\gamma \in [ 0 , 1 )$ , we canthink of the return $G _ { 0 }$ as partly terminating in one step, to the degree $1 - \gamma$ , producinga return of just the first reward, $R _ { 1 }$ , and as partly terminating after two steps, to thedegree $( 1 - \gamma ) \gamma$ , producing a return of $R _ { 1 } + R _ { 2 }$ , and so on. The latter degree correspondsto terminating on the second step, $1 - \gamma$ , and not having already terminated on thefirst step, $\gamma$ . The degree of termination on the third step is thus $( 1 - \gamma ) \gamma ^ { 2 }$ , with the $\gamma ^ { 2 }$reflecting that termination did not occur on either of the first two steps. The partialreturns here are called flat partial returns:

$$
\bar {G} _ {t: h} \doteq R _ {t + 1} + R _ {t + 2} + \dots + R _ {h}, \quad 0 \leq t <   h \leq T,
$$

where “flat” denotes the absence of discounting, and “partial” denotes that these returnsdo not extend all the way to termination but instead stop at $h$ , called the horizon (and $T$is the time of termination of the episode). The conventional full return $G _ { t }$ can be viewedas a sum of flat partial returns as suggested above as follows:

$$
\begin{array}{l} G _ {t} \dot {=} R _ {t + 1} + \gamma R _ {t + 2} + \gamma^ {2} R _ {t + 3} + \dots + \gamma^ {T - t - 1} R _ {T} \\ = (1 - \gamma) R _ {t + 1} \\ + (1 - \gamma) \gamma \left(R _ {t + 1} + R _ {t + 2}\right) \\ + (1 - \gamma) \gamma^ {2} \left(R _ {t + 1} + R _ {t + 2} + R _ {t + 3}\right) \\ \end{array}
$$

$$
\begin{array}{l} + (1 - \gamma) \gamma^ {T - t - 2} \left(R _ {t + 1} + R _ {t + 2} + \dots + R _ {T - 1}\right) \\ + \gamma^ {T - t - 1} \left(R _ {t + 1} + R _ {t + 2} + \dots + R _ {T}\right) \\ = (1 - \gamma) \sum_ {h = t + 1} ^ {T - 1} \gamma^ {h - t - 1} \bar {G} _ {t: h} + \gamma^ {T - t - 1} \bar {G} _ {t: T}. \\ \end{array}
$$

Now we need to scale the flat partial returns by an importance sampling ratio thatis similarly truncated. As $G _ { t : h }$ only involves rewards up to a horizon $h$ , we only needthe ratio of the probabilities up to $h - 1$ . We define an ordinary importance-samplingestimator, analogous to (5.5), as

$$
V (s) \doteq \frac {\sum_ {t \in \mathcal {T} (s)} \left((1 - \gamma) \sum_ {h = t + 1} ^ {T (t) - 1} \gamma^ {h - t - 1} \rho_ {t : h - 1} \bar {G} _ {t : h} + \gamma^ {T (t) - t - 1} \rho_ {t : T (t) - 1} \bar {G} _ {t : T (t)}\right)}{| \mathcal {T} (s) |}, \tag {5.9}
$$

and a weighted importance-sampling estimator, analogous to (5.6), as

$$
V (s) \doteq \frac {\sum_ {t \in \mathcal {T} (s)} \left((1 - \gamma) \sum_ {h = t + 1} ^ {T (t) - 1} \gamma^ {h - t - 1} \rho_ {t : h - 1} \bar {G} _ {t : h} + \gamma^ {T (t) - t - 1} \rho_ {t : T (t) - 1} \bar {G} _ {t : T (t)}\right)}{\sum_ {t \in \mathcal {T} (s)} \left(\left(1 - \gamma\right) \sum_ {h = t + 1} ^ {T (t) - 1} \gamma^ {h - t - 1} \rho_ {t : h - 1} + \gamma^ {T (t) - t - 1} \rho_ {t : T (t) - 1}\right)}. \tag {5.10}
$$

We call these two estimators discounting-aware importance sampling estimators. Theytake into account the discount rate but have no e↵ect (are the same as the o↵-policyestimators from Section 5.5) if $\gamma = 1$ .

# 5.9 *Per-decision Importance Sampling

There is one more way in which the structure of the return as a sum of rewards can betaken into account in o↵-policy importance sampling, a way that may be able to reducevariance even in the absence of discounting (that is, even if $\gamma = 1$ ). In the o↵-policyestimators (5.5) and (5.6), each term of the sum in the numerator is itself a sum:

$$
\begin{array}{l} \rho_ {t: T - 1} G _ {t} = \rho_ {t: T - 1} \left(R _ {t + 1} + \gamma R _ {t + 2} + \dots + \gamma^ {T - t - 1} R _ {T}\right) \\ = \rho_ {t: T - 1} R _ {t + 1} + \gamma \rho_ {t: T - 1} R _ {t + 2} + \dots + \gamma^ {T - t - 1} \rho_ {t: T - 1} R _ {T}. \tag {5.11} \\ \end{array}
$$

The o↵-policy estimators rely on the expected values of these terms, which can be writtenin a simpler way. Note that each sub-term of (5.11) is a product of a random reward anda random importance-sampling ratio. For example, the first sub-term can be written,using (5.3), as

$$
\rho_ {t: T - 1} R _ {t + 1} = \frac {\pi (A _ {t} | S _ {t})}{b (A _ {t} | S _ {t})} \frac {\pi (A _ {t + 1} | S _ {t + 1})}{b (A _ {t + 1} | S _ {t + 1})} \frac {\pi (A _ {t + 2} | S _ {t + 2})}{b (A _ {t + 2} | S _ {t + 2})} \dots \frac {\pi (A _ {T - 1} | S _ {T - 1})}{b (A _ {T - 1} | S _ {T - 1})} R _ {t + 1}. \tag {5.12}
$$

Of all these factors, one might suspect that only the first and the last (the reward)are related; all the others are for events that occurred after the reward. Moreover, theexpected value of all these other factors is one:

$$
\mathbb {E} \left[ \frac {\pi \left(A _ {k} \mid S _ {k}\right)}{b \left(A _ {k} \mid S _ {k}\right)} \right] \doteq \sum_ {a} b (a \mid S _ {k}) \frac {\pi (a \mid S _ {k})}{b (a \mid S _ {k})} = \sum_ {a} \pi (a \mid S _ {k}) = 1. \tag {5.13}
$$

With a few more steps, one can show that, as suspected, all of these other factors haveno e↵ect in expectation, in other words, that

$$
\mathbb {E} \left[ \rho_ {t: T - 1} R _ {t + 1} \right] = \mathbb {E} \left[ \rho_ {t: t} R _ {t + 1} \right]. \tag {5.14}
$$

If we repeat this process for the $k$ th sub-term of (5.11), we get

$$
\mathbb {E} \left[ \rho_ {t: T - 1} R _ {t + k} \right] = \mathbb {E} \left[ \rho_ {t: t + k - 1} R _ {t + k} \right].
$$

It follows then that the expectation of our original term (5.11) can be written

$$
\mathbb {E} [ \rho_ {t: T - 1} G _ {t} ] = \mathbb {E} \left[ \tilde {G} _ {t} \right],
$$

where

$$
\tilde {G} _ {t} = \rho_ {t: t} R _ {t + 1} + \gamma \rho_ {t: t + 1} R _ {t + 2} + \gamma^ {2} \rho_ {t: t + 2} R _ {t + 3} + \dots + \gamma^ {T - t - 1} \rho_ {t: T - 1} R _ {T}.
$$

We call this idea per-decision importance sampling.

It follows immediately that there is an alternate importance-sampling estimator, withthe same unbiased expectation (in the first-visit case) as the ordinary-importance-samplingestimator (5.5), using $\dot { G } _ { t }$ :

$$
V (s) \doteq \frac {\sum_ {t \in \mathcal {T} (s)} \tilde {G} _ {t}}{| \mathcal {T} (s) |}, \tag {5.15}
$$

which we might expect to sometimes be of lower variance.

Is there a per-decision version of weighted importance sampling? This is less clear. Sofar, all the estimators that have been proposed for this that we know of are not consistent(that is, they do not converge to the true value with infinite data).

⇤ Exercise 5.13 Show the steps to derive (5.14) from (5.12).

⇤ Exercise 5.14 Modify the algorithm for o↵-policy Monte Carlo control (page 111) to usethe idea of the truncated weighted-average estimator (5.10). Note that you will first needto convert this equation to action values. ⇤

# 5.10 Summary

The Monte Carlo methods presented in this chapter learn value functions and optimalpolicies from experience in the form of sample episodes. This gives them at least threekinds of advantages over DP methods. First, they can be used to learn optimal behaviordirectly from interaction with the environment, with no model of the environment’sdynamics. Second, they can be used with simulation or sample models. For surprisinglymany applications it is easy to simulate sample episodes even though it is di cult toconstruct the kind of explicit model of transition probabilities required by DP methods.Third, it is easy and e cient to focus Monte Carlo methods on a small subset of the states.A region of special interest can be accurately evaluated without going to the expense ofaccurately evaluating the rest of the state set (we explore this further in Chapter 8).

A fourth advantage of Monte Carlo methods, which we discuss later in the book, isthat they may be less harmed by violations of the Markov property. This is because theydo not update their value estimates on the basis of the value estimates of successor states.In other words, it is because they do not bootstrap.

In designing Monte Carlo control methods we have followed the overall schema ofgeneralized policy iteration (GPI) introduced in Chapter 4. GPI involves interactingprocesses of policy evaluation and policy improvement. Monte Carlo methods provide analternative policy evaluation process. Rather than use a model to compute the value ofeach state, they simply average many returns that start in the state. Because a state’svalue is the expected return, this average can become a good approximation to thevalue. In control methods we are particularly interested in approximating action-valuefunctions, because these can be used to improve the policy without requiring a model ofthe environment’s transition dynamics. Monte Carlo methods intermix policy evaluationand policy improvement steps on an episode-by-episode basis, and can be incrementallyimplemented on an episode-by-episode basis.

Maintaining su cient exploration is an issue in Monte Carlo control methods. It isnot enough just to select the actions currently estimated to be best, because then noreturns will be obtained for alternative actions, and it may never be learned that theyare actually better. One approach is to ignore this problem by assuming that episodesbegin with state–action pairs randomly selected to cover all possibilities. Such exploringstarts can sometimes be arranged in applications with simulated episodes, but are unlikelyin learning from real experience. In on-policy methods, the agent commits to alwaysexploring and tries to find the best policy that still explores. In o↵-policy methods, theagent also explores, but learns a deterministic optimal policy that may be unrelated tothe policy followed.

O↵-policy prediction refers to learning the value function of a target policy from datagenerated by a di↵erent behavior policy. Such learning methods are based on some formof importance sampling, that is, on weighting returns by the ratio of the probabilities oftaking the observed actions under the two policies, thereby transforming their expectationsfrom the behavior policy to the target policy. Ordinary importance sampling uses asimple average of the weighted returns, whereas weighted importance sampling uses aweighted average. Ordinary importance sampling produces unbiased estimates, but haslarger, possibly infinite, variance, whereas weighted importance sampling always hasfinite variance and is preferred in practice. Despite their conceptual simplicity, o↵-policyMonte Carlo methods for both prediction and control remain unsettled and are a subjectof ongoing research.

The Monte Carlo methods treated in this chapter di↵er from the DP methods treatedin the previous chapter in two major ways. First, they operate on sample experience,and thus can be used for direct learning without a model. Second, they do not bootstrap.That is, they do not update their value estimates on the basis of other value estimates.These two di↵erences are not tightly linked, and can be separated. In the next chapterwe consider methods that learn from experience, like Monte Carlo methods, but alsobootstrap, like DP methods.

# Bibliographical and Historical Remarks

The term “Monte Carlo” dates from the 1940s, when physicists at Los Alamos devisedgames of chance that they could study to help understand complex physical phenomenarelating to the atom bomb. Coverage of Monte Carlo methods in this sense can be foundin several textbooks (e.g., Kalos and Whitlock, 1986; Rubinstein, 1981).

5.1–2 Singh and Sutton (1996) distinguished between every-visit and first-visit MCmethods and proved results relating these methods to reinforcement learningalgorithms. The blackjack example is based on an example used by Widrow,Gupta, and Maitra (1973). The soap bubble example is a classical Dirichletproblem whose Monte Carlo solution was first proposed by Kakutani (1945; seeHersh and Griego, 1969; Doyle and Snell, 1984).

Barto and Du↵ (1994) discussed policy evaluation in the context of classicalMonte Carlo algorithms for solving systems of linear equations. They used the

analysis of Curtiss (1954) to point out the computational advantages of MonteCarlo policy evaluation for large problems.

5.3–4 Monte Carlo ES was introduced in the 1998 edition of this book. That may havebeen the first explicit connection between Monte Carlo estimation and controlmethods based on policy iteration. An early use of Monte Carlo methods toestimate action values in a reinforcement learning context was by Michie andChambers (1968). In pole balancing (page 56), they used averages of episodedurations to assess the worth (expected balancing “life”) of each possible actionin each state, and then used these assessments to control action selections. Theirmethod is similar in spirit to Monte Carlo ES with every-visit MC estimates.Narendra and Wheeler (1986) studied a Monte Carlo method for ergodic finiteMarkov chains that used the return accumulated between successive visits to thesame state as a reward for adjusting a learning automaton’s action probabilities.

5.5 E cient o↵-policy learning has become recognized as an important challengethat arises in several fields. For example, it is closely related to the idea of“interventions” and “counterfactuals” in probabilistic graphical (Bayesian) models(e.g., Pearl, 1995; Balke and Pearl, 1994). O↵-policy methods using importancesampling have a long history and yet still are not well understood. Weightedimportance sampling, which is also sometimes called normalized importancesampling (e.g., Koller and Friedman, 2009), is discussed by Rubinstein (1981),Hesterberg (1988), Shelton (2001), and Liu (2001) among others.

The target policy in o↵-policy learning is sometimes referred to in the literatureas the “estimation” policy, as it was in the first edition of this book.

5.7 The racetrack exercise is adapted from Barto, Bradtke, and Singh (1995), andfrom Gardner (1973).

5.8 Our treatment of the idea of discounting-aware importance sampling is based onthe analysis of Sutton, Mahmood, Precup, and van Hasselt (2014). It has beenworked out most fully to date by Mahmood (2017; Mahmood, van Hasselt, andSutton, 2014).

5.9 Per-decision importance sampling was introduced by Precup, Sutton, and Singh(2000). They also combined o↵-policy learning with temporal-di↵erence learning,eligibility traces, and approximation methods, introducing subtle issues that weconsider in later chapters.

Exercise 5.15 Make new equations analogous to the importance-sampling Monte Carloestimates (5.5) and (5.6), but for action value estimates $Q ( s , a )$ . You will need newnotation $\mathcal { T } ( s , a )$ for the time steps on which the state–action pair $s , a$ is visited on theepisode. Do these estimates involve more or less importance-sampling correction?

