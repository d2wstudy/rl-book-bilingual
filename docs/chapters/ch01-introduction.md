---
title: 第1章 Introduction 引言
---

## 1.1 Reinforcement Learning

::: en
The idea that we learn by interacting with our environment is probably the first to occur to us when we think about the nature of learning. When an infant plays, waves its arms, or looks about, it has no explicit teacher, but it does have a direct sensorimotor connection to its environment. Exercising this connection produces a wealth of information about cause and effect, about the consequences of actions, and about what to do in order to achieve goals. Throughout our lives, such interactions are undoubtedly a major source of knowledge about our environment and ourselves.
:::

::: zh
当我们思考学习的本质时，最先想到的可能就是：我们通过与环境的交互来学习。婴儿玩耍、挥动手臂或四处张望时，并没有一个明确的老师，但他与环境之间存在直接的<Anno t="感觉运动连接是认知科学中的核心概念，强调学习不是被动接收而是主动探索。这段话奠定了全书的基调：RL 的核心是 agent 与 environment 的交互">感觉运动连接</Anno>。通过这种连接，他获得了大量关于因果关系、行为后果以及如何实现目标的信息。在我们的一生中，这种交互无疑是我们了解环境和自身的主要知识来源。
:::

::: en
In this book we explore a computational approach to learning from interaction. Rather than directly theorizing about how people or animals learn, we explore idealized learning situations and evaluate the effectiveness of various learning methods. That is, we take more of an engineering approach. We call the approach we explore *reinforcement learning*, much more focused than the most general meaning of this term in psychology, but we believe it is an important component of a broader understanding of intelligence.
:::

::: zh
本书探索一种从交互中学习的<Anno t="这里的计算方法特指基于试错的方法，区别于监督学习中由老师直接给出正确答案的范式">计算方法</Anno>。我们不直接对人或动物如何学习进行理论化，而是探索理想化的学习情境，并评估各种学习方法的有效性。也就是说，我们采取的更多是一种工程方法。我们将所探索的方法称为*强化学习*，它比心理学中该术语最一般的含义更为聚焦，但我们相信它是更广泛地理解智能的重要组成部分。
:::

::: en
Reinforcement learning, like many topics whose names end with "ing," such as machine learning and mountaineering, is simultaneously a problem, a class of solution methods that work well on the problem, and the field that studies this problem and its solution methods. It is convenient to use a single name for all three things, but at the same time essential to keep the three conceptually separate. In particular, the distinction between problems and solution methods is very important in reinforcement learning; failing to make this distinction is the source of many confusions.
:::

::: zh
强化学习，就像许多以"ing"结尾的主题（如机器学习和登山运动）一样，同时是一个<Anno t="Sutton 特意强调了 RL 的三重含义：(1) 一个问题定义 (2) 一类算法 (3) 一个研究领域。后续章节会反复体现这种区分">问题</Anno>、一类在该问题上表现良好的求解方法，以及研究该问题及其求解方法的领域。用一个名称来指代这三者很方便，但同时必须在概念上将它们区分开来。特别是，问题与求解方法之间的区别在强化学习中非常重要；未能做出这种区分是许多困惑的根源。
:::

::: en
We formalize the problem of reinforcement learning using ideas from dynamical systems theory, specifically, as the optimal control of incompletely-known Markov decision processes. The basic idea is simply that the most important aspects of the real problem facing a learning agent are captured in three signals passing back and forth between an agent and its environment: one signal to represent the choices made by the agent (the actions), one signal to represent the basis on which the choices are made (the states), and one signal to define the agent's goal (the rewards). This framework is intended to be a simple abstraction of the problem of goal-directed learning from interaction.
:::

::: zh
我们使用动力系统理论的思想来形式化强化学习问题，具体来说，将其视为<Anno t="'不完全已知'点明了 RL 与经典动态规划的本质区别——环境模型未知，必须通过交互来学习">不完全已知的马尔可夫决策过程</Anno>的最优控制。基本思想很简单：面对学习智能体的真实问题，其最重要的方面可以用智能体与环境之间来回传递的三个信号来捕获：一个信号表示智能体做出的选择（<Anno t="action：智能体在每个时间步可以执行的操作">动作</Anno>），一个信号表示做出选择的依据（<Anno t="state：环境的当前状况，是智能体决策的依据">状态</Anno>），一个信号定义智能体的目标（<Anno t="reward：一个标量信号，定义了什么是'好的'，是 RL 的核心驱动力">奖励</Anno>）。这个框架旨在作为从交互中进行目标导向学习问题的简单抽象。
:::

## 1.2 Examples

::: en
A good way to understand reinforcement learning is to consider some of the examples and possible applications that have guided its development.
:::

::: zh
理解强化学习的一个好方法是考虑一些引导其发展的例子和可能的应用。
:::

::: en
- A master chess player makes a move. The choice is informed both by planning—anticipating possible replies and counterreplies—and by immediate, intuitive judgments of the desirability of particular positions and moves.

- An adaptive controller adjusts parameters of a petroleum refinery's operation in real time. The controller optimizes the yield/cost/quality trade-off on the basis of specified marginal costs without sticking strictly to the set points originally suggested by engineers.

- A gazelle calf struggles to its feet minutes after being born. Half an hour later it is running at 20 miles per hour.
:::

::: zh
- 一位国际象棋大师走了一步棋。这个选择既来自<Anno t="这里的 planning 对应后面第8章的内容，即通过内部模型进行前瞻搜索">规划</Anno>——预测可能的回应和反回应——也来自对特定局面和走法的即时直觉判断。

- 一个自适应控制器实时调整石油精炼厂的运行参数。该控制器根据指定的边际成本优化产量/成本/质量的权衡，而不严格遵循工程师最初建议的设定点。

- 一只小瞪羚出生几分钟后就挣扎着站了起来。半小时后它就能以每小时20英里的速度奔跑。
:::

---

<ChapterComments />
