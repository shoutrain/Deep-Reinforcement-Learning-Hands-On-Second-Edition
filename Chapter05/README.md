# Chapter 05 详解:表格型强化学习(Tabular RL)与贝尔曼方程

> 本章正式进入**经典 RL 理论**。它不再用神经网络"暴力模仿",而是用一张表格精确求解贝尔曼方程(Bellman Equation)。两个例子分别是:
>
> - `01_frozenlake_v_iteration.py` — **价值迭代 (Value Iteration)**
> - `02_frozenlake_q_iteration.py` — **Q 价值迭代 (Q-Value Iteration)**
>
> 两个算法**同属 Model-Based 表格型 DP**家族,只差一个表格列。本章末尾会精确对比它们的差异。

---

## 1. 关键背景:MDP 与贝尔曼方程

强化学习问题通常被建模为**马尔可夫决策过程 (MDP)** $\langle S, A, P, R, \gamma \rangle$:
- $S$ :状态集合(FrozenLake 有 16 个格子)。
- $A$ :动作集合(4 个方向)。
- $P(s' \mid s, a)$ :转移概率(滑冰场景下是随机的)。
- $R(s, a, s')$ :奖励。
- $\gamma$ :折扣因子。

### 1.1 状态价值函数 $V^\pi(s)$
在策略 $\pi$ 下,从状态 $s$ 出发能拿到的期望折扣回报:
$$
V^\pi(s) = \sum_a \pi(a\mid s) \sum_{s'} P(s'\mid s, a) \big[ R(s,a,s') + \gamma V^\pi(s') \big]
$$

### 1.2 动作价值函数 $Q^\pi(s, a)$
$$
Q^\pi(s, a) = \sum_{s'} P(s'\mid s, a) \big[ R(s,a,s') + \gamma \sum_{a'} \pi(a'\mid s') Q^\pi(s', a') \big]
$$

### 1.3 贝尔曼最优方程
$$
V^*(s) = \max_a \sum_{s'} P(s'\mid s, a) \big[ R + \gamma V^*(s') \big]
$$
$$
Q^*(s, a) = \sum_{s'} P(s'\mid s, a) \big[ R + \gamma \max_{a'} Q^*(s', a') \big]
$$
**值迭代就是反复套这两个公式,直到表格收敛。**

---

## 2. 一个重要事实:我们其实不知道 $P$ 和 $R$

公式很优雅,但 FrozenLake 的源码里并没有显式给出 $P(s'\mid s, a)$ 矩阵。怎么办?

**解决思路**:**用采样估计** —— 让 Agent 在环境里乱走,把 (s, a, s', r) 四元组存起来;用经验频率去近似真实概率:
$$
P(s'\mid s, a) \approx \frac{N(s, a, s')}{N(s, a)}
$$
代码里用 `collections.Counter` 实现:
```python
self.transits[(state, action)][new_state] += 1
self.rewards[(state, action, new_state)] = reward
```
- `transits[(s,a)]` 是一个 `Counter`,键是 s',值是访问次数。
- `rewards[(s,a,s')]` 是该转移拿到的奖励。
- 注意:**FloydLake 的奖励只跟 (s, a, s') 有关,所以用三元组存就够了**。

---

## 3. 共同组件:`Agent` 类

两个文件共用完全相同的脚手架(只有 `value_iteration` 不同):
```python
class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state, _ = self.env.reset()
        self.rewards  = collections.defaultdict(float)            # (s,a,s') → r
        self.transits = collections.defaultdict(collections.Counter)  # (s,a) → Counter(s')
        self.values   = collections.defaultdict(float)            # 存 V 或 Q
```

### 3.1 `play_n_random_steps(count)` —— 探索阶段
```python
def play_n_random_steps(self, count):
    for _ in range(count):
        action = self.env.action_space.sample()         # 随机选动作
        new_state, reward, terminated, _, _ = self.env.step(action)
        self.rewards[(self.state, action, new_state)] = reward
        self.transits[(self.state, action)][new_state] += 1
        self.state = self.env.reset()[0] if terminated else new_state
```
作用:
- 用**纯随机策略**与环境交互 `count` 步,填充 `rewards` / `transits` 表格。
- 这里 100 步只是**采样建模**,不是用来学策略的——策略在 `value_iteration()` 里学。

### 3.2 `play_episode(env)` —— 测试阶段
把 `test_env` 传进来,跑完一整局,用**当前最优策略**选动作(由 `select_action` 实现),累加奖励。这只用来评估,不更新表格。

---

## 4. `01_frozenlake_v_iteration.py`:价值迭代

### 4.1 `calc_action_value(state, action)`
```python
def calc_action_value(self, state, action):
    target_counts = self.transits[(state, action)]
    total = sum(target_counts.values())
    action_value = 0.0
    for tgt_state, count in target_counts.items():
        reward = self.rewards[(state, action, tgt_state)]
        val = reward + GAMMA * self.values[tgt_state]   # 套贝尔曼方程
        action_value += (count / total) * val
    return action_value
```
这就是用经验频率替换 $P$ 的贝尔曼方程:
$$
Q(s,a) \approx \sum_{s'} \frac{N(s,a,s')}{N(s,a)} \big[ R + \gamma V(s') \big]
$$

### 4.2 `select_action(state)`
```python
def select_action(self, state):
    best_action, best_value = None, None
    for action in range(self.env.action_space.n):
        action_value = self.calc_action_value(state, action)
        if best_value is None or best_value < action_value:
            best_value, best_action = action_value, action
    return best_action
```
即:
$$
\pi(s) = \arg\max_a Q(s,a) = \arg\max_a \sum_{s'} P(s'\mid s,a)[R + \gamma V(s')]
$$

### 4.3 `value_iteration()` —— **V 表格的更新**
```python
def value_iteration(self):
    for state in range(self.env.observation_space.n):
        state_values = [
            self.calc_action_value(state, action)
            for action in range(self.env.action_space.n)
        ]
        self.values[state] = max(state_values)
```
**完整对应贝尔曼最优方程**:
$$
V(s) \leftarrow \max_a \sum_{s'} P(s'\mid s,a) [R + \gamma V(s')]
$$
所以这里 `self.values` 存的是 **V 表**,键是 state,值是 float。

### 4.4 主循环节奏
```python
while True:
    iter_no += 1
    agent.play_n_random_steps(100)    # 1) 补一次采样
    agent.value_iteration()           # 2) 扫一遍 V 表做一次更新
    reward = 评估 TEST_EPISODES(20) 局平均奖励
    if reward > 0.10:                 # FrozenLake 走通有奖励
        break
```
阈值 0.10 很低,因为 4×4 FrozenLake **滑冰**时随机成功也有一定概率。

### 4.5 整体流程
```
┌──────────────┐
│  随机探索 100步 │ ──→ 更新 (rewards, transits) 模型表
└──────────────┘
        ↓
┌──────────────────┐
│ 价值迭代:扫 V 表 │ ──→ 新 V(s) = max_a Σ P (R + γV(s'))
└──────────────────┘
        ↓
┌──────────────────┐
│ 用 π(s)=argmax_a Q 跑 20 局评估 │ ──→ 看是否解了
└──────────────────┘
        ↑_________________________________|
```

---

## 5. `02_frozenlake_q_iteration.py`:Q 价值迭代

### 5.1 思路转变
V 迭代要存 `|S|` 个数;Q 迭代要存 `|S| × |A|` 个数。**看起来更费空间**,但有一个**重要便利**:策略选择不再需要扫一遍动作:
$$
\pi(s) = \arg\max_a Q(s,a)
$$
而 V 迭代每次 select_action 都要再算一次 Q。Q 迭代把这一步**预计算**到了表里。

### 5.2 `value_iteration()` —— **Q 表的更新**
```python
def value_iteration(self):
    for state in range(self.env.observation_space.n):
        for action in range(self.env.action_space.n):
            action_value = 0.0
            target_counts = self.transits[(state, action)]
            total = sum(target_counts.values())
            for tgt_state, count in target_counts.items():
                key = (state, action, tgt_state)
                reward = self.rewards[key]
                best_action = self.select_action(tgt_state)   # 注意:内层也要 argmax
                val = reward + GAMMA * self.values[(tgt_state, best_action)]
                action_value += (count / total) * val
            self.values[(state, action)] = action_value
```
**对照贝尔曼最优方程的 Q 版本**:
$$
Q(s, a) \leftarrow \sum_{s'} P(s'\mid s,a) \big[ R + \gamma \max_{a'} Q(s', a') \big]
$$
注意代码里这一行:
```python
best_action = self.select_action(tgt_state)
```
等价于 $\arg\max_{a'} Q(s', a')$,因为 `select_action` 就是扫一遍 `self.values[(s', a')]` 取最大。

### 5.3 `select_action(state)` 变得更轻
```python
def select_action(self, state):
    best_action, best_value = None, None
    for action in range(self.env.action_space.n):
        action_value = self.values[(state, action)]     # 直接查表
        ...
```
不再需要调用 `calc_action_value`,直接读 `self.values[(state, action)]` 即可。

### 5.4 主循环完全一样
阈值变为 `0.80`,因为这个写法在 FrozenLake 上稳定收敛。

---

## 6. V-Iteration vs Q-Iteration:逐项对比

| 维度 | V-Iteration (01) | Q-Iteration (02) |
| ---- | ---------------- | ---------------- |
| 表格大小 | $\|S\|$ | $\|S\| \times \|A\|$ |
| 公式 | $V(s) \leftarrow \max_a \sum_{s'} P[R + \gamma V(s')]$ | $Q(s,a) \leftarrow \sum_{s'} P[R + \gamma \max_{a'} Q(s',a')]$ |
| 选动作 | 需要重新算 $\arg\max_a \sum_{s'} P [R + \gamma V(s')]$ | 直接查 $Q(s,a)$ |
| 评估 | 阈 0.10 | 阈 0.80 |
| 收敛速度 | 较慢(评估阈值低) | 较快(评估阈值高) |

注意:**两个例子都用了相同的"环境采样建模 + 表格更新"框架**,真正的算法差异只在 `value_iteration` 方法的 5~10 行。这就是为什么把它们放成对照实验。

---

## 7. 为什么这一章重要?通向 Model-Free RL 的桥梁

**先给一个明确的答案:Ch05 的 Agent 表格操作就是动态规划 (Dynamic Programming, DP)。** 不是"像" DP,而是**严格意义上的 DP**——只是做了一点小改动(用采样估计 $P$,而经典 DP 要求 $P$ 已知)。

这两个算法(`01` 的 Value Iteration 和 `02` 的 Q-Value Iteration)都是 **Model-Based**(因为我们显式估计了 $P$ 和 $R$),都属于 **Dynamic Programming (DP)**。它们要求:
- 状态空间**小到能装进表格**;
- 环境**能反复采样建表**(用来近似 $P$)。

### 7.1 为什么说 Ch05 就是 DP?

经典 DP 算法(以 Value Iteration 为例)的伪代码是这样的(Sutton & Barto, Reinforcement Learning: An Introduction, Ch4):

```
Algorithm: Value Iteration
Input: MDP <S, A, P, R, γ>, small threshold θ > 0
Initialize V(s) arbitrarily for all s ∈ S
repeat
    Δ ← 0
    for each s ∈ S:
        v ← V(s)
        V(s) ← max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
        Δ ← max(Δ, |v - V(s)|)
until Δ < θ
return V (or extract π)
```

对照 Ch05 的 `01_frozenlake_v_iteration.py`:

```python
def value_iteration(self):
    for state in range(self.env.observation_space.n):
        state_values = [
            self.calc_action_value(state, action)   # Σ P(R + γV(s'))
            for action in range(self.env.action_space.n)
        ]
        self.values[state] = max(state_values)       # V(s) ← max_a Q(s,a)
```

**两者完全是同一段算法**——唯一差别是:经典 DP 用**已知的** $P$ 矩阵做加权和;Ch05 用**采样估计的** `transits[(s,a)]` 计数器去近似 $P$。所以 Sutton & Barto 教材 Ch4 的所有结论(收敛性、最优性、$O(|S|^2 |A|)$ 的单步复杂度)在 Ch05 里**一字不差地成立**。

### 7.2 "Model-Based" 在 Ch05 里到底体现在哪?

| 经典 DP | Ch05 |
| ------- | ---- |
| 需要预先知道 $P(s'\mid s,a)$ | 用 `play_n_random_steps` 采 100 步,把 (s,a,s') 计数存进 `self.transits`,再用频率近似 $P$ |
| 需要预先知道 $R(s,a,s')$ | 用 `self.rewards[(s,a,s')]` 直接存最近一次观测到的奖励 |
| 跑算法前 $P$、$R$ 都已就绪 | 跑算法前先做随机探索填充 `rewards` / `transits` 两表 |

**Ch05 实际上是在线 (online) 估计 + 离线 (offline) 解 DP**:每轮先采样更新 $P$、$R$ 经验表,再对这张经验表做一次 Value Iteration / Q-Iteration 的扫表更新。**采样估计**这一步是 Ch05 比 Sutton & Barto 教材多出来的"工程化补丁",但算法骨架完全是 DP。

### 7.3 Ch05 → Ch06+ 的演进方向

Ch05 假设 $|S|$ 很小(能装表)、环境能反复采样(能估计 $P$)。现实中这两个条件常常不满足(FrozenLake 8×8 还行,Atari 就不行了)。**Model-Free** 方法(Q-learning、SARSA、Policy Gradient、Actor-Critic、PPO、SAC)会**直接绕过对 $P$ 的估计**,改成在交互中**直接**用梯度下降更新 Q 表 / Q 网络 / 策略网络——但**更新的目标仍然是贝尔曼最优方程**,只是把"扫表 + 改数"换成了"前向 + 反向 + 改权重"。所以 Ch05 学到的"贝尔曼方程到底在算什么"是后面所有高级算法的共同地基。

> 一句话:**这一章让你彻底搞懂"贝尔曼方程到底在算什么",Ch06+ 是把这一章的 V/Q 表换成神经网络,Ch07+ 是把对 $P$ 的显式估计也扔掉**——但根上都是同一个东西。

---

## 8. Chapter 05 速记口诀

- **V-迭代** → "**我有 V 表,选动作要现场算 Q**"
- **Q-迭代** → "**我有 Q 表,选动作查表就行,但表格大**"
- **共同前置** → "**先用 100 步随机走,建一张 (s,a,s') 计数表**"

---

## 9. 一图回顾 Ch04 → Ch05 演进

```
Ch04: 策略搜索 (Policy Search)
  ├─ 神经网络当策略
  ├─ 用"好轨迹"做监督学习
  └─ 不显式估计 V 或 Q

Ch05: 表格型 DP (Value / Q-Iteration)
  ├─ 字典当表格
  ├─ 用采样估计 P,R,再解贝尔曼方程
  └─ 不需要神经网络

下一章将进入:用神经网络拟合 V 或 Q → DQN 系列
```

---

## 10. Agent 的具体构造:它到底是个什么结构?

这是承上启下的关键一节——Ch04 的 Agent 是一个**两层的、近线性的小 MLP**(总共 898 ~ 2560 个参数,详见 Ch04 README §11),而 Ch05 的 Agent **完全不是神经网络**,它是**三张 Python 字典**。这里把 Ch05 的 Agent 内部数据结构和"学习机制"完整拆开。

### 10.1 一句话先说结论

> **Ch05 的 Agent 没有 `nn.Module`、没有 `forward()`、没有 `loss.backward()`、没有 optimizer。它只靠三个 `defaultdict` 存数,学习就是反复"读字典 + 改字典"**——典型的**tabular model-based dynamic programming**。

如果你带着 Ch04 的"神经网络"印象来读 Ch05 的代码,会一时找不到 `net = Net(...)` 或 `optimizer = ...` 这类语句,这是正常的——**它们根本不存在**。

### 10.2 三个 defaultdict:Agent 的全部"参数"

```python
class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state, _ = self.env.reset()

        # 三张表,这就是 Agent 全部的"状态"
        self.rewards  = collections.defaultdict(float)               # 表 R
        self.transits = collections.defaultdict(collections.Counter) # 表 P 的经验计数
        self.values   = collections.defaultdict(float)               # 表 V 或表 Q
```

| 字段 | 类型 | 键 | 值 | 含义 |
| ---- | ---- | -- | -- | ---- |
| `rewards` | `defaultdict(float)` | `(state, action, new_state)` | `float` | 经验奖励 R(s,a,s') |
| `transits` | `defaultdict(Counter)` | `(state, action)` | `Counter(new_state → count)` | 经验转移频数 N(s,a,s') |
| `values` | `defaultdict(float)` | `state` (V 迭代) 或 `(state, action)` (Q 迭代) | `float` | V 表 或 Q 表 |

**对应到 MDP 五元组** $\langle S, A, P, R, \gamma \rangle$:

| MDP 组件 | 存在形式 | 大小 |
| -------- | -------- | ---- |
| $S$ | 隐含在 `transits` 的 key 里 | 16(FrozenLake 4×4) |
| $A$ | `range(self.env.action_space.n)` | 4 |
| $P(s'\mid s,a)$ | 用 `transits[(s,a)][s'] / sum(transits[(s,a)].values())` 现场算 | — |
| $R(s,a,s')$ | 直接查 `rewards[(s,a,s')]` | — |
| $\gamma$ | 全局常量 `GAMMA = 0.9` | — |

### 10.3 "参数"总数有多少?

这是和 Ch04 最直观的对比——**Ch05 的 Agent 没有可训练参数,所谓"参数"就是三张表的项数**:

| 文件 | 表格 | 键空间 | 上限项数 |
| ---- | ---- | ------ | -------- |
| `01_frozenlake_v_iteration.py` | V | $S$ | 16 |
| `01_frozenlake_v_iteration.py` | R、P | $(S, A, S')$ 和 $(S, A)$ | $16 \times 4 \times 16 = 1024$ |
| `02_frozenlake_q_iteration.py` | Q | $S \times A$ | 64 |

而 Ch04 的 MLP Agent 在 FrozenLake 上是 **2560 个 float 权重 + 258 个偏置**。换句话说:

> **Ch05 Agent 的"模型容量"就是表格的格子数,而不是连续的实数权重。**

这就是 Ch05 标题里"**表格型 (Tabular)**"这个词的全部含义——所有知识都离散地存储在字典的 key→value 里。

### 10.4 "学习"是怎么发生的?——不是梯度下降,是"扫表 + 改数"

Ch04 的训练核心:
```python
loss = CrossEntropyLoss(net(obs), acts)
loss.backward()
optimizer.step()
```

Ch05 的训练核心(V 迭代):
```python
def value_iteration(self):
    for state in range(self.env.observation_space.n):  # 扫一遍 S
        state_values = [
            self.calc_action_value(state, action)     # 算 Q(s,a)
            for action in range(self.env.action_space.n)
        ]
        self.values[state] = max(state_values)         # 写回 V(s)
```

**对照 Ch04**:
- 没有 `loss`,没有 `backward()`,没有 `Adam`,没有 mini-batch。
- 每次"学习"是**对所有 state 顺序执行一次贝尔曼最优方程的更新**。
- 更新规则是**直接赋值**:`self.values[state] = max(...)`,不是"在现有值上做小步梯度下降"。

Q 迭代的 `value_iteration` 同理,只是外层多套了一层对 action 的循环,内层再多一个 `max_{a'}`。

### 10.5 `Agent` 类的方法清单

| 方法 | 作用 | 涉及表格 |
| ---- | ---- | -------- |
| `play_n_random_steps(count)` | 用随机策略采样 `count` 步,**只写表** | R、P |
| `play_episode(env)` | 用当前最优策略跑一整局做评估,**顺便继续写表** | R、P |
| `calc_action_value(state, action)` | 用经验频数算 $Q(s,a) = \sum_{s'} P(s'\mid s,a)[R + \gamma V(s')]$ | R、P、V |
| `select_action(state)` | V 迭代:实时算 Q 后取 argmax;Q 迭代:直接查表取 argmax | V 或 Q |
| `value_iteration()` | 扫表做一次贝尔曼最优方程更新 | V 或 Q |

**没有的方法**:`fit()` / `train()` / `update()` / `backward()` / `step()` —— 这些是 Ch06+ DQN 才用 PyTorch 写出来的东西。

### 10.6 Ch04 Agent vs Ch05 Agent:逐项对比

| 维度 | Ch04 Agent | Ch05 Agent |
| ---- | ---------- | ---------- |
| 是不是神经网络 | 是 (`nn.Module`) | **否**(纯 Python 字典) |
| 状态存储 | `nn.Linear` 的 `weight` / `bias` 张量 | `defaultdict`(3 张表) |
| 参数量(CartPole / FrozenLake) | 898 / 2560 | **0**(没有可训练参数) |
| 学习方式 | 监督学习 + 梯度下降(CrossEntropyLoss + Adam) | 贝尔曼方程扫表更新(直接赋值) |
| 前向传播 | `net(obs)` 一次前向 | `calc_action_value(state, action)` 一次查表+加权求和 |
| 选动作 | `np.random.choice(..., p=Softmax(logits))` 概率采样 | `argmax_a Q(s,a)` 贪心 |
| 依赖 | PyTorch | 只有 Python 标准库 `collections` |
| 适用规模 | 状态可以是连续高维(只要网络容量够) | 状态必须小到能装进表格($|S|$ 通常 < 几千) |
| 类别 | Policy-Based(Model-Free) | Value-Based + Model-Based(DP) |

### 10.7 直观总结

> **Ch05 的 Agent = 三张 Python 字典 + 一段循环**:
> - 存储:`rewards / transits / values` 三个 `defaultdict`。
> - 学习:每轮跑一次 `value_iteration()`,把贝尔曼最优方程"扫一遍",把 V/Q 表原地覆盖。
> - 选动作:对当前 state 做 `argmax_a Q(s,a)`(Q 迭代)或先算再取 max(V 迭代)。
>
> 它是**Ch04 与 Ch06 之间的过渡形态**——既没有 Ch04 的"神经网络当模仿器",也没有 Ch06 的"神经网络拟合 Q 表"。它用最朴素的"表 + 公式"展示了**强化学习的数学骨架**。Ch06+ 才是把这里的 V/Q 表换成神经网络(也就是 DQN),用梯度下降去做同样的贝尔曼更新。

---

## 11. Agent 表格 ↔ 经典动态规划 (DP) 的精确对应

§7 给了结论:**Ch05 = DP**。这一节把对应关系**逐方法 / 逐公式**展开,让你一眼能看出 Ch05 的每一行 Python 代码对应 Sutton & Barto 教材里的哪一个 DP 算法组件。

### 11.1 三张表 ↔ DP 里的三个对象

| Ch05 Agent 字段 | 类型 / 大小 | 经典 DP 对应物 | Sutton & Barto 符号 |
| --------------- | ----------- | -------------- | ------------------- |
| `self.values` | `defaultdict(float)`(V 迭代) | 状态价值表 $V(s)$ | $V: \mathcal{S} \to \mathbb{R}$ |
| `self.values` | `defaultdict(float)`(Q 迭代) | 动作价值表 $Q(s,a)$ | $Q: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ |
| `self.transits` | `defaultdict(Counter)` | 经验估计的转移概率 $P(s'\mid s,a)$ | $\hat P$ |
| `self.rewards` | `defaultdict(float)` | 经验估计的奖励 $R(s,a,s')$ | $\hat R$ |
| `GAMMA` | 全局常量 | 折扣因子 | $\gamma$ |

**关键对应**:
- `self.values[state]` ↔ DP 的 $V(s)$ 表
- `self.values[(state, action)]` ↔ DP 的 $Q(s,a)$ 表
- `self.transits[(s,a)][s'] / total` ↔ DP 的 $P(s'\mid s,a)$ 的**经验估计**
- `self.rewards[(s,a,s')]` ↔ DP 的 $R(s,a,s')$

所以 Ch05 的 Agent 本身就是一个**MDP 五元组 $\langle S, A, P, R, \gamma \rangle$ 的容器**,只是 $P$、$R$ 不是从环境文档里读出来的,而是从采样中**数出来**的。

### 11.2 `calc_action_value` ↔ Bellman 期望方程的 Q 版本

`01_frozenlake_v_iteration.py` 里的:
```python
def calc_action_value(self, state, action):
    target_counts = self.transits[(state, action)]
    total = sum(target_counts.values())
    action_value = 0.0
    for tgt_state, count in target_counts.items():
        reward = self.rewards[(state, action, tgt_state)]
        val = reward + GAMMA * self.values[tgt_state]
        action_value += (count / total) * val
    return action_value
```

逐行翻译成 DP 公式:
| 代码行 | DP 公式 |
| ------ | ------- |
| `target_counts = self.transits[(s,a)]` | 取 $P$ 的第 $(s,a)$ 行 |
| `total = sum(target_counts.values())` | $\sum_{s'} N(s,a,s') = N(s,a)$ |
| `for tgt_state, count in target_counts.items():` | 对所有 $s'$ 求和 |
| `count / total` | $\frac{N(s,a,s')}{N(s,a)} = \hat P(s'\mid s,a)$ |
| `reward + GAMMA * self.values[tgt_state]` | $R + \gamma V(s')$ |
| `action_value += ...` | $Q(s,a) = \sum_{s'} \hat P(s'\mid s,a) [R + \gamma V(s')]$ |

**这正是贝尔曼期望方程的 Q 版本**(只不过 $P$ 换成了 $\hat P$):
$$
Q(s,a) \xleftarrow{\text{DP update}} \sum_{s'} P(s'\mid s,a)\big[R(s,a,s') + \gamma V(s')\big]
$$

### 11.3 `value_iteration` ↔ 经典 Value Iteration

```python
def value_iteration(self):
    for state in range(self.env.observation_space.n):
        state_values = [
            self.calc_action_value(state, action)
            for action in range(self.env.action_space.n)
        ]
        self.values[state] = max(state_values)   # ← 关键一行
```

| 代码段 | 经典 Value Iteration (Sutton & Barto §4.4) |
| ------ | ------------------------------------------ |
| `for state in range(...)` | `for each s ∈ S` |
| `[self.calc_action_value(state, a) for a in ...]` | 算 $Q(s,a) = \sum_{s'} P[R + \gamma V(s')]$(对每个 $a$) |
| `self.values[state] = max(state_values)` | $V(s) \leftarrow \max_a Q(s,a)$(贝尔曼最优方程) |
| 整个方法被外层 `while True: ... break if reward > 0.10` 包住 | `repeat until Δ < θ` |

**所以 Ch05 的 `01` 就是**逐行**的 Value Iteration**,只是收敛判据从 $\|V_{k+1} - V_k\|_\infty < \theta$ 改成了"评估 ≥ 20 局的平均奖励 > 阈值"——后者更工程化,但数学性质一致(都保证最终 $V \to V^*$)。

### 11.4 `02` 的 `value_iteration` ↔ 经典 Q-Value Iteration

```python
def value_iteration(self):
    for state in range(self.env.observation_space.n):
        for action in range(self.env.action_space.n):
            action_value = 0.0
            target_counts = self.transits[(state, action)]
            total = sum(target_counts.values())
            for tgt_state, count in target_counts.items():
                key = (state, action, tgt_state)
                reward = self.rewards[key]
                best_action = self.select_action(tgt_state)   # argmax
                val = reward + GAMMA * self.values[(tgt_state, best_action)]
                action_value += (count / total) * val
            self.values[(state, action)] = action_value
```

对应 Sutton & Barto §6.5 的 Q-Value Iteration 公式:
$$
Q(s,a) \leftarrow \sum_{s'} P(s'\mid s,a) \big[ R + \gamma \max_{a'} Q(s',a') \big]
$$

| 代码行 | 公式 |
| ------ | ---- |
| `best_action = self.select_action(tgt_state)` | $a^* = \arg\max_{a'} Q(s', a')$ |
| `val = reward + GAMMA * self.values[(tgt_state, best_action)]` | $R + \gamma \max_{a'} Q(s', a')$ |
| `action_value += (count / total) * val` | $\sum_{s'} \hat P(s'\mid s,a) [\cdot]$ |
| `self.values[(state, action)] = action_value` | $Q(s,a) \leftarrow$ 新值 |

**这正是**逐行**的 Q-Value Iteration**,只是把 $\arg\max$ 实现成"扫一遍 4 个 action 取最大"——和教材伪代码完全等价。

### 11.5 经典 DP 全家桶 ↔ Ch05 的实现情况

Sutton & Barto 教材 Ch4 介绍的 DP 算法主要有三个,Ch05 覆盖了其中两个,**没**有覆盖的一个值得在这里说清楚:

| 经典 DP 算法 | 是否在 Ch05 | Sutton & Barto 章节 | 算法核心 |
| ------------ | ----------- | ------------------- | -------- |
| **Policy Evaluation**(策略评估) | ❌(隐式) | §4.1 | 给定 $\pi$,反复迭代 $V^\pi(s) = \sum_a \pi(a\mid s) \sum_{s'} P(s'\mid s,a)[R + \gamma V^\pi(s')]$,直到 $V$ 收敛 |
| **Policy Iteration**(策略迭代) | ❌(未实现) | §4.3 | 反复做"Policy Evaluation → Greedy Policy Improvement"两步,直到 $\pi$ 不再变 |
| **Value Iteration**(价值迭代) | ✅ `01_frozenlake_v_iteration.py` | §4.4 | 把 Policy Evaluation 截断成"只跑一次扫表更新",再做贪心改进 |
| **Q-Value Iteration** | ✅ `02_frozenlake_q_iteration.py` | §6.5 | 同上,但更新对象是 $Q(s,a)$ 而非 $V(s)$ |

> **Policy Iteration 在 Ch05 没出现,但概念上极接近 Value Iteration**:Value Iteration 是"评估 + 改进压缩成一步",Policy Iteration 是"评估到收敛再改进"。你可以把它理解为 Value Iteration 的"慢但更稳"的兄弟版本。如果想自己写一个,在 Ch05 基础上加一个 `self.policy = defaultdict(lambda: 0)` 就行——但 FrozenLake 这种小环境里,Value Iteration 已经够快,所以教材代码直接跳过了 Policy Iteration。

### 11.6 Ch05 与经典 DP 的关键差异:三件事

虽然 Ch05 的算法骨架就是 DP,但与教材的"理想化 DP"相比,**做了三处工程化改动**:

| 维度 | 经典 DP(Sutton & Barto) | Ch05 |
| ---- | ---------------------- | ---- |
| $P(s'\mid s,a)$ | **已知**(MDP 五元组的一部分) | **未知**——用 `transits` 计数估计 $\hat P$ |
| $R(s,a,s')$ | **已知** | **未知**——用 `rewards` 存最近一次观测 |
| 收敛判据 | $\|V_{k+1} - V_k\|_\infty < \theta$ | 评估 20 局的平均奖励 > 阈值 |

**这三处改动的后果**:
- 优势:不再依赖"环境模型已知"这个强假设,可以在 gym 这种只暴露 `step()` 接口的环境上跑。
- 代价:$\hat P$ 是不准的(有限样本),所以 $V$ 表的收敛会"抖动",不会像理论 DP 那样单调下降 $\|V - V^*\|$。这就是 Ch05 为什么用"评估奖励 > 0.10/0.80"而不是"$\|V\|$ 变化 < $\theta$"作判据——前者对 $\hat P$ 的噪声更鲁棒。

### 11.7 一图把"Ch05 Agent"翻译成"DP 标准语言"

```
┌─────────────────────────────────────────────────────────────────┐
│            Ch05 Agent(实际代码)               DP 标准语言             │
├─────────────────────────────────────────────────────────────────┤
│ self.values[state]                    ↔  V(s)                   │
│ self.values[(state, action)]          ↔  Q(s,a)                 │
│ self.transits[(s,a)][s']/total        ↔  P̂(s' | s, a)            │
│ self.rewards[(s,a,s')]                ↔  R̂(s, a, s')             │
│ self.calc_action_value(s, a)          ↔  Q(s,a) = Σ P̂(R̂ + γV)  │
│ self.values[s] = max_a Q(s,a)         ↔  V(s) ← max_a Q(s,a)    │
│ self.select_action(s)                 ↔  π(s) = argmax_a Q(s,a) │
│ self.value_iteration() 整体           ↔  Value / Q-Value Iteration│
│ self.transits / self.rewards 的填充    ↔  模型估计 (model learning)│
│ 外层 while 循环                         ↔  repeat until converge  │
└─────────────────────────────────────────────────────────────────┘
```

> 读到这里,你可以把 Ch05 的 `01`/`02` 直接当作 Sutton & Barto 教材 Ch4 的 Python 翻译来读——它们是**完全同一个东西**。Ch06 的 DQN 才会把"扫表 + 改数"换成"前向 + 反向 + 改权重",让 DP 能扩展到 Atari 这种大状态空间上。
