# Chapter 04 详解:交叉熵方法(Cross-Entropy Method, CEM)

> 本章是《Deep Reinforcement Learning Hands-On》第二版的第四章,核心主题是**策略梯度家族的入门代表:交叉熵方法**。它不依赖神经网络估计 Q 值或 V 值,而是直接把策略参数化为"动作的概率分布",通过"筛选精英经验 + 监督学习"反复训练。

---

## 1. 强化学习基本概念速览

| 符号 | 含义 |
| ---- | ---- |
| $s$ / state | 环境当前状态 |
| $a$ / action | Agent 选择的行为 |
| $r$ / reward | 环境给出的即时奖励 |
| $\pi(a \mid s)$ | 策略:在状态 s 下选 a 的概率 |
| Episode | 从 reset 到 terminated/truncated 的一局完整轨迹 |
| Return $G$ | 一条轨迹的累计奖励,本章最朴素的就是直接对 reward 求和 |

强化学习的目标:学到 $\pi^*$ 使期望回报最大化。

---

## 2. 交叉熵方法的核心思想

CEM 是一种**基于采样的策略搜索**方法,完全不要价值函数。它的工作循环只有两步:

1. **采样 (Play)**:用当前策略 $\pi_\theta$ 在环境中跑 N 个 episode,记录 (obs, action, reward)。
2. **筛选 + 拟合 (Learn)**:把奖励高的(称为 **elite** )轨迹挑出来,把它们当成"专家数据",用监督学习(交叉熵损失)更新网络 $\theta$,让网络在这些 (obs, action) 上输出的动作概率更高。

伪代码:
```
初始化策略网络 π_θ
repeat:
    用 π_θ 玩 batch_size 条 episode
    选奖励 top-percentile% 的作为 elite
    用 elite 轨迹中的 (obs, action) 做监督训练
until 平均奖励达标
```

直观理解:**让"赢的轨迹"里出现过的动作,以后在同样状态下被选中的概率变大**。这就是为什么损失函数可以直接用 `CrossEntropyLoss`:本质上把 RL 问题降级成了一个**模仿学习**(imitation learning)问题。

---

## 3. 共同组件:Episode / EpisodeStep

四个例子都用了同一种数据组织方式:
```python
Episode = namedtuple("Episode", field_names=["reward", "steps"])
EpisodeStep = namedtuple("EpisodeStep", field_names=["observation", "action"])
```
- `Episode` 装一整局。
- `EpisodeStep` 装一局的某一步。
- `iterate_batches` 是一个**生成器**,源源不断地吐出 batch。

---

## 4. `01_cartpole.py`:最经典的 CEM 模板

### 4.1 环境
- **CartPole-v1**:小车推杆子不倒。`obs_size=4` (位置、速度、角度、角速度),`n_actions=2` (左/右)。
- `render_mode="rgb_array"` + `RecordVideo` 用来在 `mon` 目录里录制 mp4。

### 4.2 网络结构
```python
Linear(obs_size, 128) → ReLU → Linear(128, n_actions)
```
输出是每个动作的 **logits** (不是概率),外面再套 `Softmax` 得到采样概率。

### 4.3 关键超参数
| 参数 | 值 | 作用 |
| ---- | -- | ---- |
| HIDDEN_SIZE | 128 | 隐层宽度 |
| BATCH_SIZE | 16 | 每轮跑多少局 |
| PERCENTILE | 70 | 选奖励前 30% 作为 elite |
| 学习率 | 0.01 | Adam |
| 收敛条件 | reward_mean > 199 | CartPole-v1 单局满分 200 |

### 4.4 训练循环要点
```python
action = np.random.choice(len(act_probs), p=act_probs)
```
**按概率采样**,而不是 `argmax`,这就是"探索 (exploration)"。如果换成贪心,CartPole 永远学不会。

`filter_batch`:
- `reward_bound = np.percentile(rewards, 70)` = 第 70 百分位的奖励值。
- 把所有 `reward < reward_bound` 的局丢弃,只保留 top 30%。
- 剩下的所有 (obs, action) 拼成训练集。

### 4.5 为什么 CartPole 适合 CEM?
- 奖励**稠密**:每多撑一步都得 +1,信号密集。
- 状态**连续且低维**,神经网络好拟合。
- 决策**反应式**:当下往哪推和之前动作几乎无关,适合"哪个动作让杆子不倒就重复该动作"这种局部策略。

---

## 5. `02_frozenlake_naive.py`:为什么同样的代码直接搬到 FrozenLake 就崩了

### 5.1 环境
**FrozenLake-v1**:4×4 冰面格(S, F, F, F, ... , G),Agent 要从 S 走到 G,掉洞 H 失败。
- 状态:16 个离散格子 → 整数 0~15。
- 动作:4 个(左/下/右/上)。
- 奖励:0/1。**只有走到 G 才得 1,其它所有步都是 0**。
- 冰面默认 `is_slippery=True`,Agent 走的不是它想去的方向。

### 5.2 必须做的预处理
```python
class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        res = np.copy(self.observation_space.low)  # 全 0
        res[observation] = 1.0                      # 对应位设为 1
        return res
```
因为 FrozenLake 输出 `Discrete(16)`(一个整数),而后面的 Net 是 `nn.Linear`,要求浮点向量。所以把整数 7 转成 `[0,0,0,0,0,0,0,1,0,...]`。

### 5.3 朴素 CEM 失败的根本原因
直接套用 01 的代码,你会发现**根本解不出来**。原因是双重:

1. **奖励极度稀疏**:99% 的局是失败(奖励 0),只有 1% 走到终点(奖励 1)。`np.percentile(rewards, 70)` 算出来的 `reward_bound` 几乎永远是 0,**所有局都被当成 elite**,等价于"用纯随机数据训练"。
2. **滑冰 + 长链路**:就算偶尔成功一次,中间某一步选了错误的方向,后面的轨迹和"成功策略"几乎无关,噪声极大。

`filter_batch` 里这一行可以看清问题:
```python
for example in batch:
    if example.reward < reward_bound:   # 99% 的局 reward=0
        continue
```
当 `reward_bound=0` 时,**所有 0 奖励的局都被跳过**,**所有 1 奖励的局**都被保留——但 FrozenLake 1 奖励出现的频率极低,常常 batch 里一个 1 都没有 → `train_obs=[]`,网络啥也没学到。

### 5.4 一句话总结
> 01 的 CEM 隐含假设"奖励稠密"。一旦奖励稀疏,直接套用就会失败。

---

## 6. `03_frozenlake_tweaked.py`:两个让 CEM 在稀疏奖励下能用的关键补丁

### 6.1 补丁 A:折扣奖励(Discounted Reward)
```python
filter_fun = lambda s: s.reward * (GAMMA ** len(s.steps))
disc_rewards = list(map(filter_fun, batch))
reward_bound = np.percentile(disc_rewards, PERCENTILE)  # PERCENTILE=30
```
即使最终奖励是 0,但 episode 越长,也意味着 Agent 走得更远、踩过的洞更少,这种"**过程信息**"通过 $\gamma^{len(steps)}$ 体现在折扣奖励里,成为有区分度的训练信号。

### 6.2 补丁 B:精英集跨 batch 累积
```python
full_batch = []                                    # 跨 batch 保存所有 elite
full_batch, obs, acts, reward_bound = filter_batch(
    full_batch + batch, PERCENTILE                  # 旧 elite + 新轨迹 再筛一次
)
...
full_batch = full_batch[-500:]                      # 最多留 500 条
```
单 batch 几乎没有 1 奖励的轨迹 → 没有 elite → 没法训练。所以把所有历史 elite 攒起来一起筛,训练集就不再是空。
- `full_batch = full_batch[-500:]` 是滑动窗口,防止内存爆炸,也能让策略"忘记"过老的经验。

### 6.3 其它改动
- `BATCH_SIZE`:16 → 100。稀疏奖励需要更多样本才有好轨迹。
- `PERCENTILE`:70 → 30。elite 比例提高,样本够用。
- `GAMMA=0.9`、`lr=0.001`。
- `random.seed(12345)` 便于复现。

### 6.4 解决条件
`reward_mean > 0.8`,即平均 80% 的局能走到 G。

---

## 7. `04_frozenlake_nonslippery.py`:简化环境,凸显算法差异

代码几乎与 03 一模一样,**唯一的差别**:
```python
env = frozen_lake.FrozenLakeEnv(render_mode="rgb_array", is_slippery=False)
env.spec = gym.spec("FrozenLake-v1")           # 重新指定 spec,RecordVideo 需要它
env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
```
- `is_slippery=False`:**关掉随机滑冰**,Agent 想走哪个方向就走哪个方向。这把环境从随机过程简化成"完全可预测的格子世界"。
- `TimeLimit(..., 100)`:防止某些策略死循环,超过 100 步就强制结束。

### 7.1 这版有什么用?
- **对照实验**:03 里"折扣奖励 + elite 累积"到底是因为稀疏奖励必要,还是因为滑冰才必要?04 告诉你——即便没有滑冰,**折扣奖励 + elite 累积这两招依然管用**,因为它们解决的是"奖励稀疏"问题。
- 跑得**更快更稳**,适合教学演示。

---

## 8. Chapter 04 总结:一张表看四种配置

| 文件 | 环境 | 奖励特点 | PERCENTILE | BATCH_SIZE | 关键技巧 | 解决条件 |
| ---- | ---- | -------- | ---------- | ---------- | -------- | -------- |
| 01_cartpole | CartPole | 稠密 | 70 | 16 | 朴素 CEM | reward_mean > 199 |
| 02_frozenlake_naive | FrozenLake(slippery) | 极稀疏 | 70 | 16 | 无 | (学不到) |
| 03_frozenlake_tweaked | FrozenLake(slippery) | 极稀疏 | 30 | 100 | 折扣奖励 + 累积 elite | reward_mean > 0.8 |
| 04_frozenlake_nonslippery | FrozenLake(no-slip) | 稀疏 | 30 | 100 | 折扣奖励 + 累积 elite | reward_mean > 0.8 |

---

## 9. CEM 的优点 / 局限(回头看书必备)

**优点**
- 极简:不需要 value function,不 bootstrapping,实现就一个 for 循环。
- 稳定:目标函数 = 监督学习 loss,直接用现成优化器。
- 适合奖励稠密、决策局部的环境(CartPole 那种)。

**局限**
- 奖励稀疏时直接失效(必须靠折扣奖励 + elite 累积这种 hack)。
- 样本效率低,需要丢弃大量轨迹。
- 学到的策略是**短视的**,只模仿精英轨迹的局部动作,不会做"先牺牲后获利"这种长链规划。
- 连续动作空间需要重新设计采样方式(用高斯分布代替 Softmax)。

这正是后续章节引入 Q-learning、Actor-Critic、PPO 的动机。

---

## 10. 常见疑问与延伸思考

读到这里,初学者通常会有几个直觉性疑问,这里集中澄清。

### 10.1 CEM 和遗传算法(GA)什么关系?

**CEM 本质上就是遗传算法在 RL 场景下的特化版本。** 共享同一个"基于采样的随机优化"范式:

| 步骤 | 遗传算法(GA) | 交叉熵方法(CEM) |
| ---- | ------------- | ---------------- |
| 个体 | 一条染色体(参数向量 $\theta$) | 一条 episode(完整轨迹) |
| 种群 | $N$ 条染色体 | $N$ 条 episode |
| 适应度 | fitness 函数 | episode 的累计 reward |
| 选择 | 锦标赛 / 轮盘赌 / 精英保留 | 奖励 top-percentile%(精英) |
| 繁殖 | 交叉(crossover)+ 变异(mutation) | 没有显式交叉,直接对精英做监督学习 |
| 下一代 | 交叉变异后产生新种群 | 用精英数据更新参数 $\theta$,再采样 |

**关键差异**:

1. **没有显式交叉(crossover)**:GA 经典做法是把两条染色体的"基因"拼接;CEM 直接丢弃网络结构,只保留精英经验,然后做监督学习。神经网络的梯度下降就是某种"软交叉 + 软变异"——参数往"高奖励轨迹"的方向连续微调,而不是离散拼接。
2. **精英定义不同**:CEM 用"百分位"而不是"top-K",每轮挑出来的精英比例恒定,跟奖励绝对值无关,所以在奖励尺度变化时更稳。
3. **种群位置不同**:GA 的种群是 $\theta_1, \theta_2, ..., \theta_N$(参数空间);CEM 的种群是 $\tau_1, \tau_2, ..., \tau_N$(轨迹空间)。真正被"迭代"的,仍然是网络参数 $\theta$。

> **一句话**:CEM = GA 思想 × RL 场景的特化版本。Ch13 的进化策略(ES/NES)才是直接对网络权重做进化的算法,与 CEM 同源但路线不同。

### 10.2 神经网络真的"输出奖励"吗?——常见误解澄清

初学者常会以为策略网络既输出动作又输出奖励。**这是错的。** 下面是常见说法的逐项核对:

| 说法 | 实际情况 | 是否正确 |
| ---- | -------- | -------- |
| 有一个小型神经网络 | 策略网络 $\pi_\theta$,输入 obs,输出每个动作的 logits | ✅ |
| **输出对应行为以及奖励** | 网络**只输出动作分布**,**奖励完全由环境给出** | ❌ |
| 把这些作为结果保存起来 | 保存 `(obs, action, reward)` 三元组(整条 episode) | ✅ |
| 选择前面 top-k 的结果 | 代码里实际是**百分位**(top-percentile%),不是固定 K | ⚠️ 近似正确 |
| 作为数据集训练神经网络 | 损失函数 = CrossEntropyLoss,监督学习 `(obs, action)` 对 | ✅ |

**奖励的来源链**:
$$
\text{obs} \xrightarrow{\text{Net}} \text{logits} \xrightarrow{\text{Softmax}} \pi(a\mid s) \xrightarrow{\text{采样}} a \xrightarrow{\text{env.step()}} (s', r, \text{done})
$$
注意 $r$ 来自 `env.step()`,**与网络完全无关**。如果让网络去预测奖励,那就退化成了"世界模型"类方法(World Models / Model-Based RL),不是 CEM。

**Top-k vs Top-percentile**:
- **top-k**:每轮固定选 K 条,跟奖励分布无关。
- **top-percentile**:每轮选固定**比例**,奖励尺度变了也不用调 K。

CEM 用 percentile 是因为**奖励的绝对值会随训练漂移**(比如 FrozenLake 早期偶尔得 1,后期常常得 1),用比例更稳。

**演员比喻**:
> **网络是"演员",环境是"评委"。** 演员只负责"按当前剧本表演",评委负责打分。CEM 每轮做完表演后,只留下"得分高"的演员片段作为下次演出的**范例**,让演员去**模仿**这些范例。

注意是**模仿好动作**,不是**预测分数**。这就是为什么损失函数能直接用现成的 `CrossEntropyLoss`——本质上是个分类问题。

### 10.3 为什么按概率采样而不是取 argmax?——探索与利用的平衡

**这是强化学习的第一公理:探索 (exploration) 与利用 (exploitation) 的平衡。**

实际代码:
```python
sm = nn.Softmax(dim=1)
act_probs_v = sm(net(obs_v))              # 网络输出 → 动作概率
act_probs = act_probs_v.data.numpy()[0]   # 例如 [0.1, 0.2, 0.6, 0.1]
action = np.random.choice(len(act_probs), p=act_probs)  # 按概率采样
```

如果换成 `np.argmax(act_probs)`,就**永远是 2 号动作**。

**反直觉实验**:把 `01_cartpole.py` 中的 `np.random.choice` 改成 `np.argmax`,再跑一次,大概率会观察到:
1. 早期 loss 下降很快(模仿"自认为的好动作"很容易)。
2. 但 `reward_mean` 涨到某个值就**永远卡住**,再也上不去 199。
3. 这就是"陷入局部最优"的典型症状。

**训练时与测试时的策略不同**:

| 阶段 | 选动作方式 | 原因 |
| ---- | ---------- | ---- |
| **训练时**(收集数据) | 采样 | 需要探索,发现更好的动作 |
| **测试时**(评估智能体) | 看你想测什么 | 一般两种都试 |

本书的代码全程都用了采样(偷懒没区分),但 Ch06+ 引入 DQN 时,**"ε-greedy"** 就会是更显式的探索机制,本质是同一件事。

> **一句话记牢**:**训练时:按概率采样(探索)。评估时:取 argmax(贪心利用)。**

这条规则不仅适用于 CEM,也是几乎所有 on-policy / off-policy RL 的共同准则。

### 10.4 什么时候 CEM 这种理解会失效?

在你现在读的 Ch04 里没问题。但请记住:
- **CEM 不会"理解"为什么某些动作好**,只会复制它们的统计模式。
- 所以遇到**长链因果**("先掉进坑里,才能学会绕过去")或**对手策略**(石头剪刀布),CEM 会很挣扎。
- 这是后续 Ch06+ 引入 **Q-learning / DQN / Policy Gradient** 的根本动机——它们能更精确地归因"哪个动作对最终奖励贡献了多少"。

### 10.5 一个亲手验证的好实验

把 `01_cartpole.py` 里的 `PERCENTILE` 从 70 改成 99,看 CartPole 是否还能学得动:
- **PERCENTILE=70**:保留 top 30%,样本够多,学得快。
- **PERCENTILE=99**:几乎只保留最好的 1%,样本极少,训练集塌缩成几条轨迹,网络学不到东西。

这能帮你**亲手感受 CEM 对 percentile 的敏感度**,对"为什么用百分位而不是绝对 K"有更深体会。

---

## 11. Agent 的具体构造:它到底是个什么网络?

这是你特别关心的部分,这里把 Ch04 的"策略网络"和"训练配置"完整展开。所有 4 个文件(CartPole / FrozenLake 三种)用的**都是同一个 `Net` 类**,只是输入维度(`obs_size`)和输出维度(`n_actions`)随环境变化。

### 11.1 `Net` 类的定义

```python
class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.net(x)
```

实例化参数(以 `01_cartpole.py` 为例):
```python
obs_size  = env.observation_space.shape[0]   # CartPole-v1 → 4
n_actions = env.action_space.n                # CartPole-v1 → 2
net = Net(obs_size, HIDDEN_SIZE, n_actions)   # Net(4, 128, 2)
```

### 11.2 网络结构的精确描述

> 严格说它是一个**"两层的、近线性变化"**的神经网络——最后一层是纯线性变化,第一层因为后面挂了 `ReLU`,等价于"线性变换 + 分段线性激活",**不是严格意义的线性变换**。

```
输入 obs(维度 = obs_size)
   │
   ▼
┌─────────────────────────────┐
│ Linear(obs_size → 128)      │  ← W₁ ∈ R^{128 × obs_size}, b₁ ∈ R^{128}
└─────────────────────────────┘
   │
   ▼
┌─────────────────────────────┐
│ ReLU()                      │  ← max(0, ·), 引入非线性
└─────────────────────────────┘
   │
   ▼
┌─────────────────────────────┐
│ Linear(128 → n_actions)     │  ← W₂ ∈ R^{n_actions × 128}, b₂ ∈ R^{n_actions}
└─────────────────────────────┘
   │
   ▼
输出 logits(维度 = n_actions)  ← 不是概率,后面才套 Softmax
```

以 `01_cartpole.py` 为例的具体形状(`obs_size=4, hidden_size=128, n_actions=2`):

| 层 | 输入形状 | 输出形状 | 可训练参数 |
| --- | -------- | -------- | ---------- |
| `Linear(4, 128)` | `(B, 4)` | `(B, 128)` | `4×128 + 128 = 640` |
| `ReLU` | `(B, 128)` | `(B, 128)` | 0 |
| `Linear(128, 2)` | `(B, 128)` | `(B, 2)` | `128×2 + 2 = 258` |
| **合计** | — | — | **898** |

> 整个 Agent 总共只有 **898 个参数**,非常小。所以 Ch04 的网络是"两层 MLP,中间夹一个 ReLU",不是真正意义上的"线性网络",但因为只有一层非线性,仍然可以归类为"**近线性 (near-linear)**"或"**浅层 (shallow)**"网络。

### 11.3 训练配置(顺带也补上)

```python
objective = nn.CrossEntropyLoss()            # 监督学习分类损失
optimizer = optim.Adam(params=net.parameters(), lr=0.01)
```

- **损失函数**:`CrossEntropyLoss` —— 把网络输出当作 logits,把精英轨迹中实际执行过的动作当作标签,本质就是"多分类监督学习"。
- **优化器**:Adam,学习率 `0.01` (CartPole)/ `0.001` (FrozenLake 三种变体)。
- **前向 + 采样**:
  ```python
  sm = nn.Softmax(dim=1)
  obs_v   = torch.FloatTensor([obs])         # (1, obs_size)
  logits  = net(obs_v)                       # (1, n_actions), 注意:不是概率
  probs   = sm(logits)                       # (1, n_actions), 转成概率
  action  = np.random.choice(n_actions, p=probs.data.numpy()[0])
  ```
  注意 `nn.CrossEntropyLoss` 内部已经会做 `LogSoftmax`,所以训练时**直接用 logits**,**不要在外面再 Softmax 一次**;但**采样时必须**显式 Softmax 才能按概率选动作。

### 11.4 四个文件的网络结构是否一样?

| 文件 | `obs_size` | `n_actions` | `Net(...)` | 网络本质 |
| ---- | ---------- | ----------- | ---------- | -------- |
| `01_cartpole.py` | 4 | 2 | `Net(4, 128, 2)` | 898 参数,2 层 MLP + ReLU |
| `02_frozenlake_naive.py` | 16 (OneHot 后) | 4 | `Net(16, 128, 4)` | 2560 参数,2 层 MLP + ReLU |
| `03_frozenlake_tweaked.py` | 16 | 4 | `Net(16, 128, 4)` | 同上 |
| `04_frozenlake_nonslippery.py` | 16 | 4 | `Net(16, 128, 4)` | 同上 |

**所有四种都是同一个 `Net` 类**(`obs_size`、`n_actions` 不同而已),都没有卷积层、注意力层、归一化层,也没有共享 encoder——是一个非常朴素的"输入向量 → 一层 ReLU MLP → 输出 logits"结构。

### 11.5 直观总结

> **Ch04 的 Agent ≈ 一个"两层的、近线性"的小 MLP**:
> - 参数量:898(CartPole) / 2560(FrozenLake),都是千级以下。
> - 激活:只有第一层后接 ReLU,最后一层是纯线性,输出 logits。
> - 训练:CrossEntropyLoss + Adam,纯粹的有监督分类。
> - 推理:Softmax 后按概率采样(训练时)或 argmax(评估时)。
>
> 真正"难"的部分不在网络结构(网络很简单),而在**怎么选 elite 数据**——所以 Ch04 的功夫全花在 `filter_batch` / `iterate_batches` / 各种 percentile 与折扣奖励的调参上,网络本身只是个"模仿器"。
