# Deep-Reinforcement-Learning-Hands-On-Second-Edition

Deep-Reinforcement-Learning-Hands-On-Second-Edition, published by Packt

The original example projects don't work well with the up-to-date libs, so I have to update some code to meet the up-to-date libs.

I am using python 3.12.2, torch 2.4.1, gymnasium 0.29.1(instead of gym <=0.26.2) and corresponding other packages. I also installed ale-py and AutoRom to make gymnasium can work with atari:

```python
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)  # make gymnasium work with atari
```