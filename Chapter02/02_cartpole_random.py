import gymnasium as gym


if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")

    total_reward = 0.0
    total_steps = 0
    obs, info = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1
        if terminated or truncated:
            break

    env.close()
    print(f"Episode done in { total_steps} steps, total reward {total_reward}")
