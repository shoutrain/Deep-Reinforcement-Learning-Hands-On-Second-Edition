import gymnasium as gym

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, "recording")

    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if terminated:
            break

    print(f"Episode done in {total_steps} steps, total reward {total_reward}")
    env.close()
    env.env.close()
