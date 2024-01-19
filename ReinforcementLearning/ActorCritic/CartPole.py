import gym
import HyperParams
import matplotlib.pyplot as plt
from ActorCritic import ActorCritic
from numpy import mean
from torch import tensor

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_card = env.action_space.n
    agent = ActorCritic(state_dim, HyperParams.hidden_dim, action_card, HyperParams.actor_lr, HyperParams.critic_lr,
                        HyperParams.gamma, HyperParams.target_update_frequency, HyperParams.device)

    returns = []

    for i in range(HyperParams.num_episodes):
        transitions = []
        # record episode's return to plot
        episode_return = 0
        # 状态变量：
        # 1. 小车在轨道上的位置（position of the cart on the track）
        # 2. 杆子与竖直方向的夹角（angle of the pole with the vertical）
        # 3. 小车速度（cart velocity）
        # 4. 角度变化率（rate of change of the angle）
        state = env.reset()[0]
        while True:
            # action为0代表推动向左移，为1代表推动向右移，施加的力减小或增加的速度不是固定的，它取决于杆子指向的角度。杆子的重心会改变将推车移动到其下方所需的能量
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            transitions.append(
                (tensor(state), tensor(action), tensor(next_state), tensor(reward), terminated or truncated))
            if terminated or truncated:
                returns.append(episode_return)
                break
            state = next_state
        agent.update(transitions)
        if (i + 1) % 50 == 0:
            print("episodes:{}->{}, episode_returns_mean:{}.".format(i - 49, i, mean(returns[i - 49:i])))
        if mean(returns[i - 49: i]) > 400:
            break
    env.close()

    # plot
    plt.figure(dpi=400)
    plt.plot(returns, c="darkblue")
    plt.xlabel("episode")
    plt.ylabel("return")
    plt.show()

    env = gym.make("CartPole-v1", render_mode="human")

    for i in range(10):
        state = env.reset()[0]
        while True:
            env.render()
            action = agent.take_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
            state = next_state
