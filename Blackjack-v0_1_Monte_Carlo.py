# Monte Carlo control without exploring starts
# Sample performance: Win rate: 0.407   Loss Rate: 0.0   Draw Rate: 0.08
import gym
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    env = gym.make('Blackjack-v0')
    EPS = 0.05
    GAMMA = 1.0

    Q = {}
    agent_sum_space = [i for i in range(4,22)]
    dealer_show_card_space = [i+1 for i in range(10)]
    agent_ace_space = [False, True]
    action_space = [0,1]

    state_space = []
    returns = {}
    pairs_visited = {}
    for total in agent_sum_space:
        for card in dealer_show_card_space:
            for ace in agent_ace_space:
                for action in action_space:
                    Q[((total, card, ace), action)] = 0
                    returns[((total, card, ace), action)] = 0
                    pairs_visited[((total, card, ace), action)] = 0
                state_space.append((total, card, ace))

    policy = {}
    for state in state_space:
        policy[state] = np.random.choice(action_space)

    number_of_episodes = 1000000
    for i in range(number_of_episodes):
        states_actions_returns = []
        memory = []
        if i % 50000 == 0:
            print("Starting episode ", i)
        observation = env.reset()
        done = False
        while not done:
            action = policy[observation]
            observation_, reward, done, info = env.step(action)
            memory.append((observation[0], observation[1], observation[2], action, reward))
            observation = observation_
        memory.append((observation[0], observation[1], observation[2], action, reward))

        G = 0
        last = True
        for player_sum, dealer_card, usable_ace, action, reward in reversed(memory):
            if last:
                last = False
            else:
                states_actions_returns.append((player_sum, dealer_card, usable_ace, action, G))
            G = GAMMA*G + reward

        states_actions_returns.reverse()
        states_actions_visited = []

        for player_sum, dealer_card, usable_ace, action, G in states_actions_returns:
            sa = ((player_sum, dealer_card, usable_ace), action)
            if sa not in states_actions_visited:
                pairs_visited[sa] += 1
                # incremental implementation
                # new estimate = 1 / N * [sample - old estimate]
                returns[(sa)] += (1 / pairs_visited[(sa)])*(G-returns[(sa)])
                Q[sa] = returns[sa]
                rand = np.random.random()
                if rand < 1 - EPS:
                    state = (player_sum, dealer_card, usable_ace)
                    values = np.array([Q[(state, a)] for a in action_space ])
                    best = np.random.choice(np.where(values==values.max())[0])
                    policy[state] = action_space[best]
                else:
                    policy[state] = np.random.choice(action_space)
                states_actions_visited.append(sa)
        if EPS - 1e-7 > 0:
            EPS -= 1e-7
        else:
            EPS = 0

    # reassigning this variable
    number_of_episodes = 1000
    rewards = np.zeros(number_of_episodes)
    total_reward = 0
    wins = 0
    losses = 0
    draws = 0
    print('getting ready to test policy')
    for i in range(number_of_episodes):
        observation = env.reset()
        done = False
        while not done:
            action = policy[observation]
            observation_, reward, done, info = env.step(action)
            observation = observation_
        total_reward += reward
        rewards[i] = total_reward

        if reward >= 1:
            wins += 1
        elif reward == 0:
            draws  += 1
        elif reward == 1:
            losses += 1

    wins /= number_of_episodes
    losses /= number_of_episodes
    draws /= number_of_episodes
    print('Win rate: ', wins, 'Loss Rate: ', losses, 'Draw Rate: ', draws)
    plt.plot(rewards)
    plt.show()
