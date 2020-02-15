import numpy as np
import matplotlib.pyplot as plt
import gym

def maxAction(Q1, Q2, state):
    values = np.array([Q1[state,a] + Q2[state,a] for a in range(2)])
    action = np.argmax(values)
    return action

#discretize the spaces
pole_theta_space = np.linspace(-0.20943951, 0.20943951, 10)
pole_theta_velocity_space = np.linspace(-4, 4, 10)
cart_position_space = np.linspace(-2.4, 2.4, 10)
cart_velocity_space = np.linspace(-4, 4, 10)

def getState(observation):
    cart_x, cart_x_dot, cart_theta, cart_theta_dot = observation
    cart_x = int(np.digitize(cart_x, cart_position_space))
    cart_x_dot = int(np.digitize(cart_x_dot, cart_velocity_space))
    cart_theta = int(np.digitize(cart_theta, pole_theta_space))
    cart_theta_dot = int(np.digitize(cart_theta_dot, pole_theta_velocity_space))

    return (cart_x, cart_x_dot, cart_theta, cart_theta_dot)

def plot_running_avg(total_rewards):
	N = len(total_rewards)
	running_avg = np.empty(N)
	for t in range(N):
		running_avg[t] = np.mean(total_rewards[max(0, t-100):(t+1)])
	plt.plot(running_avg)
	plt.title("Running Average")
	plt.show()

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    # model hyperparamters
    ALPHA = 0.1
    GAMMA = 0.9
    EPS = 1.0

    #construct state space
    states = []
    for i in range(len(cart_position_space)+1):
        for j in range(len(cart_velocity_space)+1):
            for k in range(len(pole_theta_space)+1):
                for l in range(len(pole_theta_velocity_space)+1):
                    states.append((i,j,k,l))

    Q1, Q2 = {}, {}
    for s in states:
        for a in range(2):
            Q1[s,a] = 0
            Q2[s,a] = 0

    number_of_games = 10000
    total_rewards = np.zeros(number_of_games)
    for i in range(number_of_games):
        if i % 1000 == 0:
            print('starting game', i)
        # cart x position, cart velocity, pole theta, pole velocity
        observation = env.reset()
        done = False
        episode_rewards = 0
        while not done:
            if i % 500 == 0:
                env.render()

            s = getState(observation)
            rand = np.random.random()
            a = maxAction(Q1,Q2,s) if rand < (1-EPS) else env.action_space.sample()
            observation_, reward, done, info = env.step(a)
            episode_rewards += reward

            s_ = getState(observation_)
            rand = np.random.random()
            if rand <= 0.5:
                a_ = maxAction(Q1,Q1,s)
                Q1[s,a] = Q1[s,a] + ALPHA*(reward + GAMMA*Q2[s_,a_] - Q1[s,a])
            elif rand > 0.5:
                a_ = maxAction(Q2,Q2,s)
                Q2[s,a] = Q2[s,a] + ALPHA*(reward + GAMMA*Q1[s_,a_] - Q2[s,a])

            observation = observation_

        EPS -= 2/(number_of_games) if EPS > 0 else 0
        total_rewards[i] = episode_rewards

    plot_running_avg(total_rewards)
