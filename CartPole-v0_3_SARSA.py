import numpy as np
import matplotlib.pyplot as plt
import gym

def maxAction(Q, state):
    values = np.array([Q[state,a] for a in range(2)])
    action = np.argmax(values)
    return action

#discretize the space
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

    Q = {}
    for s in states:
        for a in range(2):
            Q[s, a] = 0

    number_of_games = 10000
    total_rewards = np.zeros(number_of_games)
    for i in range(number_of_games):
        if i % 1000 == 0:
            print('starting game', i)
        # cart x position, cart velocity, pole theta, pole velocity
        observation = env.reset()
        s = getState(observation)
        rand = np.random.random()
        a = maxAction(Q, s) if rand < (1-EPS) else env.action_space.sample()
        done = False
        episode_rewards = 0
        while not done:
            if i % 500 == 0:
                env.render()
            observation_, reward, done, info = env.step(a)
            s_ = getState(observation_)
            rand = np.random.random()
            a_ = maxAction(Q, s_) if rand < (1-EPS) else env.action_space.sample()
            episode_rewards += reward
            Q[s,a] = Q[s,a] + ALPHA*(reward + GAMMA*Q[s_,a_] - Q[s,a])
            s, a = s_, a_
        EPS -= 2/(number_of_games) if EPS > 0 else 0
        total_rewards[i] = episode_rewards
    plot_running_avg(total_rewards)
