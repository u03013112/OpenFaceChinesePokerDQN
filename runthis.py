from dqn import DoubleDQNAgent
from Server import Env
import numpy as np
import pylab
import time

EPISODES = 30000

if __name__ == "__main__":
    env = Env()
    # get size of state and action from environment
    state_size = env.stateSize
    action_size = env.actionSize

    agent = DoubleDQNAgent(state_size, action_size)

    scores, episodes = [], []
    step = 0
    totalScore = 0
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            step += 1
            actions = env.getActions(np.reshape(state,[state_size]).tolist())

            action = agent.get_action(state,actions)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            actions = env.getActions(np.reshape(next_state,[state_size]).tolist())#找到目标状态可使用的动作集合
            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done,actions)#这里多存储一个actions以供学习时使用
            # every time step do the training
            if step >= agent.train_start and step%5==0:
                agent.train_model()
            score = reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                scores.append(score)
                episodes.append(e)
                # pylab.plot(episodes, scores, 'b')
                # pylab.savefig("./save_graph/dqn.png")
                totalScore += score
                print("episode:", e, "  score:", score,"  epsilon:", agent.epsilon,"e-score:",totalScore/(e+1))

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                # if np.mean(scores[-min(10, len(scores)):]) > 490:
                #     sys.exit()

        # save the model
        if e % 50 == 0:
            # print("save")
            agent.model.save_weights("./save_model/dqn1.h5")
