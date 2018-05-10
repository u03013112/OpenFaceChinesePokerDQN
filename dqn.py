import sys
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from SumTree import SumTree

# Double DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.load_model = True
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these is hyper parameters for the Double DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 0.1
        # self.epsilon_decay = 0.1 
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 500
        self.memory_size = 2000
        # create replay memory using deque
        self.memory = Memory(self.memory_size)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/dqn1.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state ,actions):
        # print("actions:",actions)
        if np.random.rand() <= self.epsilon:
            # return random.randrange(self.action_size)
            i = random.randrange(len(actions))
            # print("random:",actions[i])
            return actions[i]
        else:
            q_value = self.model.predict(state)
            for i in range(len(q_value[0])):
                findit = False
                for a in actions:
                    if i == a:
                        findit = True
                        break
                if findit == False:
                    q_value[0][i] = -1
            # print("q_value",q_value[0])
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done ,actions):
        # self.memory.append((state, action, reward, next_state, done ,actions))
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
        target = self.model.predict([state])
        old_val = target[0][action]
        target_val = self.target_model.predict([next_state])
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.discount_factor * (
                np.amax(target_val[0]))
        error = abs(old_val - target[0][action])

        self.memory.add(error, (state, action, reward, next_state, done, actions))

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # print("train")
        # batch_size = min(self.batch_size, len(self.memory))
        # mini_batch = random.sample(self.memory, batch_size)
        batch_size = self.batch_size
        mini_batch = self.memory.sample(self.batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []
        actions = []

        errors = np.zeros(self.batch_size)
        
        # print("0:>>>",mini_batch[0],"<<<")

        for i in range(batch_size):
            update_input[i] = mini_batch[i][1][0]
            action.append(mini_batch[i][1][1])
            reward.append(mini_batch[i][1][2])
            update_target[i] = mini_batch[i][1][3]
            done.append(mini_batch[i][1][4])
            actions.append(mini_batch[i][1][5])
            
        target = self.model.predict(update_input)#当前动作q估计
        target_next = self.model.predict(update_target)#下一个动作q估计
        target_val = self.target_model.predict(update_target)#下一个动作target网络的q估计

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            old_val = target[i][action[i]]
            if done[i]:
                target[i][action[i]] = reward[i]#做实一个状态下的指定动作的奖励
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model

                for j in range(len(target_next[i])):
                    findit = False
                    for a in actions[i]:
                        if j == a:
                            findit = True
                            break
                    if findit == False:
                        target_next[i][j] = -1
                # print("ob:",update_target[i])
                # print("actions:",actions[i])
                # print("target_next:",target_next[i])
                a = np.argmax(target_next[i])#预计下一个动作
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    target_val[i][a])#这里不理解，为什么要用target网络
                errors[i] = abs(old_val - target[i][action[i]])

        for i in range(self.batch_size):
            idx = mini_batch[i][0]
            self.memory.update(idx, errors[i])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)