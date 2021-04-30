import random
import matplotlib.pyplot as plt
import pickle

plane = [[None for i in range(100)] for i in range(100)]

class Agent:
    def __init__(self, start = (0,0), end = (100,50)):
        self.start = start
        self.end = end
        self.g = 9.8
        self.pos = (0,0)
        self.q = self.load_model()
        # self.dx = .1
        # self.dy = .1
        self.epsilon = .1
        self.alpha = .5
        self.time_needed = 0
        self.visited = set()

    def load_model(self):
        f = open("model_test.pickle", "rb")
        x = pickle.load(f)
        f.close()
        return x

    # def grad(self):
    #     pos = self.pos
    #     return -1/((pos[1]-self.end[1])/(pos[0]-self.end[0]))

    def check_action(self, pos, action):
        x, y = pos
        x_, y_ = action
        if action in self.visited:
            return False
        if x_>100 or y_>100 or y_==0:            #..
            return False
        if y<=50 and y_<y:                       #..
            return False
        if x==100 and y>50 and y_>y:             #..
            return False
        return True

    def available_actions(self, pos):
        x, y = pos
        right = (x+1,y)                          #..
        up = (x, y-1)                            #..
        down = (x, y+1)                          #..

        possibles = [right, up, down]

        def check(action):
            return self.check_action(pos, action)

        confirmed = filter(check, possibles)

        return list(confirmed)

    def dt(self, action):
        return 1/(2 * self.g * action[1])**.5

    def rounding(self, state):
        return (round(state[0],1), round(state[1],1))

    def get_q_value(self, state, action):
        state = self.rounding(state)
        action = self.rounding(action)
        if (state, action) not in self.q:
            return 0
        return self.q[state,tuple(action)]

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        state = self.rounding(state)
        action = self.rounding(action)
        self.q[state, action] = old_q + self.alpha*((reward + future_rewards) - old_q)

    def check_end(self):
        if self.pos == self.end:
            return True
        return False

    def best_future_reward(self, state):
        actions = self.available_actions(state)
        if actions == []:
            return 0
        return max([self.get_q_value(state, action) for action in actions])

    def choose_action(self, state, epsilon=True):
        actions = self.available_actions(state)
        if actions == []:
            return None
        greedy_choice = actions[0]
        for action in actions:
            if self.get_q_value(state, action) > self.get_q_value(state, greedy_choice):
                greedy_choice = action
        random_choice = random.choice(actions)
        if epsilon:
            return random.choices([random_choice,greedy_choice],
                                    weights=[self.epsilon, 1-self.epsilon])[0]
        return greedy_choice

    def update(self, old_state, action, new_state, reward):
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def reset(self):
        self.pos = (0,0)
        self.time_needed = 0
        self.visited = set()

    def train(self, n):
        for i in range(n):
            path = [self.pos,]
            last = {"state": None, "action": None}

            while True:
                # Keep track of current state and action
                state = self.pos
                self.visited.add(state)
                state = self.rounding(state)
                action = self.choose_action(state)
                action = self.rounding(action)
                path.append(action)

                # Keep track of last state and action
                last["state"] = state
                last["action"] = action

                # Make move
                new_state = action
                self.time_needed += self.dt(action)
                self.pos = action

                # When path is ended, update Q values with inverse times
                if self.check_end() == True:
                    # print("gg")
                    self.update(
                        last["state"],
                        last["action"],
                        new_state,
                        1000/self.time_needed
                    )
                    break

                # If game is continuing, no rewards yet
                elif last["state"] is not None:
                    self.update(
                        last["state"],
                        last["action"],
                        new_state,
                        0
                    )
            self.reset()

            if i%1000 == 0:
                print(f"done {i} times...")
        print("Done training")

    def walk(self):
        path = [self.pos,]
        last = {"state": None, "action": None}

        while True:
            # Keep track of current state and action
            state = self.pos
            self.visited.add(state)
            state = self.rounding(state)
            action = self.choose_action(state, epsilon=False)
            action = self.rounding(action)
            path.append(action)

            # Make move
            new_state = action
            self.pos = action

            # When path is ended
            if self.check_end() == True:
                self.reset()
                break

        return path



a = Agent()


a.train(10_000)

f = open("model_test.pickle", "wb")
pickle.dump(a.q, f)
f.close()

# import pprint
# pprint.pprint(a.q)

data = a.walk()

print(data)

x_val = [x[0] for x in data]
y_val = [-x[1] for x in data]
plt.scatter(x_val, y_val)
plt.show()


