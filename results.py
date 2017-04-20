import time
import random
import numpy as np
import matplotlib.pyplot as plt
from ipythonblocks import BlockGrid
from IPython.display import clear_output
from sets import Set

class BlockingMaze:
    def __init__(self):
        self.state = [0, 3]
        self.GOAL = [5, 8]
        self.MAX_ROW = 5
        self.MAX_COLUMN = 8
        self.BLOCKS = [[2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7]]
        self.stepCount = 0
        self.secondPhase = False
    
    def takeAction(self, action):
        if self.isMoveAllowed(action):
            self.state += action

        self.stepCount += 1
        if not self.secondPhase and self.stepCount > 1000:
            self.BLOCKS[0] = [2, 8]
            self.secondPhase = True
            
        if np.array_equal(self.state, self.GOAL):
            return 1
        else:
            return 0
    
    def isMoveAllowed(self, action):
        stateAfterAction = self.state + action
        
        if (stateAfterAction[0] < 0 or stateAfterAction[0] > self.MAX_ROW or
            stateAfterAction[1] < 0 or stateAfterAction[1] > self.MAX_COLUMN):
                return False
        
        for block in self.BLOCKS:
            if np.array_equal(stateAfterAction, block):
                return False
        
        return True
    
    def reset(self):
        self.state = [0, 3]
        self.stepCount = 0
        self.BLOCKS[0] = [2, 0]
        self.secondPhase = False
        
    def resetState(self):
        self.state = [0, 3]
        
    # Displays the current state of the environment in a nice colored grid
    def view(self):
        gridHeight = self.MAX_ROW
        gridWidth = self.MAX_COLUMN
        
        grid = BlockGrid(self.MAX_COLUMN + 1, self.MAX_ROW + 1, fill=(224, 224, 224))
        for block in self.BLOCKS:
            grid[gridHeight - block[0], block[1]] = (255, 255, 255)
        grid[gridHeight - self.state[0], self.state[1]] = (0, 0, 0)
        grid[gridHeight - self.GOAL[0], self.GOAL[1]] = (255, 0, 0)
        
        grid.show()

class ControlAlgorithm:
    def __init__(self, environment, ALPHA, DISCOUNT, EPSILON):
        self.Q = np.zeros((environment.MAX_ROW + 1, environment.MAX_COLUMN + 1, ACTIONS.shape[0]))
        self.environment = environment
        self.environment.reset()
        self.ALPHA = ALPHA
        self.DISCOUNT = DISCOUNT
        self.EPSILON = EPSILON
      
    def averageMTimesOverNSteps(self, M, N):
        episodesPerTimeStep = np.zeros((M, N))
        for i in range(M):
            self.reset()
            self.environment.reset()
            episodesPerTimeStep[i] = self.runForNTimeSteps(N)

        return np.average(episodesPerTimeStep, axis=0)
    
    def runForNTimeSteps(self, N):
        stepCount = 0
        cumulativeReward = 0
        rewardsPerTimeStep = np.zeros(N)
        
        while stepCount < N:
            episodeLength, episodeReward = self.generateEpisode()
            cumulativeReward += episodeReward
            
            for i in range(stepCount, stepCount + episodeLength):
                if i >= N:
                    break
                
                rewardsPerTimeStep[i] = cumulativeReward
                
            stepCount += episodeLength
        
        return rewardsPerTimeStep
    
    def chooseAction(self):
        if np.random.rand(1)[0] <= self.EPSILON:
            actionId = np.random.randint(ACTIONS.shape[0], size=1)[0]
        else:
            actionId = np.argmax(self.Q[self.environment.state[0], self.environment.state[1]])
            
            # If multiple actions have the same max value, we need to choose one of them randomly
            possibleActions = []
            for i in range(ACTIONS.shape[0]):
                if (self.Q[self.environment.state[0], self.environment.state[1], i] == 
                   self.Q[self.environment.state[0], self.environment.state[1], actionId]):
                    possibleActions.append(i)
                    
            actionId = random.choice(possibleActions)
        
        return ACTIONS[actionId], actionId
    
    # Displays the learned policy trying to complete an episode.
    def viewLearnedPolicy(self):
        self.environment.resetState()
        self.environment.view()
        
        reward = 0
        
        while reward == 0:
            previousState = np.copy(self.environment.state)
            
            action, actionId = self.chooseAction()
            reward = self.environment.takeAction(action)
            
            clear_output()
            self.environment.view()
            time.sleep(0.5)

class DynaQ(ControlAlgorithm):
    def __init__(self, environment, ALPHA, DISCOUNT, EPSILON, N):
        ControlAlgorithm.__init__(self, environment, ALPHA, DISCOUNT, EPSILON)
        
        # model stores for each state-action pair:
        # 0 - Following state row
        # 1 - Following state column
        # 2 - Reward gotten
        # 3 - If this action was ever tried (0 no, 1 yes)
        self.model = np.zeros((environment.MAX_ROW + 1, environment.MAX_COLUMN + 1, ACTIONS.shape[0], 4))
        
        self.triedStateActionPairs = []
        
        self.N = N
        
    def reset(self):
        self.Q = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0]))
        self.model = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0], 4))
        self.triedStateActionPairs = []
        
    def generateEpisode(self):
        self.environment.resetState()
        
        stepCount = 0
        reward = 0
        cumulativeReward = 0

        while not np.array_equal(self.environment.state, self.environment.GOAL):
            previousState = np.copy(self.environment.state)
            
            action, actionId = self.chooseAction()
            reward = self.environment.takeAction(action)
            cumulativeReward += reward
        
            self.Q[previousState[0], previousState[1], actionId] += self.ALPHA * (reward + 
                self.DISCOUNT * np.max(self.Q[self.environment.state[0], self.environment.state[1]]) - 
                self.Q[previousState[0], previousState[1], actionId])
            
            self.model[previousState[0], previousState[1], actionId, 0] = self.environment.state[0]
            self.model[previousState[0], previousState[1], actionId, 1] = self.environment.state[1]
            self.model[previousState[0], previousState[1], actionId, 2] = reward
            
            if self.model[previousState[0], previousState[1], actionId, 3] == 0:
                self.triedStateActionPairs.append([previousState[0], previousState[1], actionId])
                self.model[previousState[0], previousState[1], actionId, 3] = 1
                
            
            # Planning
            for i in range(self.N):
                stateActionPair = random.choice(self.triedStateActionPairs)
                state = stateActionPair[0:2]
                actionId = stateActionPair[2]
                nextState = [0, 0]
                nextState[0] = int(self.model[state[0], state[1], actionId, 0])
                nextState[1] = int(self.model[state[0], state[1], actionId, 1])
                reward = self.model[state[0], state[1], actionId, 2]
                
                self.Q[state[0], state[1], actionId] += self.ALPHA * (reward + 
                    self.DISCOUNT * np.max(self.Q[nextState[0], nextState[1]]) - 
                    self.Q[state[0], state[1], actionId])
            
            stepCount += 1
        
        return stepCount, cumulativeReward

class DynaQPlus(ControlAlgorithm):
    def __init__(self, environment, ALPHA, DISCOUNT, EPSILON, N, K):
        ControlAlgorithm.__init__(self, environment, ALPHA, DISCOUNT, EPSILON)
        # model stores for each state-action pair:
        # 0 - Following state row
        # 1 - Following state column
        # 2 - Reward gotten
        # 3 - Time steps since the action was last tried, in this state
        # 4 - If this action was ever tried (0 no, 1 yes)
        self.model = np.zeros((environment.MAX_ROW + 1, environment.MAX_COLUMN + 1, ACTIONS.shape[0], 5))
        
        self.triedStateActionPairs = []
        
        self.N = N
        self.K = K
        
    def reset(self):
        self.Q = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0]))
        self.model = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0], 5))
        self.triedStateActionPairs = []
        
    def generateEpisode(self):
        self.environment.resetState()
        
        stepCount = 0
        reward = 0
        cumulativeReward = 0

        while not np.array_equal(self.environment.state, self.environment.GOAL):
            previousState = np.copy(self.environment.state)
            
            action, actionId = self.chooseAction()
            reward = self.environment.takeAction(action)
            cumulativeReward += reward
        
            self.Q[previousState[0], previousState[1], actionId] += self.ALPHA * (reward + 
                self.DISCOUNT * np.max(self.Q[self.environment.state[0], self.environment.state[1]]) - 
                self.Q[previousState[0], previousState[1], actionId])
            
            # We update the time since we last tried all previous state-action pairs
            for i in range(environment.MAX_ROW + 1):
                for j in range(environment.MAX_COLUMN + 1):
                    for k in range(ACTIONS.shape[0]):
                        self.model[i, j, k, 3] += 1
            
            self.model[previousState[0], previousState[1], actionId, 0] = self.environment.state[0]
            self.model[previousState[0], previousState[1], actionId, 1] = self.environment.state[1]
            self.model[previousState[0], previousState[1], actionId, 2] = reward
            self.model[previousState[0], previousState[1], actionId, 3] = 0
            
            if self.model[previousState[0], previousState[1], actionId, 4] == 0:
                self.triedStateActionPairs.append([previousState[0], previousState[1], actionId])
                self.model[previousState[0], previousState[1], actionId, 4] = 1
            
            # Planning
            for i in range(self.N):
                stateActionPair = random.choice(self.triedStateActionPairs)
                state = stateActionPair[0:2]
                actionId = stateActionPair[2]
                nextState = [0, 0]
                nextState[0] = int(self.model[state[0], state[1], actionId, 0])
                nextState[1] = int(self.model[state[0], state[1], actionId, 1])
                reward = (self.model[state[0], state[1], actionId, 2] + 
                          self.K * np.sqrt(self.model[state[0], state[1], actionId, 3]))
                
                self.Q[state[0], state[1], actionId] += self.ALPHA * (reward + 
                    self.DISCOUNT * np.max(self.Q[nextState[0], nextState[1]]) - 
                    self.Q[state[0], state[1], actionId])
            
            stepCount += 1
        
        return stepCount, cumulativeReward

class DynaQGreedyPlus(ControlAlgorithm):
    def __init__(self, environment, ALPHA, DISCOUNT, N, K):
        ControlAlgorithm.__init__(self, environment, ALPHA, DISCOUNT, 0)
        # model stores for each state-action pair:
        # 0 - Following state row
        # 1 - Following state column
        # 2 - Reward gotten
        # 3 - Time steps since the action was last tried, in this state
        # 4 - If this action was ever tried (0 no, 1 yes)
        self.model = np.zeros((environment.MAX_ROW + 1, environment.MAX_COLUMN + 1, ACTIONS.shape[0], 5))
        
        self.triedStateActionPairs = []
        
        self.N = N
        self.K = K
        
    def reset(self):
        self.Q = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0]))
        self.model = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0], 5))
        self.triedStateActionPairs = []
        
    def generateEpisode(self):
        self.environment.resetState()
        
        stepCount = 0
        reward = 0
        cumulativeReward = 0

        while not np.array_equal(self.environment.state, self.environment.GOAL):
            previousState = np.copy(self.environment.state)
            
            action, actionId = self.chooseAction()
            reward = self.environment.takeAction(action)
            cumulativeReward += reward
        
            self.Q[previousState[0], previousState[1], actionId] += self.ALPHA * (reward + 
                self.DISCOUNT * np.max(self.Q[self.environment.state[0], self.environment.state[1]]) - 
                self.Q[previousState[0], previousState[1], actionId])
            
            # We update the time since we last tried all previous state-action pairs
            for i in range(environment.MAX_ROW + 1):
                for j in range(environment.MAX_COLUMN + 1):
                    for k in range(ACTIONS.shape[0]):
                        self.model[i, j, k, 3] += 1
            
            self.model[previousState[0], previousState[1], actionId, 0] = self.environment.state[0]
            self.model[previousState[0], previousState[1], actionId, 1] = self.environment.state[1]
            self.model[previousState[0], previousState[1], actionId, 2] = reward
            self.model[previousState[0], previousState[1], actionId, 3] = 0
            
            if self.model[previousState[0], previousState[1], actionId, 4] == 0:
                self.triedStateActionPairs.append([previousState[0], previousState[1], actionId])
                self.model[previousState[0], previousState[1], actionId, 4] = 1
            
            # Planning
            for i in range(self.N):
                stateActionPair = random.choice(self.triedStateActionPairs)
                state = stateActionPair[0:2]
                actionId = stateActionPair[2]
                nextState = [0, 0]
                nextState[0] = int(self.model[state[0], state[1], actionId, 0])
                nextState[1] = int(self.model[state[0], state[1], actionId, 1])
                reward = self.model[state[0], state[1], actionId, 2]
                
                self.Q[state[0], state[1], actionId] += self.ALPHA * (reward + 
                    self.DISCOUNT * np.max(self.Q[nextState[0], nextState[1]]) - 
                    self.Q[state[0], state[1], actionId])
            
            stepCount += 1
        
        return stepCount, cumulativeReward
    
    def chooseAction(self): # We override the usual chooseAction() function to implement the mentionned exploration bonus
        maxValue = np.NINF
        possibleActions = []
        
        for i in range(ACTIONS.shape[0]):
            if ((self.Q[self.environment.state[0], self.environment.state[1], i] + 
                 self.K * np.sqrt(self.model[self.environment.state[0], self.environment.state[1], i, 3])) 
                 > maxValue):
                possibleActions = [i]
                maxValue = (self.Q[self.environment.state[0], self.environment.state[1], i] + 
                            self.K * np.sqrt(self.model[self.environment.state[0], self.environment.state[1], i, 3]))
            elif ((self.Q[self.environment.state[0], self.environment.state[1], i] + 
                 self.K * np.sqrt(self.model[self.environment.state[0], self.environment.state[1], i, 3])) 
                 == maxValue):
                possibleActions.append(i)
                maxValue = (self.Q[self.environment.state[0], self.environment.state[1], i] + 
                            self.K * np.sqrt(self.model[self.environment.state[0], self.environment.state[1], i, 3]))

        actionId = random.choice(possibleActions)
        
        return ACTIONS[actionId], actionId

UP = np.array([1, 0])
RIGHT = np.array([0, 1])
DOWN = np.array([-1, 0])
LEFT = np.array([0, -1])

ACTIONS = np.array([UP, RIGHT, DOWN, LEFT])

environment = BlockingMaze()
ALPHA = 0.1
DISCOUNT = 0.95
EPSILON = 0.1
N = 50
K = 0.000001

dq = DynaQ(environment, ALPHA, DISCOUNT, EPSILON, N)
dqp = DynaQPlus(environment, ALPHA, DISCOUNT, EPSILON, N, K)
dqgp = DynaQGreedyPlus(environment, ALPHA, DISCOUNT, N, K)

rewardsPerTimeStepDynaQ = dq.averageMTimesOverNSteps(100, 3000)
rewardsPerTimeStepDynaQPlus = dqp.averageMTimesOverNSteps(100, 3000)
rewardsPerTimeStepDynaQGreedyPlus = dqgp.averageMTimesOverNSteps(100, 3000)

plt.figure(1)
plt.plot(np.arange(1, 3001), rewardsPerTimeStepDynaQ, 'r', label="Dyna-Q N=50")
plt.plot(np.arange(1, 3001), rewardsPerTimeStepDynaQPlus, 'b', label="Dyna-Q+ N=50 K=0.000001")
plt.plot(np.arange(1, 3001), rewardsPerTimeStepDynaQGreedyPlus, 'g', label="Greedy Dyna-Q+  N=50 K=0.000001")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Reward')
plt.title('Blocking Maze, averaged 100 times, over 3000 steps')

class DynaQGreedyPlus2(ControlAlgorithm):
    def __init__(self, environment, ALPHA, DISCOUNT, N, K):
        ControlAlgorithm.__init__(self, environment, ALPHA, DISCOUNT, 0)
        # model stores for each state-action pair:
        # 0 - Following state row
        # 1 - Following state column
        # 2 - Reward gotten
        # 3 - Time steps since the action was last tried, in this state
        # 4 - If this action was ever tried (0 no, 1 yes)
        self.model = np.zeros((environment.MAX_ROW + 1, environment.MAX_COLUMN + 1, ACTIONS.shape[0], 5))
        
        self.triedStateActionPairs = []
        
        self.N = N
        self.K = K
        
    def reset(self):
        self.Q = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0]))
        self.model = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0], 5))
        self.triedStateActionPairs = []
        
    def generateEpisode(self):
        self.environment.resetState()
        
        stepCount = 0
        reward = 0
        cumulativeReward = 0

        while not np.array_equal(self.environment.state, self.environment.GOAL):
            previousState = np.copy(self.environment.state)
            
            action, actionId = self.chooseAction()
            reward = self.environment.takeAction(action)
            cumulativeReward += reward
        
            self.Q[previousState[0], previousState[1], actionId] += self.ALPHA * (reward + 
                self.DISCOUNT * np.max(self.Q[self.environment.state[0], self.environment.state[1]]) - 
                self.Q[previousState[0], previousState[1], actionId])
            
            # We update the time since we last tried all previous state-action pairs
            for i in range(ACTIONS.shape[0]):
                self.model[previousState[0], previousState[1], i, 3] += 1
            
            self.model[previousState[0], previousState[1], actionId, 0] = self.environment.state[0]
            self.model[previousState[0], previousState[1], actionId, 1] = self.environment.state[1]
            self.model[previousState[0], previousState[1], actionId, 2] = reward
            self.model[previousState[0], previousState[1], actionId, 3] = 0
            
            if self.model[previousState[0], previousState[1], actionId, 4] == 0:
                self.triedStateActionPairs.append([previousState[0], previousState[1], actionId])
                self.model[previousState[0], previousState[1], actionId, 4] = 1
            
            # Planning
            for i in range(self.N):
                stateActionPair = random.choice(self.triedStateActionPairs)
                state = stateActionPair[0:2]
                actionId = stateActionPair[2]
                nextState = [0, 0]
                nextState[0] = int(self.model[state[0], state[1], actionId, 0])
                nextState[1] = int(self.model[state[0], state[1], actionId, 1])
                reward = self.model[state[0], state[1], actionId, 2]
                
                self.Q[state[0], state[1], actionId] += self.ALPHA * (reward + 
                    self.DISCOUNT * np.max(self.Q[nextState[0], nextState[1]]) - 
                    self.Q[state[0], state[1], actionId])
            
            stepCount += 1
        
        return stepCount, cumulativeReward
    
    def chooseAction(self): # We override the usual chooseAction() function to implement the mentionned exploration bonus
        maxValue = np.NINF
        possibleActions = []
        
        for i in range(ACTIONS.shape[0]):
            if ((self.Q[self.environment.state[0], self.environment.state[1], i] + 
                 self.K * np.sqrt(self.model[self.environment.state[0], self.environment.state[1], i, 3])) 
                 > maxValue):
                possibleActions = [i]
                maxValue = (self.Q[self.environment.state[0], self.environment.state[1], i] + 
                            self.K * np.sqrt(self.model[self.environment.state[0], self.environment.state[1], i, 3]))
            elif ((self.Q[self.environment.state[0], self.environment.state[1], i] + 
                 self.K * np.sqrt(self.model[self.environment.state[0], self.environment.state[1], i, 3])) 
                 == maxValue):
                possibleActions.append(i)
                maxValue = (self.Q[self.environment.state[0], self.environment.state[1], i] + 
                            self.K * np.sqrt(self.model[self.environment.state[0], self.environment.state[1], i, 3]))

        actionId = random.choice(possibleActions)
        
        return ACTIONS[actionId], actionId

UP = np.array([1, 0])
RIGHT = np.array([0, 1])
DOWN = np.array([-1, 0])
LEFT = np.array([0, -1])

ACTIONS = np.array([UP, RIGHT, DOWN, LEFT])

environment = BlockingMaze()
ALPHA = 0.1
DISCOUNT = 0.95
EPSILON = 0.1
N = 50
K = 0.000001

dqgp2 = DynaQGreedyPlus2(environment, ALPHA, DISCOUNT, N, K)
rewardsPerTimeStepDynaQGreedyPlus2 = dqgp2.averageMTimesOverNSteps(100, 3000)

plt.figure(2)
plt.plot(np.arange(1, 3001), rewardsPerTimeStepDynaQGreedyPlus, 'r', label="Greedy Dyna-Q+ N=50 K=0.000001")
plt.plot(np.arange(1, 3001), rewardsPerTimeStepDynaQGreedyPlus2, 'b', label="Greedy Dyna-Q+ variant N=50 K=0.000001")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Reward')
plt.title('Blocking Maze, averaged 100 times, over 3000 steps')

class DynaQGreedyPlusLN(ControlAlgorithm):
    def __init__(self, environment, ALPHA, DISCOUNT, N, K):
        ControlAlgorithm.__init__(self, environment, ALPHA, DISCOUNT, 0)
        # model stores for each state-action pair:
        # 0 - Following state row
        # 1 - Following state column
        # 2 - Reward gotten
        # 3 - Time steps since the action was last tried, in this state
        # 4 - If this action was ever tried (0 no, 1 yes)
        self.model = np.zeros((environment.MAX_ROW + 1, environment.MAX_COLUMN + 1, ACTIONS.shape[0], 5))
        
        self.triedStateActionPairs = []
        
        self.N = N
        self.K = K
        
    def reset(self):
        self.Q = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0]))
        self.model = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0], 5))
        self.triedStateActionPairs = []
        
    def generateEpisode(self):
        self.environment.resetState()
        
        stepCount = 0
        reward = 0
        cumulativeReward = 0

        while not np.array_equal(self.environment.state, self.environment.GOAL):
            previousState = np.copy(self.environment.state)
            
            action, actionId = self.chooseAction()
            reward = self.environment.takeAction(action)
            cumulativeReward += reward
        
            self.Q[previousState[0], previousState[1], actionId] += self.ALPHA * (reward + 
                self.DISCOUNT * np.max(self.Q[self.environment.state[0], self.environment.state[1]]) - 
                self.Q[previousState[0], previousState[1], actionId])
            
            # We update the time since we last tried all previous state-action pairs
            for i in range(environment.MAX_ROW + 1):
                for j in range(environment.MAX_COLUMN + 1):
                    for k in range(ACTIONS.shape[0]):
                        self.model[i, j, k, 3] += 1
            
            self.model[previousState[0], previousState[1], actionId, 0] = self.environment.state[0]
            self.model[previousState[0], previousState[1], actionId, 1] = self.environment.state[1]
            self.model[previousState[0], previousState[1], actionId, 2] = reward
            self.model[previousState[0], previousState[1], actionId, 3] = 0
            
            if self.model[previousState[0], previousState[1], actionId, 4] == 0:
                self.triedStateActionPairs.append([previousState[0], previousState[1], actionId])
                self.model[previousState[0], previousState[1], actionId, 4] = 1
            
            # Planning
            for i in range(self.N):
                stateActionPair = random.choice(self.triedStateActionPairs)
                state = stateActionPair[0:2]
                actionId = stateActionPair[2]
                nextState = [0, 0]
                nextState[0] = int(self.model[state[0], state[1], actionId, 0])
                nextState[1] = int(self.model[state[0], state[1], actionId, 1])
                reward = self.model[state[0], state[1], actionId, 2]
                
                self.Q[state[0], state[1], actionId] += self.ALPHA * (reward + 
                    self.DISCOUNT * np.max(self.Q[nextState[0], nextState[1]]) - 
                    self.Q[state[0], state[1], actionId])
            
            stepCount += 1
        
        return stepCount, cumulativeReward
    
    def chooseAction(self): # We override the usual chooseAction() function to implement the mentionned exploration bonus
        maxValue = np.NINF
        possibleActions = []
        
        for i in range(ACTIONS.shape[0]):
            bonusReward = 0
            if self.model[self.environment.state[0], self.environment.state[1], i, 3] != 0:
                bonusReward = np.sqrt(np.log(self.model[self.environment.state[0], self.environment.state[1], i, 3]))
                
            if (self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward) > maxValue:
                possibleActions = [i]
                maxValue = self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward
            elif (self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward) == maxValue:
                possibleActions.append(i)
                maxValue = self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward

        actionId = random.choice(possibleActions)
        
        return ACTIONS[actionId], actionId

UP = np.array([1, 0])
RIGHT = np.array([0, 1])
DOWN = np.array([-1, 0])
LEFT = np.array([0, -1])

ACTIONS = np.array([UP, RIGHT, DOWN, LEFT])

environment = BlockingMaze()
ALPHA = 0.1
DISCOUNT = 0.95
EPSILON = 0.1
N = 50
K = 0.000001

dqgpln = DynaQGreedyPlusLN(environment, ALPHA, DISCOUNT, N, K)
rewardsPerTimeStepDynaQGreedyPlusLN = dqgpln.averageMTimesOverNSteps(100, 3000)

plt.figure(3)
plt.plot(np.arange(1, 3001), rewardsPerTimeStepDynaQGreedyPlus, 'r', label="Greedy Dyna-Q+ N=50 K=0.000001")
plt.plot(np.arange(1, 3001), rewardsPerTimeStepDynaQGreedyPlusLN, 'b', label="Greedy Dyna-Q+ log variant N=50 K=0.000001")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Reward')
plt.title('Blocking Maze, averaged 100 times, over 3000 steps')

class DynaQGreedyPlusUCB(ControlAlgorithm):
    def __init__(self, environment, ALPHA, DISCOUNT, N, K):
        ControlAlgorithm.__init__(self, environment, ALPHA, DISCOUNT, 0)
        # model stores for each state-action pair:
        # 0 - Following state row
        # 1 - Following state column
        # 2 - Reward gotten
        # 3 - Number of times action was taken
        self.model = np.zeros((environment.MAX_ROW + 1, environment.MAX_COLUMN + 1, ACTIONS.shape[0], 4))
        
        self.triedStateActionPairs = []
        
        self.N = N
        self.K = K
        
        self.timeSteps = 0
        
    def reset(self):
        self.Q = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0]))
        self.model = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0], 4))
        self.triedStateActionPairs = []
        self.timeSteps = 0
        
    def generateEpisode(self):
        self.environment.resetState()
        
        stepCount = 0
        reward = 0
        cumulativeReward = 0

        while not np.array_equal(self.environment.state, self.environment.GOAL):
            previousState = np.copy(self.environment.state)
            
            action, actionId = self.chooseAction()
            reward = self.environment.takeAction(action)
            cumulativeReward += reward
        
            self.Q[previousState[0], previousState[1], actionId] += self.ALPHA * (reward + 
                self.DISCOUNT * np.max(self.Q[self.environment.state[0], self.environment.state[1]]) - 
                self.Q[previousState[0], previousState[1], actionId])
            
            self.model[previousState[0], previousState[1], actionId, 0] = self.environment.state[0]
            self.model[previousState[0], previousState[1], actionId, 1] = self.environment.state[1]
            self.model[previousState[0], previousState[1], actionId, 2] = reward
            if self.model[previousState[0], previousState[1], actionId, 3] == 0:
                self.triedStateActionPairs.append([previousState[0], previousState[1], actionId])
            self.model[previousState[0], previousState[1], actionId, 3] += 1
            
            # Planning
            for i in range(self.N):
                stateActionPair = random.choice(self.triedStateActionPairs)
                state = stateActionPair[0:2]
                actionId = stateActionPair[2]
                nextState = [0, 0]
                nextState[0] = int(self.model[state[0], state[1], actionId, 0])
                nextState[1] = int(self.model[state[0], state[1], actionId, 1])
                reward = self.model[state[0], state[1], actionId, 2]
                
                self.Q[state[0], state[1], actionId] += self.ALPHA * (reward + 
                    self.DISCOUNT * np.max(self.Q[nextState[0], nextState[1]]) - 
                    self.Q[state[0], state[1], actionId])
            
            stepCount += 1
            self.timeSteps += 1
        
        return stepCount, cumulativeReward
    
    def chooseAction(self): # We override the usual chooseAction() function to implement the mentionned exploration bonus
        maxValue = np.NINF
        possibleActions = []
        
        for i in range(ACTIONS.shape[0]):
            if self.model[self.environment.state[0], self.environment.state[1], i, 3] == 0:
                if maxValue != np.inf:
                    possibleActions = []
                    maxValue = np.inf
                possibleActions.append(i)
                continue
                
            bonusReward = np.sqrt(np.log(self.timeSteps) / self.model[self.environment.state[0], self.environment.state[1], i, 3])
                
            if (self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward) > maxValue:
                possibleActions = [i]
                maxValue = self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward
            elif (self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward) == maxValue:
                possibleActions.append(i)
                maxValue = self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward

        actionId = random.choice(possibleActions)
        
        return ACTIONS[actionId], actionId

UP = np.array([1, 0])
RIGHT = np.array([0, 1])
DOWN = np.array([-1, 0])
LEFT = np.array([0, -1])

ACTIONS = np.array([UP, RIGHT, DOWN, LEFT])

environment = BlockingMaze()
ALPHA = 0.1
DISCOUNT = 0.95
EPSILON = 0.1
N = 50
K = 0.000001

dqgpucb = DynaQGreedyPlusUCB(environment, ALPHA, DISCOUNT, N, K)
rewardsPerTimeStepDynaQGreedyPlusUCB = dqgpucb.averageMTimesOverNSteps(100, 3000)

plt.figure(4)
plt.plot(np.arange(1, 3001), rewardsPerTimeStepDynaQGreedyPlus, 'r', label="Greedy Dyna-Q+ N=50 K=0.000001")
plt.plot(np.arange(1, 3001), rewardsPerTimeStepDynaQGreedyPlusUCB, 'b', label="Greedy Dyna-Q+ UCB variant N=50 K=0.000001")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Reward')
plt.title('Blocking Maze, averaged 100 times, over 3000 steps')

class ShortcutMaze:
    def __init__(self):
        self.state = [0, 3]
        self.GOAL = [5, 8]
        self.MAX_ROW = 5
        self.MAX_COLUMN = 8
        self.BLOCKS = [[2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8]]
        self.stepCount = 0
        self.secondPhase = False
    
    def takeAction(self, action):
        if self.isMoveAllowed(action):
            self.state += action

        self.stepCount += 1
        if not self.secondPhase and self.stepCount > 3000:
            self.BLOCKS.pop()
            self.secondPhase = True
            
        if np.array_equal(self.state, self.GOAL):
            return 1
        else:
            return 0
    
    def isMoveAllowed(self, action):
        stateAfterAction = self.state + action
        
        if (stateAfterAction[0] < 0 or stateAfterAction[0] > self.MAX_ROW or
            stateAfterAction[1] < 0 or stateAfterAction[1] > self.MAX_COLUMN):
                return False
        
        for block in self.BLOCKS:
            if np.array_equal(stateAfterAction, block):
                return False
        
        return True
    
    def reset(self):
        self.state = [0, 3]
        self.stepCount = 0
        self.BLOCKS = [[2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8]]
        self.secondPhase = False
        
    def resetState(self):
        self.state = [0, 3]
        
    # Displays the current state of the environment in a nice colored grid
    def view(self):
        gridHeight = self.MAX_ROW
        gridWidth = self.MAX_COLUMN
        
        grid = BlockGrid(self.MAX_COLUMN + 1, self.MAX_ROW + 1, fill=(224, 224, 224))
        for block in self.BLOCKS:
            grid[gridHeight - block[0], block[1]] = (255, 255, 255)
        grid[gridHeight - self.state[0], self.state[1]] = (0, 0, 0)
        grid[gridHeight - self.GOAL[0], self.GOAL[1]] = (255, 0, 0)
        
        grid.show()

UP = np.array([1, 0])
RIGHT = np.array([0, 1])
DOWN = np.array([-1, 0])
LEFT = np.array([0, -1])

ACTIONS = np.array([UP, RIGHT, DOWN, LEFT])

environment = ShortcutMaze()
ALPHA = 0.1
DISCOUNT = 0.95
EPSILON = 0.1
N = 50
K = 0.000001

dq = DynaQ(environment, ALPHA, DISCOUNT, EPSILON, N)
dqp = DynaQPlus(environment, ALPHA, DISCOUNT, EPSILON, N, K)
dqgpucb = DynaQGreedyPlusUCB(environment, ALPHA, DISCOUNT, N, K)

rewardsPerTimeStepDynaQ = dq.averageMTimesOverNSteps(100, 3000)
rewardsPerTimeStepDynaQPlus = dqp.averageMTimesOverNSteps(100, 3000)
rewardsPerTimeStepDynaQGreedyPlusUCB = dqgpucb.averageMTimesOverNSteps(100, 3000)

plt.figure(5)
plt.plot(np.arange(1, 3001), rewardsPerTimeStepDynaQ, 'r', label="Dyna-Q N=50")
plt.plot(np.arange(1, 3001), rewardsPerTimeStepDynaQPlus, 'b', label="Dyna-Q+ N=50 K=0.000001")
plt.plot(np.arange(1, 3001), rewardsPerTimeStepDynaQGreedyPlusUCB, 'g', label="Greedy Dyna-Q+ UCB variant N=50 K=0.000001")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Time Steps')
plt.ylabel('Cumulative Reward')
plt.title('Shortcut Maze, averaged 100 times, over 3000 steps')

class ControlAlgorithm2:
    def __init__(self, environment, ALPHA, DISCOUNT, EPSILON):
        self.Q = np.zeros((environment.MAX_ROW + 1, environment.MAX_COLUMN + 1, ACTIONS.shape[0]))
        self.environment = environment
        self.ALPHA = ALPHA
        self.DISCOUNT = DISCOUNT
        self.EPSILON = EPSILON
      
    def averageMTimesOverNSteps(self, M, N):
        episodesPerTimeStep = np.zeros((M, N))
        for i in range(M):
            self.resetQ()
            episodesPerTimeStep[i] = self.runForNTimeSteps(N)

        return np.average(episodesPerTimeStep, axis=0)
    
    def resetQ(self):
        self.Q = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0]))
    
    def runForNTimeSteps(self, N):
        stepCount = 0
        episodeCount = 0
        episodesPerTimeStep = np.zeros(N)
        
        while stepCount < N:
            episodeLength = self.generateEpisode()
            
            for i in range(stepCount, stepCount + episodeLength):
                if i >= N:
                    break
                
                episodesPerTimeStep[i] = episodeCount
                
            stepCount += episodeLength
            episodeCount += 1
        
        return episodesPerTimeStep
    
    def chooseAction(self):
        if np.random.rand(1)[0] <= self.EPSILON:
            actionId = np.random.randint(ACTIONS.shape[0], size=1)[0]
        else:
            actionId = np.argmax(self.Q[self.environment.state[0], self.environment.state[1]])
            
            # If multiple actions have the same max value, we need to choose one of them randomly
            possibleActions = []
            for i in range(ACTIONS.shape[0]):
                if (self.Q[self.environment.state[0], self.environment.state[1], i] == 
                   self.Q[self.environment.state[0], self.environment.state[1], actionId]):
                    possibleActions.append(i)
                    
            actionId = random.choice(possibleActions)
        
        return ACTIONS[actionId], actionId
    
    def chooseGreedyAction(self):
        actionId = np.argmax(self.Q[self.environment.state[0], self.environment.state[1]])

        # If multiple actions have the same max value, we need to choose one of them randomly
        possibleActions = []
        for i in range(ACTIONS.shape[0]):
            if (self.Q[self.environment.state[0], self.environment.state[1], i] == 
               self.Q[self.environment.state[0], self.environment.state[1], actionId]):
                possibleActions.append(i)

        actionId = random.choice(possibleActions)

        return ACTIONS[actionId], actionId
    
    # Displays the learned policy trying to complete an episode.
    # The greedy policy is used and thus the episode might not finish if the learned policy is not trained enough.
    def viewLearnedPolicy(self):
        self.environment.reset()
        self.environment.view()
        
        reward = -1
        
        while reward == -1:
            previousState = np.copy(self.environment.state)
            
            action, actionId = self.chooseGreedyAction()
            reward = self.environment.takeAction(action)
            
            clear_output()
            self.environment.view()
            time.sleep(0.5)

class Sarsa(ControlAlgorithm2):
    def __init__(self, environment, ALPHA, DISCOUNT, EPSILON):
        ControlAlgorithm2.__init__(self, environment, ALPHA, DISCOUNT, EPSILON)
        
    def generateEpisode(self):
        self.environment.reset()
        
        stepCount = 0
        reward = -1

        action, actionId = self.chooseAction()
        
        while reward == -1:
            previousState = np.copy(self.environment.state)
            reward = self.environment.takeAction(action)
        
            nextAction, nextActionId = self.chooseAction()
        
            self.Q[previousState[0], previousState[1], actionId] += self.ALPHA * (reward + 
                self.DISCOUNT * self.Q[self.environment.state[0], self.environment.state[1], nextActionId] - 
                self.Q[previousState[0], previousState[1], actionId])
            
            action = nextAction
            actionId = nextActionId
            stepCount += 1
        
        return stepCount

class Qlearning(ControlAlgorithm2):
    def __init__(self, environment, ALPHA, DISCOUNT, EPSILON):
        ControlAlgorithm2.__init__(self, environment, ALPHA, DISCOUNT, EPSILON)
        
    def generateEpisode(self):
        self.environment.reset()
        
        stepCount = 0
        reward = -1

        while reward == -1:
            previousState = np.copy(self.environment.state)
            
            action, actionId = self.chooseAction()
            reward = self.environment.takeAction(action)
        
            self.Q[previousState[0], previousState[1], actionId] += self.ALPHA * (reward + 
                self.DISCOUNT * np.max(self.Q[self.environment.state[0], self.environment.state[1]]) - 
                self.Q[previousState[0], previousState[1], actionId])
            
            stepCount += 1
        
        return stepCount

class WindyGridWorld:
    def __init__(self):
        self.state = [3, 0]
        self.GOAL = [3, 7]
        self.MAX_ROW = 6
        self.MAX_COLUMN = 9
        self.WIND_COLUMNS = {3, 4, 5, 8}
        self.STRONG_WIND_COLUMNS = {6, 7}
        
    def takeAction(self, action):
        self.state += action + self.getWindFactor()
        
        self.adjustStateIfOutsideGrid() # Agent is not allowed outside the grid, either by moving or pushed by wind

        if np.array_equal(self.state, self.GOAL):
            return 0
        else:
            return -1
    
    def getWindFactor(self):
        if self.state[1] in self.WIND_COLUMNS:
            return UP
        elif self.state[1] in self.STRONG_WIND_COLUMNS:
            return UP * 2
        else:
            return NO_MOVEMENT
    
    def adjustStateIfOutsideGrid(self):
        if self.state[0] < 0:
            self.state[0] = 0
        elif self.state[0] > self.MAX_ROW:
            self.state[0] = self.MAX_ROW
            
        if self.state[1] < 0:
            self.state[1] = 0
        elif self.state[1] > self.MAX_COLUMN:
            self.state[1] = self.MAX_COLUMN
    
    def reset(self):
        self.state = [3, 0]
        
    # Displays the current state of the environment in a nice colored grid
    def view(self):
        gridHeight = self.MAX_ROW
        gridWidth = self.MAX_COLUMN
        
        grid = BlockGrid(self.MAX_COLUMN + 1, self.MAX_ROW + 1, fill=(224, 224, 224))
        grid[:, 3:9] = (204, 229, 255)
        grid[:, 6:8] = (153, 229, 255)
        grid[gridHeight - self.state[0], self.state[1]] = (0, 0, 0)
        grid[gridHeight - self.GOAL[0], self.GOAL[1]] = (255, 0, 0)
        
        grid.show()

class DynaQGreedyPlusUCBWindy(ControlAlgorithm2):
    def __init__(self, environment, ALPHA, DISCOUNT, N, K):
        ControlAlgorithm2.__init__(self, environment, ALPHA, DISCOUNT, 0)
        # model stores for each state-action pair:
        # 0 - Following state row
        # 1 - Following state column
        # 2 - Reward gotten
        # 3 - Number of times action was taken
        self.model = np.zeros((environment.MAX_ROW + 1, environment.MAX_COLUMN + 1, ACTIONS.shape[0], 4))
        
        self.triedStateActionPairs = []
        
        self.N = N
        self.K = K
        
        self.timeSteps = 0
        
    def reset(self):
        self.Q = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0]))
        self.model = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0], 4))
        self.triedStateActionPairs = []
        self.timeSteps = 0
        
    def generateEpisode(self):
        self.environment.reset()
        
        stepCount = 0
        reward = 0
        cumulativeReward = 0

        while not np.array_equal(self.environment.state, self.environment.GOAL):
            previousState = np.copy(self.environment.state)
            
            action, actionId = self.chooseAction()
            reward = self.environment.takeAction(action)
            cumulativeReward += reward
        
            self.Q[previousState[0], previousState[1], actionId] += self.ALPHA * (reward + 
                self.DISCOUNT * np.max(self.Q[self.environment.state[0], self.environment.state[1]]) - 
                self.Q[previousState[0], previousState[1], actionId])
            
            self.model[previousState[0], previousState[1], actionId, 0] = self.environment.state[0]
            self.model[previousState[0], previousState[1], actionId, 1] = self.environment.state[1]
            self.model[previousState[0], previousState[1], actionId, 2] = reward
            if self.model[previousState[0], previousState[1], actionId, 3] == 0:
                self.triedStateActionPairs.append([previousState[0], previousState[1], actionId])
            self.model[previousState[0], previousState[1], actionId, 3] += 1
            
            # Planning
            for i in range(self.N):
                stateActionPair = random.choice(self.triedStateActionPairs)
                state = stateActionPair[0:2]
                actionId = stateActionPair[2]
                nextState = [0, 0]
                nextState[0] = int(self.model[state[0], state[1], actionId, 0])
                nextState[1] = int(self.model[state[0], state[1], actionId, 1])
                reward = self.model[state[0], state[1], actionId, 2]
                
                self.Q[state[0], state[1], actionId] += self.ALPHA * (reward + 
                    self.DISCOUNT * np.max(self.Q[nextState[0], nextState[1]]) - 
                    self.Q[state[0], state[1], actionId])
            
            stepCount += 1
            self.timeSteps += 1
        
        return stepCount
    
    def chooseAction(self): # We override the usual chooseAction() function to implement the mentionned exploration bonus
        maxValue = np.NINF
        possibleActions = []
        
        for i in range(ACTIONS.shape[0]):
            if self.model[self.environment.state[0], self.environment.state[1], i, 3] == 0:
                if maxValue != np.inf:
                    possibleActions = []
                    maxValue = np.inf
                possibleActions.append(i)
                continue
                
            bonusReward = np.sqrt(np.log(self.timeSteps) / self.model[self.environment.state[0], self.environment.state[1], i, 3])
                
            if (self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward) > maxValue:
                possibleActions = [i]
                maxValue = self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward
            elif (self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward) == maxValue:
                possibleActions.append(i)
                maxValue = self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward

        actionId = random.choice(possibleActions)
        
        return ACTIONS[actionId], actionId

UP = np.array([1, 0])
RIGHT = np.array([0, 1])
DOWN = np.array([-1, 0])
LEFT = np.array([0, -1])
NO_MOVEMENT = np.array([0, 0]) # Used when no wind affects the agent and is a possible action in Exercice 6.7

ACTIONS = np.array([UP, RIGHT, DOWN, LEFT])

ALPHA = 0.5
DISCOUNT = 1.0
EPSILON = 0.1
N = 50
K = 0.000001
environment = WindyGridWorld()

s = Sarsa(environment, ALPHA, DISCOUNT, EPSILON)
ql = Qlearning(environment, ALPHA, DISCOUNT, EPSILON)
dqgpucb = DynaQGreedyPlusUCBWindy(environment, ALPHA, DISCOUNT, N, K)

episodesPerTimeStepSarsa = s.averageMTimesOverNSteps(100, 8000)
episodesPerTimeStepQlearning = ql.averageMTimesOverNSteps(100, 8000)
episodesPerTimeStepDynaQGreedyPlusUCB = dqgpucb.averageMTimesOverNSteps(100, 8000)

plt.figure(6)
plt.plot(np.arange(1, 8001), episodesPerTimeStepSarsa, 'r', label="Sarsa epsilon=0.1")
plt.plot(np.arange(1, 8001), episodesPerTimeStepQlearning, 'b', label="Q-learning epsilon=0.1")
plt.plot(np.arange(1, 8001), episodesPerTimeStepDynaQGreedyPlusUCB, 'g', label="Greedy Dyna-Q+ UCB variant N=50 K=0.000001")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Time Steps')
plt.ylabel('Episodes')
plt.title('Windy Gridworld, averaged 100 times, over 8000 steps')

class StochasticWindyGridWorld(WindyGridWorld):
    def __init__(self):
        WindyGridWorld.__init__(self)
    
    def getWindFactor(self):
        if self.state[1] in self.WIND_COLUMNS:
            return UP * np.random.randint(3, size=1)[0]
        elif self.state[1] in self.STRONG_WIND_COLUMNS:
            return UP * np.random.randint(3, size=1)[0] + UP
        else:
            return NO_MOVEMENT

UP_LEFT = np.array([1, -1])
UP_RIGHT = np.array([1, 1])
DOWN_RIGHT = np.array([-1, 1])
DOWN_LEFT = np.array([-1, -1])
ACTIONS = np.array([UP, RIGHT, DOWN, LEFT, UP_LEFT, UP_RIGHT, DOWN_RIGHT, DOWN_LEFT])

ALPHA = 0.5
DISCOUNT = 1.0
EPSILON = 0.1
N = 50
K = 0.000001
environment = StochasticWindyGridWorld()

s = Sarsa(environment, ALPHA, DISCOUNT, EPSILON)
ql = Qlearning(environment, ALPHA, DISCOUNT, EPSILON)
dqgpucb = DynaQGreedyPlusUCBWindy(environment, ALPHA, DISCOUNT, N, K)

episodesPerTimeStepSarsa = s.averageMTimesOverNSteps(100, 8000)
episodesPerTimeStepQlearning = ql.averageMTimesOverNSteps(100, 8000)
episodesPerTimeStepDynaQGreedyPlusUCB = dqgpucb.averageMTimesOverNSteps(100, 8000)

plt.figure(7)
plt.plot(np.arange(1, 8001), episodesPerTimeStepSarsa, 'r', label="Sarsa epsilon=0.1")
plt.plot(np.arange(1, 8001), episodesPerTimeStepQlearning, 'b', label="Q-learning epsilon=0.1")
plt.plot(np.arange(1, 8001), episodesPerTimeStepDynaQGreedyPlusUCB, 'g', label="Greedy Dyna-Q+ UCB variant N=50 K=0.000001")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Time Steps')
plt.ylabel('Episodes')
plt.title('Stochastic Windy Gridworld, 8 actions, averaged 100 times, over 8000 steps')

class TheCliff:
    def __init__(self):
        self.state = [0, 0]
        self.GOAL = [0, 11]
        self.MAX_ROW = 3
        self.MAX_COLUMN = 11
        
    def takeAction(self, action):
        self.state += action
        
        if self.state[0] == 0 and 1 <= self.state[1] <= 10:
            self.state = [0, 0]
            return -100
        
        self.adjustStateIfOutsideGrid() # Agent is not allowed outside the grid

        if np.array_equal(self.state, self.GOAL):
            return 0
        else:
            return -1
    
    def adjustStateIfOutsideGrid(self):
        if self.state[0] < 0:
            self.state[0] = 0
        elif self.state[0] > self.MAX_ROW:
            self.state[0] = self.MAX_ROW
            
        if self.state[1] < 0:
            self.state[1] = 0
        elif self.state[1] > self.MAX_COLUMN:
            self.state[1] = self.MAX_COLUMN
    
    def reset(self):
        self.state = [0, 0]
        
    # Displays the current state of the environment in a nice colored grid
    def view(self):
        gridHeight = self.MAX_ROW
        gridWidth = self.MAX_COLUMN
        
        grid = BlockGrid(self.MAX_COLUMN + 1, self.MAX_ROW + 1, fill=(224, 224, 224))
        grid[gridHeight, 1:11] = (170, 170, 170)
        grid[gridHeight - self.state[0], self.state[1]] = (0, 0, 0)
        grid[gridHeight - self.GOAL[0], self.GOAL[1]] = (255, 0, 0)
        
        grid.show()

class ControlAlgorithm3:
    def __init__(self, environment, ALPHA, DISCOUNT, EPSILON):
        self.Q = np.zeros((environment.MAX_ROW + 1, environment.MAX_COLUMN + 1, ACTIONS.shape[0]))
        self.environment = environment
        self.ALPHA = ALPHA
        self.DISCOUNT = DISCOUNT
        self.EPSILON = EPSILON
      
    def averageMSumOfRewardsOverNEpisodes(self, M, N):
        sumOfRewardsPerEpisode = np.zeros((M, N))
        for i in range(M):
            self.resetQ()
            
            for j in range(N):
                sumOfRewardsPerEpisode[i, j] = self.generateEpisode()
        
        return np.average(sumOfRewardsPerEpisode, axis=0)
    
    def averageMSumOfRewardsSmoothedOverKEpisodesOverNEpisodes(self, M, K, N):
        sumOfRewardsPerEpisode = np.zeros((M, N/K))
        for i in range(M):
            self.resetQ()
            
            kCounter = 0
            sumOfRewards = 0
            for j in range(N):
                sumOfRewards += self.generateEpisode()
                if j % K == K-1:
                    sumOfRewardsPerEpisode[i, kCounter] = sumOfRewards / K
                    sumOfRewards = 0
                    kCounter += 1
        
        return np.average(sumOfRewardsPerEpisode, axis=0)
    
    def resetQ(self):
        self.Q = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0]))
    
    def chooseAction(self):
        if np.random.rand(1)[0] <= self.EPSILON:
            actionId = np.random.randint(ACTIONS.shape[0], size=1)[0]
        else:
            actionId = np.argmax(self.Q[self.environment.state[0], self.environment.state[1]])
            
            # If multiple actions have the same max value, we need to choose one of them randomly
            possibleActions = []
            for i in range(ACTIONS.shape[0]):
                if (self.Q[self.environment.state[0], self.environment.state[1], i] == 
                   self.Q[self.environment.state[0], self.environment.state[1], actionId]):
                    possibleActions.append(i)
                    
            actionId = random.choice(possibleActions)
        
        return ACTIONS[actionId], actionId
    
    # Displays the learned policy trying to complete an episode.
    def viewLearnedPolicy(self):
        self.environment.reset()
        self.environment.view()
        
        while not np.array_equal(self.environment.state, self.environment.GOAL):
            action, actionId = self.chooseAction()
            self.environment.takeAction(action)
            
            clear_output()
            self.environment.view()
            time.sleep(0.5)

class Sarsa2(ControlAlgorithm3):
    def __init__(self, environment, ALPHA, DISCOUNT, EPSILON):
        ControlAlgorithm3.__init__(self, environment, ALPHA, DISCOUNT, EPSILON)
        
    def generateEpisode(self):
        self.environment.reset()
        
        reward = 0
        sumOfRewards = 0

        action, actionId = self.chooseAction()
        
        while not np.array_equal(self.environment.state, self.environment.GOAL):
            previousState = np.copy(self.environment.state)
            reward = self.environment.takeAction(action)
            sumOfRewards += reward
        
            nextAction, nextActionId = self.chooseAction()
        
            self.Q[previousState[0], previousState[1], actionId] += self.ALPHA * (reward + 
                self.DISCOUNT * self.Q[self.environment.state[0], self.environment.state[1], nextActionId] - 
                self.Q[previousState[0], previousState[1], actionId])
            
            action = nextAction
            actionId = nextActionId
        
        return sumOfRewards

class Qlearning2(ControlAlgorithm3):
    def __init__(self, environment, ALPHA, DISCOUNT, EPSILON):
        ControlAlgorithm3.__init__(self, environment, ALPHA, DISCOUNT, EPSILON)
        
    def generateEpisode(self):
        self.environment.reset()
        
        reward = 0
        sumOfRewards = 0

        while not np.array_equal(self.environment.state, self.environment.GOAL):
            previousState = np.copy(self.environment.state)
            
            action, actionId = self.chooseAction()
            reward = self.environment.takeAction(action)
            sumOfRewards += reward
        
            self.Q[previousState[0], previousState[1], actionId] += self.ALPHA * (reward + 
                self.DISCOUNT * np.max(self.Q[self.environment.state[0], self.environment.state[1]]) - 
                self.Q[previousState[0], previousState[1], actionId])
        
        return sumOfRewards

class ExpectedSarsa(ControlAlgorithm3):
    def __init__(self, environment, ALPHA, DISCOUNT, EPSILON):
        ControlAlgorithm3.__init__(self, environment, ALPHA, DISCOUNT, EPSILON)
        
    def generateEpisode(self):
        self.environment.reset()
        
        reward = 0
        sumOfRewards = 0

        while not np.array_equal(self.environment.state, self.environment.GOAL):
            previousState = np.copy(self.environment.state)
            
            action, actionId = self.chooseAction()
            reward = self.environment.takeAction(action)
            sumOfRewards += reward
        
            expectedNextStateActionPairValue = self.getExpectedNextStateActionPairValue()
            
            self.Q[previousState[0], previousState[1], actionId] += self.ALPHA * (reward + self.DISCOUNT *
                expectedNextStateActionPairValue - self.Q[previousState[0], previousState[1], actionId])
        
        return sumOfRewards
    
    def getExpectedNextStateActionPairValue(self):
        maxValue = np.max(self.Q[self.environment.state[0], self.environment.state[1]])
            
        # If multiple actions have the same max value
        maxValueActions = Set()
        for i in range(ACTIONS.shape[0]):
            if self.Q[self.environment.state[0], self.environment.state[1], i] == maxValue:
                maxValueActions.add(i)
        
        value = 0
        
        for i in range(ACTIONS.shape[0]):
            probability = self.EPSILON / ACTIONS.shape[0]
            if i in maxValueActions:
                probability += (1 - self.EPSILON) / len(maxValueActions)
            
            value += probability * self.Q[self.environment.state[0], self.environment.state[1], i]
        
        return value

class DynaQGreedyPlusUCBCliff(ControlAlgorithm3):
    def __init__(self, environment, ALPHA, DISCOUNT, N, K):
        ControlAlgorithm3.__init__(self, environment, ALPHA, DISCOUNT, 0)
        # model stores for each state-action pair:
        # 0 - Following state row
        # 1 - Following state column
        # 2 - Reward gotten
        # 3 - Number of times action was taken
        self.model = np.zeros((environment.MAX_ROW + 1, environment.MAX_COLUMN + 1, ACTIONS.shape[0], 4))
        
        self.triedStateActionPairs = []
        
        self.N = N
        self.K = K
        
        self.timeSteps = 0
        
    def reset(self):
        self.Q = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0]))
        self.model = np.zeros((self.environment.MAX_ROW + 1, self.environment.MAX_COLUMN + 1, ACTIONS.shape[0], 4))
        self.triedStateActionPairs = []
        self.timeSteps = 0
        
    def generateEpisode(self):
        self.environment.reset()
        
        stepCount = 0
        reward = 0
        cumulativeReward = 0

        while not np.array_equal(self.environment.state, self.environment.GOAL):
            previousState = np.copy(self.environment.state)
            
            action, actionId = self.chooseAction()
            reward = self.environment.takeAction(action)
            cumulativeReward += reward
        
            self.Q[previousState[0], previousState[1], actionId] += self.ALPHA * (reward + 
                self.DISCOUNT * np.max(self.Q[self.environment.state[0], self.environment.state[1]]) - 
                self.Q[previousState[0], previousState[1], actionId])
            
            self.model[previousState[0], previousState[1], actionId, 0] = self.environment.state[0]
            self.model[previousState[0], previousState[1], actionId, 1] = self.environment.state[1]
            self.model[previousState[0], previousState[1], actionId, 2] = reward
            if self.model[previousState[0], previousState[1], actionId, 3] == 0:
                self.triedStateActionPairs.append([previousState[0], previousState[1], actionId])
            self.model[previousState[0], previousState[1], actionId, 3] += 1
            
            # Planning
            for i in range(self.N):
                stateActionPair = random.choice(self.triedStateActionPairs)
                state = stateActionPair[0:2]
                actionId = stateActionPair[2]
                nextState = [0, 0]
                nextState[0] = int(self.model[state[0], state[1], actionId, 0])
                nextState[1] = int(self.model[state[0], state[1], actionId, 1])
                reward = self.model[state[0], state[1], actionId, 2]
                
                self.Q[state[0], state[1], actionId] += self.ALPHA * (reward + 
                    self.DISCOUNT * np.max(self.Q[nextState[0], nextState[1]]) - 
                    self.Q[state[0], state[1], actionId])
            
            stepCount += 1
            self.timeSteps += 1
        
        return cumulativeReward
    
    def chooseAction(self): # We override the usual chooseAction() function to implement the mentionned exploration bonus
        maxValue = np.NINF
        possibleActions = []
        
        for i in range(ACTIONS.shape[0]):
            if self.model[self.environment.state[0], self.environment.state[1], i, 3] == 0:
                if maxValue != np.inf:
                    possibleActions = []
                    maxValue = np.inf
                possibleActions.append(i)
                continue
                
            bonusReward = np.sqrt(np.log(self.timeSteps) / self.model[self.environment.state[0], self.environment.state[1], i, 3])
                
            if (self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward) > maxValue:
                possibleActions = [i]
                maxValue = self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward
            elif (self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward) == maxValue:
                possibleActions.append(i)
                maxValue = self.Q[self.environment.state[0], self.environment.state[1], i] + self.K * bonusReward

        actionId = random.choice(possibleActions)
        
        return ACTIONS[actionId], actionId

UP = np.array([1, 0])
RIGHT = np.array([0, 1])
DOWN = np.array([-1, 0])
LEFT = np.array([0, -1])

ACTIONS = np.array([UP, RIGHT, DOWN, LEFT])

ALPHA = 0.5
DISCOUNT = 1.0
EPSILON = 0.1
N = 50
K = 0.000001

s = Sarsa2(environment, ALPHA, DISCOUNT, EPSILON)
ql = Qlearning2(environment, ALPHA, DISCOUNT, EPSILON)
ex = ExpectedSarsa(environment, ALPHA, DISCOUNT, EPSILON)
dqgp = DynaQGreedyPlusUCBCliff(environment, ALPHA, DISCOUNT, N, K)

sumOfRewardsPerEpisodeSarsa = s.averageMSumOfRewardsOverNEpisodes(20, 500)
sumOfRewardsPerEpisodeQLearning = ql.averageMSumOfRewardsOverNEpisodes(20, 500)
sumOfRewardsPerEpisodeExpectedSarsa = ex.averageMSumOfRewardsOverNEpisodes(20, 500)
sumOfRewardsPerEpisodeDynaQGreedyPlus = dqgp.averageMSumOfRewardsOverNEpisodes(20, 500)

plt.figure(8)
plt.plot(np.arange(2, 501), sumOfRewardsPerEpisodeSarsa[1:500], 'r', label="Sarsa")
plt.plot(np.arange(2, 501), sumOfRewardsPerEpisodeQLearning[1:500], 'b', label="Q-learning")
plt.plot(np.arange(2, 501), sumOfRewardsPerEpisodeDynaQGreedyPlus[1:500], 'g', label="Greedy Dyna-Q+ UCB variant N=50 K=0.000001")
plt.plot(np.arange(2, 501), sumOfRewardsPerEpisodeExpectedSarsa[1:500], 'y', label="Expected Sarsa")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.xlabel('Episodes')
plt.ylabel('Sum of rewards during episode')
plt.title('The Cliff, averaged 20 times, over 500 episodes')
plt.show()