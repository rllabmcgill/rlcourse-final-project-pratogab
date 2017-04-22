import numpy as np

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