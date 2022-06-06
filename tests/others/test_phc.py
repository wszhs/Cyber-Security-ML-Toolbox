import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def generateRandomFromDistribution (distribution):
    randomIndex = 0
    randomSum = distribution[randomIndex]
    randomFlag = np.random.random_sample()
    while randomFlag > randomSum:
        randomIndex += 1
        randomSum += distribution[randomIndex]
    return randomIndex

class agent:
    def __init__(self, initialStrategy = (0.5, 0.5), gammma = 0.9, delta = 0.0001):
        self.timeStep = 0
        self.alpha = 1 / (10 + 0.00001 * self.timeStep)
        self.gamma = gammma
        self.actions = [0, 1]
        self.lengthOfAction = len(self.actions)
        self.reward = 0.0
        self.strategy = list(initialStrategy)
        self.actionValues = np.zeros((self.lengthOfAction)) # Q(s, a)
        self.currentAction = np.random.choice(self.actions)
        self.currentReward = 0
        self.maxAction = np.random.choice(self.actions)
        self.EPSILON = 0.5 / (1 + 0.0001 * self.timeStep)
        self.deltaAction = np.zeros((self.lengthOfAction))
        self.deltaActionTop = np.zeros((self.lengthOfAction))
        self.delta = delta
        #WoLF parameters
        self.stateCount = 0.0
        self.averageStrategy = np.zeros((self.lengthOfAction))
        for i in self.actions:
            self.averageStrategy[i] = 1.0 /self.lengthOfAction
        self.deltaWin = 0.0
        self.deltaLose = 0.0

    def initialSelfStrategy (self):
        for i in range(self.lengthOfAction):
            self.strategy[i] = 1.0 / self.lengthOfAction

    def initialActionValues (self):
        for i in range(self.lengthOfAction):
            self.actionValues[i] = 0

    def chooseAction (self):
        if np.random.binomial(1, self.EPSILON) == 1:
            self.currentAction = np.random.choice(self.actions)
        else:
            self.currentAction = self.actions[generateRandomFromDistribution(self.strategy)]

    def chooseActionWithFxiedStrategy (self):
        self.currentAction = self.actions[generateRandomFromDistribution(self.strategy)]

    def getCurrentAction (self):
        return self.currentAction

    def setReward (self, agentReward):
        self.currentReward = agentReward

    def updateActionValues (self):
        self.actionValues[self.currentAction] = (1 - self.alpha) * self.actionValues[self.currentAction] \
                                                 + self.alpha * (self.currentReward + self.gamma * np.amax(self.actionValues[:]))
    def updateStrategy (self):
        self.stateCount += 1.0
        self.deltaWin = 1.0 / (20000 + self.timeStep)
        self.deltaLose = 2.0 * self.deltaWin
        for action_i in self.actions:
            self.averageStrategy[action_i] += (1.0 / self.stateCount) * (self.strategy[action_i] - self.averageStrategy[action_i])
        self.sumActionValue = 0.0
        self.sumAverageActionValue = 0.0
        for action_i in self.actions:
            self.sumActionValue += self.strategy[action_i] * self.actionValues[action_i]
            self.sumAverageActionValue += self.averageStrategy[action_i] * self.actionValues[action_i]
        if self.sumActionValue > self.sumAverageActionValue:
            self.delta = self.deltaWin
        else:
            self.delta = self.deltaLose

        self.maxAction = np.argmax(self.actionValues)
        for i in range(self.lengthOfAction):
            self.deltaAction[i] = np.min([self.strategy[i], self.delta / (self.lengthOfAction - 1)])
        self.sumDeltaAction = 0
        for action_i in [action_j for action_j in self.actions if action_j != self.maxAction]:
            self.deltaActionTop[action_i] = -self.deltaAction[action_i]
            self.sumDeltaAction += self.deltaAction[action_i]
        self.deltaActionTop[self.maxAction] = self.sumDeltaAction
        for i in range(self.lengthOfAction):
            self.strategy[i] += self.deltaActionTop[i]

        # if self.currentAction != self.maxAction:
        #     self.deltaActionTop[self.currentAction] = -self.deltaAction[self.currentAction]
        # else:
        #     self.sumDeltaAction = 0
        #     for action_i in [action_j for action_j in self.actions if action_j != self.currentAction]:
        #         self.sumDeltaAction += self.deltaAction[action_i]
        #     self.deltaActionTop[self.currentAction] = self.sumDeltaAction
        # self.strategy[self.currentAction] += self.deltaActionTop[self.currentAction]

    def updateTimeStep (self):
        self.timeStep += 1

    def updateEpsilon (self):
        self.EPSILON = 0.5 / (1 + 0.0001 * self.timeStep)

    def updateAlpha (self):
        self.alpha = 1 / (10 + 0.00001 * self.timeStep)



def calReward (action_1, action_2):
    reward_1 = 0
    reward_2 = 0
    if (action_1, action_2) == (0, 0):
        reward_1 = 1
        reward_2 = -1
    elif (action_1, action_2) == (1, 0):
        reward_1 = -1
        reward_2 = 1
    elif (action_1, action_2) == (0, 1):
        reward_1 = -1
        reward_2 = 1
    elif (action_1, action_2) == (1, 1):
        reward_1 = 1
        reward_2 = -1
    return (reward_1, reward_2)

def figureBothLearning():
    a = agent(initialStrategy=(0.8, 0.2))
    b = agent(initialStrategy=(0.2, 0.8))
    time = 0
    aStrategyActionZero = []
    aStrategyActionZero.append(a.strategy[0])
    while (time < 100):
        a.chooseAction()
        actionA = a.getCurrentAction()
        b.chooseAction()
        actionB = b.getCurrentAction()
        rewardA, rewardB = calReward(actionA, actionB)
        a.setReward(rewardA)
        b.setReward(rewardB)
        a.updateActionValues()
        b.updateActionValues()
        a.updateStrategy()
        b.updateStrategy()
        a.updateTimeStep()
        b.updateTimeStep()
        a.updateEpsilon()
        b.updateEpsilon()
        a.updateAlpha()
        b.updateAlpha()
        time += 1
        aStrategyActionZero.append(a.strategy[0])
    plt.figure(1)
    plt.plot(aStrategyActionZero)
    plt.xlabel('timestep')
    plt.ylabel('probability')

def figureALearningBNS():
    a = agent(initialStrategy=(0.8, 0.2))
    b = agent(initialStrategy=(0.7, 0.3))
    time = 0
    aStrategyActionZero = []
    aStrategyActionZero.append(a.strategy[0])
    while (time < 1000):
        a.chooseAction()
        actionA = a.getCurrentAction()
        b.chooseActionWithFxiedStrategy()
        actionB = b.getCurrentAction()
        rewardA, rewardB = calReward(actionA, actionB)
        a.setReward(rewardA)
        a.updateActionValues()
        a.updateStrategy()
        a.updateTimeStep()
        a.updateEpsilon()
        a.updateAlpha()
        time += 1
        aStrategyActionZero.append(a.strategy[0])
    plt.figure(2)
    plt.plot(aStrategyActionZero)
    plt.xlabel('timestep')
    plt.ylabel('probability')


# figureBothLearning()
figureALearningBNS()
plt.show()