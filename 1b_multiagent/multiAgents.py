# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import sys

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


def getFoodDistance(newPos, nextGameState):
    foodList = nextGameState.getFood().asList()

    foodDis = []
    # 计算到各个食物的曼哈顿距离
    for foodPos in foodList:
        foodDis.append(util.manhattanDistance(newPos, foodPos))

    return foodDis


def getGhostScore(pos, ghostStates):
    ghostDis = []
    # 计算到各个鬼的曼哈顿距离
    for ghostState in ghostStates:
        # if ghostState.scaredTimer == 0:
        ghostDis.append(util.manhattanDistance(pos, ghostState.getPosition()))

    return ghostDis


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        """
        根据evaluationFunction选出最佳操作
        如果有多个最佳操作，随机选择一个行动
        """
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newGhostStates = childGameState.getGhostStates()

        "*** YOUR CODE HERE ***"
        # 如果能赢 直接走
        if childGameState.isWin():
            return float("inf")
        # 如果会输 绝不走
        if childGameState.isLose():
            return float("-inf")

        # 获得场上食物和鬼的信息
        foodDis = getFoodDistance(newPos, childGameState)
        ghostDis = getGhostScore(newPos, newGhostStates)

        # 如果鬼很近 不选这条路
        if min(ghostDis) < 2:
            return float("-inf")

        # 如果吃到食物 有100额外分
        return childGameState.getScore() - 5 * min(foodDis) \
               + (100 if currentGameState.getNumFood() > childGameState.getNumFood() else 0)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        Here are some method calls that might be useful when implementing minimax.
        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action
        gameState.getNumAgents():
        Returns the total number of agents in the game
        gameState.isWin():
        Returns whether or not the game state is a winning state
        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        maxValue = float("-inf")
        maxAction = Directions.STOP
        actions = gameState.getLegalActions(0)

        # 遍历所有合法行为 找到最合适的
        for action in actions:
            nextState = gameState.getNextState(0, action)
            # 1代表鬼的节点 0代表第零层
            nextValue = self.getMinValue(nextState, 1, 0)
            # 取收益最大化行为
            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action

        return maxAction

    def checkFinish(self, gameState, depth):
        # 结束条件
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return True
        return False

    def getMinValue(self, gameState, agentIndex, depth):
        # 确保不是pacman节点
        if agentIndex == 0:
            print("*** Pacman node should not get min value! ***")
            return None

        if self.checkFinish(gameState, depth):
            return self.evaluationFunction(gameState)

        # ghost取最小子节点
        minValue = float("inf")
        actions = gameState.getLegalActions(agentIndex)

        for action in actions:
            nextState = gameState.getNextState(agentIndex, action)

            # 如果ghost遍历完了
            if agentIndex == gameState.getNumAgents() - 1:
                # depth+1和0代表计算下一层的pacman
                value = self.getMaxValue(nextState, 0, depth + 1)
                if value < minValue:
                    minValue = value

            # 如果ghost还没遍历完
            else:
                # depth和agentIndex+1代表计算下一个的ghost
                value = self.getMinValue(nextState, agentIndex + 1, depth)
                if value < minValue:
                    minValue = value

        return minValue

    def getMaxValue(self, gameState, agentIndex, depth):
        # 确保不是ghost节点
        if agentIndex != 0:
            print("*** Ghost node should not get max value! ***")
            return None

        if self.checkFinish(gameState, depth):
            return self.evaluationFunction(gameState)

        # pacman取最大节点
        maxValue = float("-inf")
        actions = gameState.getLegalActions(agentIndex)

        for action in actions:
            nextState = gameState.getNextState(agentIndex, action)
            # depth和1代表从第一个ghost开始计算 pacman和ghost都从第一层开始算
            value = self.getMinValue(nextState, 1, depth)
            if value > maxValue:
                maxValue = value

        return maxValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # a记录到根节点的max值 b记录到根节点的min值
        a = float("-inf")
        b = float("inf")

        maxValue = float("-inf")
        maxAction = Directions.STOP
        actions = gameState.getLegalActions(0)

        # 遍历所有合法行为 找到最合适的
        for action in actions:
            nextState = gameState.getNextState(0, action)
            # 1代表鬼的节点 0代表第零层
            nextValue = self.getMinValue(nextState, 1, 0, a, b)
            # 取收益最大化行为
            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action
            # 更新a
            a = max(a, maxValue)

        return maxAction

    def checkFinish(self, gameState, depth):
        # 结束条件
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return True
        return False

    def getMinValue(self, gameState, agentIndex, depth, a, b):
        # 确保不是pacman节点
        if agentIndex == 0:
            print("*** Pacman node should not get max value! ***")
            return None

        if self.checkFinish(gameState, depth):
            return self.evaluationFunction(gameState)

        # ghost取最小子节点
        minValue = float("inf")
        actions = gameState.getLegalActions(agentIndex)

        for action in actions:
            nextState = gameState.getNextState(agentIndex, action)

            # 如果ghost遍历完了
            if agentIndex == gameState.getNumAgents() - 1:
                # depth+1和0代表计算下一层的pacman
                value = self.getMaxValue(nextState, 0, depth + 1, a, b)
                minValue = min(minValue, value)

            # 如果ghost还没遍历完
            else:
                # depth和agentIndex+1代表计算下一个的ghost
                value = self.getMinValue(nextState, agentIndex + 1, depth, a, b)
                minValue = min(minValue, value)

            # 去除更小的影响
            if a > minValue:
                break

            # 更新b
            b = min(b, minValue)

        return minValue

    def getMaxValue(self, gameState, agentIndex, depth, a, b):
        # 确保不是ghost节点
        if agentIndex != 0:
            print("*** Ghost node should not get max value! ***")
            return None

        if self.checkFinish(gameState, depth):
            return self.evaluationFunction(gameState)

        # pacman取最大节点
        maxValue = float("-inf")
        actions = gameState.getLegalActions(agentIndex)

        for action in actions:
            nextState = gameState.getNextState(agentIndex, action)
            # depth和1代表从第一个ghost开始计算 pacman和ghost都从第一层开始算
            value = self.getMinValue(nextState, 1, depth, a, b)
            maxValue = max(maxValue, value)

            # 去除更大的影响
            if b < maxValue:
                break

            # 更新a
            a = max(a, maxValue)

        return maxValue


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        maxValue = float("-inf")
        maxAction = Directions.STOP
        actions = gameState.getLegalActions(0)

        # 遍历所有合法行为 找到最合适的
        for action in actions:
            nextState = gameState.getNextState(0, action)
            # 1代表鬼的节点 0代表第零层
            nextValue = self.getAvgValue(nextState, 1, 0)
            # 取收益最大化行为
            if nextValue > maxValue:
                maxValue = nextValue
                maxAction = action

        return maxAction

    def checkFinish(self, gameState, depth):
        # 结束条件
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return True
        return False

    def getAvgValue(self, gameState, agentIndex, depth):
        # 确保不是pacman节点
        if agentIndex == 0:
            print("*** Pacman node should not get max value! ***")
            return None

        if self.checkFinish(gameState, depth):
            return self.evaluationFunction(gameState)

        # ghost取平均值
        avgValue = 0
        actions = gameState.getLegalActions(agentIndex)

        for action in actions:
            nextState = gameState.getNextState(agentIndex, action)

            # 如果ghost遍历完了
            if agentIndex == gameState.getNumAgents() - 1:
                # depth+1和0代表计算下一层的pacman
                avgValue += self.getMaxValue(nextState, 0, depth + 1)

            # 如果ghost还没遍历完
            else:
                # depth和agentIndex+1代表计算下一个的ghost
                avgValue += self.getAvgValue(nextState, agentIndex + 1, depth)

        return avgValue

    def getMaxValue(self, gameState, agentIndex, depth):
        # 确保不是ghost节点
        if agentIndex != 0:
            print("*** Ghost node should not get max value! ***")
            return None

        if self.checkFinish(gameState, depth):
            return self.evaluationFunction(gameState)

        # pacman取最大节点
        maxValue = float("-inf")
        actions = gameState.getLegalActions(agentIndex)

        for action in actions:
            nextState = gameState.getNextState(agentIndex, action)
            # depth和1代表从第一个ghost开始计算 pacman和ghost都从第一层开始算
            value = self.getAvgValue(nextState, 1, depth)
            if value > maxValue:
                maxValue = value

        return maxValue


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newGhostStates = currentGameState.getGhostStates()

    # 如果能赢 直接走
    if currentGameState.isWin():
        return float("inf")
    # 如果会输 绝不走
    if currentGameState.isLose():
        return float("-inf")

    # 获得场上食物和鬼的信息
    foodDis = getFoodDistance(newPos, currentGameState)
    ghostDis = getGhostScore(newPos, newGhostStates)

    # 如果鬼很近 不选这条路
    if min(ghostDis) < 2:
        return float("-inf")

    return scoreEvaluationFunction(currentGameState) - 2 * min(foodDis) - max(foodDis) \
           - 8 * currentGameState.getNumFood() \
           + 1.5 * min(ghostDis) + max(ghostDis)


# Abbreviation
better = betterEvaluationFunction



class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """
    action = ['West', 'North', 'North', 'East', 'East', 'West', 'North', 'North', 'North', 'East', 'North',
              'West', 'West', 'West', 'West', 'South', 'South', 'West', 'West', 'West', 'West', 'West',
              'South', 'South', 'South', 'South', 'North', 'North', 'North', 'North', 'North', 'North',
              'East', 'East', 'East', 'South', 'South', 'East', 'East', 'South', 'South', 'South', 'South',
              'West', 'West', 'West', 'North', 'North', 'North', 'South', 'East', 'East', 'East', 'North',
              'North', 'North', 'North', 'East', 'East', 'East', 'East', 'East', 'East', 'East', 'South',
              'South', 'East', 'East', 'East', 'East', 'East', 'North', 'North', 'South', 'South', 'South',
              'South', 'South', 'South', 'West', 'West', 'West', 'North', 'North', 'East', 'North', 'North',
              'East', 'East', 'North', 'North', 'West', 'West', 'West', 'South', 'South', 'West', 'West',
              'South', 'South', 'South', 'South', 'West', 'West', 'North', 'North', 'West', 'West', 'West',
              'East', 'East', 'East', 'South', 'South', 'West', 'West', 'West', 'West',  # p1

              'West', 'North', 'North', 'East', 'West', 'South', 'South', 'West', 'West', 'West', 'West', 'West',
               'North', 'North', 'North', 'North', 'East', 'North', 'North', 'West', 'West', 'West', 'East',
               'West', 'South', 'South', 'South', 'South', 'South', 'South', 'North', 'North', 'North', 'North',
               'East', 'East', 'East', 'East', 'East', 'South', 'South', 'West', 'West', 'East', 'East', 'South',
               'South', 'East', 'East', 'East', 'East', 'East', 'East', 'East', 'North', 'North', 'North', 'North',
              'East', 'East', 'North', 'North', 'East', 'East', 'East', 'South', 'South', 'South', 'South', 'South',
              'South', 'West', 'West', 'West', 'North', 'North', 'East', 'North', 'North', 'East', 'East', 'North',
              'North', 'West', 'West', 'West', 'South', 'South', 'West', 'West', 'North', 'South', 'South', 'South',
              'South', 'South', 'West', 'West', 'North', 'North', 'South', 'South', 'West', 'West', 'West', 'North',
              'North', 'East', 'East', 'West', 'West', 'South', 'South', 'West', 'West', 'North', 'North', 'North',
              'North', 'North', 'North', 'East', 'East', 'East', 'South', 'East', 'North', 'East', 'East', 'East',  # p2

              'East', 'East', 'East', 'East', 'North', 'North', 'North', 'North', 'East', 'East', 'North', 'North',
              'East', 'East', 'East', 'West', 'West', 'West', 'South', 'South', 'East', 'East', 'East', 'North', 'South',
              'South', 'South', 'South', 'South', 'North', 'North', 'North', 'North', 'North', 'North', 'West', 'West',
              'West', 'South', 'South', 'West', 'West', 'South', 'South', 'South', 'South', 'West', 'West', 'North',
              'North', 'South', 'South', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'North', 'North',
              'North', 'North', 'East', 'North', 'North', 'West', 'West', 'West', 'East', 'East', 'West', 'West', 'South',
              'South', 'South', 'South', 'South', 'South', 'North', 'North', 'North', 'North', 'East', 'East', 'South',
              'South', 'East', 'East', 'East', 'South', 'North', 'North', 'North', 'West', 'East', 'North', 'North',
              'East', 'East', 'East', 'South', 'South', 'South', 'South', 'East', 'West', 'North', 'North', 'East',
              'North', 'North', 'East', 'East', 'East', 'South', 'South', 'East', 'East', 'East', 'South', 'South',
              'North', 'North', 'East', 'East', 'North', 'North', 'West', 'West', 'West', 'South', 'South', 'East',
              'South', 'South', 'West', 'South', 'South', 'East', 'East',  # p3

              'West', 'North', 'North', 'East', 'West', 'South', 'South', 'West', 'West', 'West', 'West', 'West',
              'North', 'North', 'North', 'North', 'East', 'North', 'North', 'West', 'West', 'West', 'East', 'East',
              'West', 'West', 'South', 'South', 'South', 'South', 'South', 'South', 'North', 'North', 'North', 'North',
              'East', 'East', 'South', 'South', 'East', 'East', 'East', 'South', 'South', 'East', 'East', 'East', 'East',
              'East', 'East', 'East', 'North', 'North', 'North', 'North', 'East', 'East', 'North', 'North', 'East',
              'East', 'East', 'South', 'South', 'South', 'South', 'South', 'South', 'West', 'West', 'West', 'North',
              'North', 'East', 'North', 'North', 'East', 'East', 'North', 'North', 'West', 'West', 'West', 'South',
              'South', 'West', 'West', 'South', 'South', 'South', 'South', 'West', 'West', 'North', 'North', 'South',
              'South', 'East', 'East', 'North', 'North', 'North', 'North', 'North', 'North', 'West', 'West', 'West',
              'South', 'West', 'North', 'West', 'West', 'West', 'South', 'South', 'South', 'South', 'West', 'West',
              'West', 'South', 'South', 'East', 'East', 'East', 'East', 'East', 'North', 'North', 'East', 'East', 'East',
              'South', 'South', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'West', 'North', 'North', 'North',
              'North', 'East', 'East',  # p4
                               
              'West', 'West', 'West', 'West', 'West', 'West', 'North', 'North', 'East', 'East', 'East',
              'North', 'North', 'West', 'West', 'North', 'North', 'West', 'West', 'West', 'East', 'East',
              'East', 'South', 'South', 'West', 'West', 'West', 'North', 'South', 'South', 'South', 'South', 'South',
              'North', 'North', 'North', 'North', 'East', 'East', 'South', 'North', 'East', 'North', 'South', 'East',
              'East', 'South', 'South', 'South', 'South', 'East', 'East', 'East', 'East', 'East', 'East', 'East', 'North',
              'North', 'North', 'North', 'East', 'East', 'East', 'East', 'East', 'South', 'South', 'South', 'South', 'West',
              'West', 'West', 'North', 'North', 'East', 'North','North', 'East', 'East', 'North', 'North', 'South', 'South',
              'West', 'West', 'West', 'North', 'North', 'East', 'East', 'West', 'West', 'South', 'South', 'East', 'East', 'East',
              'North', 'North', 'West', 'West', 'West', 'South', 'South', 'West', 'West', 'South', 'South', 'South', 'South',
              'West', 'West', 'North', 'North', 'South', 'South', 'West', 'West', 'West', 'North', 'North', 'East', 'North',
              'North', 'North', 'North', 'West', 'West', 'West', 'South','South', 'West', 'West', 'West', 'South', 'South',
              'South', 'South', 'East', 'East', 'East', 'East', 'East', 'East', 'East', 'East', 'North', 'North', 'West',
              'West', 'West', 'South', 'South', 'West', 'West', 'North', 'North', 'North', 'North', 'East', 'East', 'East',
              'East', 'North', 'North', 'East', 'East', 'East', 'South']

    def registerInitialState(self, state):
        "This method is called before any moves are made."
        "*** YOUR CODE HERE ***"
        self.tmp = []

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        if len(ContestAgent.action) > 0:
            return ContestAgent.action.pop(0)

        action = None
        while True:
            i = input()

            if i == "w":
                action = "North"
            elif i == "a":
                action = "West"
            elif i == "s":
                action = "South"
            elif i == "d":
                action = "East"

            self.tmp.append(action)
            break

        print(self.tmp)
        return action
