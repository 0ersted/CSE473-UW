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


from util import manhattanDistance
from game import Directions
from game import Actions
import random, util
import sys

from game import Agent

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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        #print"move: ", legalMoves
        #print"score: ", scores
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        #print "my move: ", legalMoves[chosenIndex]

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        Pos = currentGameState.getPacmanPosition()
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        walls = successorGameState.getWalls()
        m,n = walls.height, walls.width

        "*** YOUR CODE HERE ***"
        foodList = currentGameState.getFood().asList()
        alpha = (m-2)*(n-2)
        value = 0.0
        if(action == 'Stop'):
            value -= 1.1
        nearestFood = (-1,-1)
        foodDis = m*n
        for food in foodList:
            tmpdis = manhattanDistance(Pos, food)
            if(tmpdis < foodDis or nearestFood == (-1,-1)):
                foodDis = tmpdis
                nearestFood = food

        #print "nearest food: ", nearestFood 
        #if nearestFood != (-1,-1):
        dis = manhattanDistance(newPos, nearestFood)
        #print "next distance: ", dis 

        for ghost in newGhostStates:
            ghostDis = manhattanDistance(newPos, ghost.getPosition())
            scared = ghost.scaredTimer
            if scared == 0:
                if ghostDis <= 3:
                    value += -(4-ghostDis)**2
                else:
                    if currentGameState.hasFood(newPos[0], newPos[1]):
                        value += 1.5
            else:
                value += scared/(ghostDis+1)

        value = value - dis 
        #print"value", value
        return  value

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()
        self.agentNum = gameState.getNumAgents()

        successors = []
        for move in legalMoves:
            # a list of gamestate
            successors.append(gameState.generateSuccessor(0, move))

        scores = [self.minValue(self.depth, successor, 1) for successor in successors]
        #print"scores: ", scores
        bestScore = max(scores)
        #print"best score: ", bestScore
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        #print"bestIndices: ", bestIndices
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        #print"my move: ", legalMoves[chosenIndex]

        return legalMoves[chosenIndex]

    # pacman's move 
    def maxValue(self, currentDepth, gameState):
        if (currentDepth == 0 or gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions()
        successors = []
        for move in legalMoves:
            # a list of gamestate
            successors.append(gameState.generateSuccessor(0, move))
        value = [self.minValue(currentDepth, successor, 1) for successor in successors]
        return max(value)

    # ghosts' move
    def minValue(self, currentDepth, gameState, agentIndex):
        if (currentDepth == 0 or gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(agentIndex)
        successors = []
        for move in legalMoves:
            # a list of gamestate
            successors.append(gameState.generateSuccessor(agentIndex, move))
        if (agentIndex+1 < self.agentNum):
            value = [self.minValue(currentDepth, successor, agentIndex+1) for successor in successors]
        else:
            value = [self.maxValue(currentDepth-1, successor) for successor in successors]
        return min(value) 

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        legalMoves = gameState.getLegalActions()
        self.agentNum = gameState.getNumAgents()

        alpha = float("-inf")
        beta = float("inf")
        bestValue = -float("inf")
        bestMove = ""

        for move in legalMoves:
            successor = gameState.generateSuccessor(0, move)
            value = self.minValue(self.depth, successor, 1, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestMove = move
                #print "current best value: ", value, "best move: ", move
            if value > beta:
                return bestMove
            alpha = max(alpha, bestValue)
        #print "bestMove: ", bestMove
        return bestMove


    # pacman's move 
    def maxValue(self, currentDepth, gameState, alpha, beta):
        if (currentDepth == 0 or gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)

        value = -float("inf")
        legalMoves = gameState.getLegalActions()

        for move in legalMoves:
            # a list of gamestate
            successor = gameState.generateSuccessor(0, move)
            value = max(value, self.minValue(currentDepth, successor, 1, alpha, beta))
            if value > beta:
                #print "value: ", value, "beta: ", beta
                return value
            alpha = max(alpha, value)
        return value

    # ghosts' move
    def minValue(self, currentDepth, gameState, agentIndex, alpha, beta):
        if (currentDepth == 0 or gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(agentIndex)
        value = float("inf") 

        for move in legalMoves:
            successor = gameState.generateSuccessor(agentIndex, move)
            # a list of gamestate
            if (agentIndex+1 < self.agentNum):
                value = min(value, self.minValue(currentDepth, successor, agentIndex+1, alpha, beta))
            else:
                value = min(value, self.maxValue(currentDepth-1, successor, alpha, beta))
            #print"in minValue"
            #print "value: ", value, "alpha: ", alpha 
            if value < alpha:
                return value
            beta = min(beta, value)
        return value

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
        legalMoves = gameState.getLegalActions()
        self.agentNum = gameState.getNumAgents()

        successors = []
        for move in legalMoves:
            # a list of gamestate
            successors.append(gameState.generateSuccessor(0, move))

        scores = [self.expValue(self.depth, successor, 1) for successor in successors]
        #print"scores: ", scores
        bestScore = max(scores)
        #print"best score: ", bestScore
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        #print"bestIndices: ", bestIndices
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        #print"my move: ", legalMoves[chosenIndex]

        return legalMoves[chosenIndex]

    def maxValue(self, currentDepth, gameState):
        if (currentDepth == 0 or gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions()
        successors = []
        for move in legalMoves:
            # a list of gamestate
            successors.append(gameState.generateSuccessor(0, move))
        value = [self.expValue(currentDepth, successor, 1) for successor in successors]
        return max(value)

    def expValue(self, currentDepth, gameState, agentIndex):
        if (currentDepth == 0 or gameState.isLose() or gameState.isWin()):
            return self.evaluationFunction(gameState)

        legalMoves = gameState.getLegalActions(agentIndex)
        value = 0.0
        for move in legalMoves:
            # a list of gamestate
            successor = gameState.generateSuccessor(agentIndex, move)
            if (agentIndex+1 < self.agentNum):
                value += self.expValue(currentDepth, successor, agentIndex+1)
            else:
                value += self.maxValue(currentDepth-1, successor) 
        return value/len(legalMoves)

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      Observed that in some general cases of q1, the pacman sometimes stops at
      the wall while there is a food on the other side of the wall. Hence it is
      not quite reasonable to evaluate by Manhattan distance. We should do it
      with a general search method. In the following implementation, we use
      A* search to find the actual distance in the maze
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    ghostState = currentGameState.getGhostStates()
    walls = currentGameState.getWalls()

    foodWithDis = []
    value = 0
    bestDis = 0

    for food in foodList:
        foodWithDis.append((food, manhattanDistance(food, pos)))
    foodWithDis.sort(key=getDis)

    for i in range(0, len(foodWithDis)):
        dis = aStarSearch(pos, foodWithDis[i][0], walls)
        if i == 0 or bestDis > dis:
            bestIndex = i
            bestDis = dis
        else:
            break

    for ghost in ghostState:
        ghostDis = aStarSearch(ghost.getPosition(), pos, walls)
        scared = ghost.scaredTimer
        if scared == 0:
            if ghostDis <= 3:
                value -= (3.5-ghostDis)**2 + 1
        else:
            value += scared/(ghostDis+1)

    value = value - bestDis + scoreEvaluationFunction(currentGameState)
    """
    important to add score to final value because pacman should eat food as
    early as possible.
    """
    return value

def aStarSearch(pos, foodPos, walls):
    """Search the node that has the lowest combined cost and heuristic first."""

    start = pos
    state = (start, 0, 0)
    visitedDict = {}
    fatherDict = {}
    path = []
    heap = util.PriorityQueue()
    goal = foodPos
    heap.push((state, 0, manhattanDistance(start, goal)), 0) # ((child_info, father, accumulated cost)

    while( not heap.isEmpty() ):
        node = heap.pop()
        while visitedDict.has_key(node[0][0]):
            node = heap.pop()
        #print "node: ", node
        (info, father, cost) = node
        state = info[0]
        if father != 0:
            fatherDict[state] = father
        visitedDict[state] = info
        #print "cost: ", cost 
        
        if state == goal:
            #print "visitedDict: ", visitedDict 
            while (state != start):
                path.append(visitedDict[state])
                state = fatherDict[state] 

            return len(path)

        succs = getSuccessors(state, walls)
        #print "succs: ",succs
        for succ in succs:
            if not visitedDict.has_key(succ[0]):
                heap.push((succ, state,
                    succ[2]+cost),succ[2]+cost+manhattanDistance(succ[0], goal))

def getSuccessors(pos, walls):
    successors = []
    for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
        x,y = pos
        dx, dy = Actions.directionToVector(action)
        nextx, nexty = int(x + dx), int(y + dy)
        if not walls[nextx][nexty]:
            nextState = (nextx, nexty)
            successors.append( (nextState, action, 1) )
    return successors

def getDis(foodWithDis):
    return foodWithDis[1]

# Abbreviation
better = betterEvaluationFunction

