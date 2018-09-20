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
import random, util
import math

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
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

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
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        
        eval = 0
        
        #successor distance to food
        mindist = float("inf")
        for foodPos in newFood.asList():
            mandist = manhattanDistance(newPos, foodPos)
            if mandist < mindist:
                mindist = mandist
        eval += (1/mindist)*5
        
        #print(mindist)
        
        #number of food left
        numFoodLeft = successorGameState.getNumFood()
        curFoodLeft = currentGameState.getNumFood()
        if numFoodLeft < curFoodLeft:
            if numFoodLeft != 0:
                eval = abs(eval)*(curFoodLeft**10/numFoodLeft)
            else:
                eval += 2000
            
        #distance to capsule
        mindist = float("inf")
        for capPos in successorGameState.getCapsules():
            mandist = manhattanDistance(newPos, capPos)
            if mandist < mindist:
                mindist = mandist
        if mindist != 0:
            eval += (1/mindist)*10
        
        #number of capsules left
        numCapLeft = len(successorGameState.getCapsules())
        curCapLeft = len(currentGameState.getCapsules())
        if numCapLeft < curCapLeft:
            if numCapLeft != 0:
                eval = abs(eval)*((curCapLeft**10)/numCapLeft)
            else:
                eval += 2000
        
        #distance to ghosts
        mindist = float("inf")
        ghost = None
        for ghostPos in currentGameState.getGhostStates():
            mandist = manhattanDistance(newPos, ghostPos.getPosition())
            if mandist < mindist:
                mindist = mandist
                ghost = ghostPos 
        if ghost.scaredTimer != 0 and mindist != 0:
            eval += (1/mindist)*20
        elif mindist <= 1:
            eval -= 2000
        elif mindist != 0:
            eval -= (1/mindist)*20
            
        #stop condition
        if action == Directions.STOP:
            eval -= 10
            
        eval += successorGameState.getScore()/10
        
        if(successorGameState.isWin()):
            eval += 1000000
        
        return eval

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
        self.agentindex = 0
        self.action = 0

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
        def minmax(state, index, d, numagent):
            if state.isWin() or state.isLose() or (d == 0):
                return (self.evaluationFunction(state), None)
            
            childList = state.getLegalActions(index-1)      
                
            if index == 1: #max
                max = -float("inf")
                maxstate = None
                for child in childList:
                    if child != Directions.STOP:
                        succ = state.generateSuccessor(index-1, child)
                        x = minmax(succ,2,d,numagent)[0] 
                        if x > max:
                            max = x
                            maxstate = child
                return (max, maxstate)
            
            else: #min
                min = float("inf")
                minstate = None
                for child in childList:
                    if child != Directions.STOP:
                        succ = state.generateSuccessor(index-1, child)
                        if index == numagent:
                            x = minmax(succ,1, d-1, numagent)[0]
                        else:
                            x = minmax(succ,index+1,d, numagent)[0]
                        if x < min:
                            min = x
                            minstate = child
                return (min, minstate)
                
        numagent = gameState.getNumAgents()
        
        return minmax(gameState, 1, self.depth, numagent)[1]
            
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def ab(state, index, d, numagent, a, b):
            if state.isWin() or state.isLose() or (d == 0):
                return [self.evaluationFunction(state), None]
            
            childList = state.getLegalActions(index-1)      
                
            if index == 1: #max
                x = [-float("inf"), None]
                for child in childList:
                    succ = state.generateSuccessor(index-1, child)
                    maxa = ab(succ,2,d,numagent,a,b)
                    maxa[1] = child
                    if x[0] != maxa[0]:
                        x = max(x, maxa)
                    if a[0] != x[0]:
                        a = max(a, x)
                    if b[0] <= a[0]:
                        return x
                return x
            
            else: #min
                x = [float("inf"), None]
                for child in childList:
                    succ = state.generateSuccessor(index-1, child)
                    if index == numagent:
                        minb = ab(succ,1, d-1, numagent,a,b)
                    else:
                        minb = ab(succ,index+1,d, numagent,a,b)
                    minb[1] = child
                    if x[0] != minb[0]:
                        x = min(x, minb)
                    if b[0] != x[0]:
                        b = min(b, x)
                    if b[0] <= a[0]:
                        return x
                return x
                
        numagent = gameState.getNumAgents()
        a = [-float("inf"), None]
        b = [float("inf"), None]
        
        return ab(gameState, 1, self.depth, numagent, a, b)[1]
        
        
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
        def minmax(state, index, d, numagent):
            if state.isWin() or state.isLose() or (d == 0):
                return (self.evaluationFunction(state), None)
            
            childList = state.getLegalActions(index-1)      
                
            if index == 1: #max
                max = -float("inf")
                maxstate = None
                for child in childList:
                    if child != Directions.STOP:
                        succ = state.generateSuccessor(index-1, child)
                        x = minmax(succ,2,d,numagent)[0] 
                        if x > max:
                            max = x
                            maxstate = child
                return (max, maxstate)
            
            else: #min
                min = float("inf")
                minstate = None
                avglist = []
                for child in childList:
                    if child != Directions.STOP:
                        succ = state.generateSuccessor(index-1, child)
                        if index == numagent:
                            x = minmax(succ,1, d-1, numagent)[0]
                        else:
                            x = minmax(succ,index+1,d, numagent)[0]
                        avglist.append(x)
                        if x < min:
                            min = x
                            minstate = child
                return (float(sum(avglist)/len(avglist)), minstate)
                
        numagent = gameState.getNumAgents()
        
        return minmax(gameState, 1, self.depth, numagent)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Various factors influenced my choice for conditions upon which the eval number retuned was based.  The deciding factors that I thought best affected a PacMan's choice of state were distancfe to nearest food, amount of food left, distance to nearest capsule, number of capsules left, distance to nearest scared ghost, distance to nearest ghost, the overall score, and if the state is a winning state.  The weights for each condition contributing to the total eval number were found through experimentation with the autograder.  For the ghost condition, I made PacMan advance towards a ghost that was scared while it remained scared.  I made pacman not go towards a ghost, and if PacMan was close to a close (within 1 unit of ghost), I then forced him to go the other way.  All of these conditions together ensured a score average of above 1000 which was required. 
    """
    "*** YOUR CODE HERE ***"
    
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    
    eval = 0
    
    #successor distance to food
    mindist = float("inf")
    for foodPos in food.asList():
        mandist = manhattanDistance(pos, foodPos)
        if mandist < mindist:
            mindist = mandist
    eval += (1/mindist)*5
    #number of food left

    curFoodLeft = currentGameState.getNumFood()
    if curFoodLeft != 0:
        eval += (1/curFoodLeft)*10
    else:
        eval += 2000
        
    #distance to capsule
    mindist = float("inf")
    for capPos in currentGameState.getCapsules():
        mandist = manhattanDistance(pos, capPos)
        if mandist < mindist:
            mindist = mandist
    if mindist != 0:
        eval += (1/mindist)*10
    
    #number of capsules left
    curCapLeft = len(currentGameState.getCapsules())
    if curCapLeft != 0:
        eval += (1/curCapLeft)*5

    
    #distance to ghosts
    mindist = float("inf")
    ghost = None
    for ghostPos in currentGameState.getGhostStates():
        mandist = manhattanDistance(pos, ghostPos.getPosition())
        if mandist < mindist:
            mindist = mandist
            ghost = ghostPos 
    if ghost.scaredTimer != 0 and mindist != 0:
        eval += (1/mindist)*20
    elif mindist <= 1:
        eval -= 2000
    elif mindist != 0:
        eval -= (1/mindist)*20
        
    #score    
        
    eval += currentGameState.getScore()*10
    
    if(currentGameState.isWin()):
        eval += 1000000
    
    return eval
    
# Abbreviation
better = betterEvaluationFunction

