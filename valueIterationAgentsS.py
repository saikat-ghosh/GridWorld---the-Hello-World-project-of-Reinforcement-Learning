# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        allStates = self.mdp.getStates()
        #print(allStates)
        prevIterValues = self.values
        for i in range(iterations):
          for state in allStates:
            if self.mdp.isTerminal(state):
              continue
            possibleActions = self.mdp.getPossibleActions(state)
            actionReward = []
            for action in possibleActions:
              if action == 'exit':
                actionReward.append(self.mdp.getReward(state,action,'TERMINAL_STATE'))
                continue
              tranStateProb = self.mdp.getTransitionStatesAndProbs(state, action)
              nextStateReward = 0
              for eachNextState in tranStateProb:
                nextState, nextStateProb = eachNextState
                nextStateReward += (nextStateProb * (self.mdp.getReward(state,action,nextState) + self.discount * prevIterValues[nextState]))
              actionReward.append(nextStateReward)
            max_reward = max(actionReward)
            self.values[state] = max_reward
            """if i == 0:
              print("*** Test case ****")
              goalAction = self.computeQValueFromValues((1,0),self.computeActionFromValues((1,0)))
              goal_Action = self.computeQValueFromValues((0,1),self.computeActionFromValues((0,1)))
              print(goalAction, goal_Action)
            #print("State --> curr Reward", state, "-->", self.values[state]) """
          prevIterValues = self.values
        
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        tranStateProb = self.mdp.getTransitionStatesAndProbs(state, action)
        nextStateVals = 0
        for eachNextState in tranStateProb:
          nextState, nextStateProb = eachNextState
          nextStateVals += (nextStateProb * (self.mdp.getReward(state,action,nextState) + self.discount * self.values[nextState]))
        #util.raiseNotDefined()
        return nextStateVals

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        possibleActions = self.mdp.getPossibleActions(state)
        if self.mdp.isTerminal(state):
          return possibleActions
        actionReward = []
        for action in possibleActions:
          actionReward.append(self.computeQValueFromValues(state,action))
        max_reward = max(actionReward)
        #print("max-reward-action",possibleActions[actionReward.index(max_reward)])
        return possibleActions[actionReward.index(max_reward)]
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
