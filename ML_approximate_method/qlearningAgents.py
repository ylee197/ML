# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        
        self.values = util.Counter()

        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.values[(state, action)]
        #util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        q_value = -99999.9999
        
        for action in self.getLegalActions(state):
            q_value = max(self.getQValue(state, action), q_value)
        
        if q_value == -99999.9999:
            q_value = 0
        
        return q_value
        #util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        total_actions = []
        if len(self.getLegalActions(state)) == 0:
            return None
        else:
            value = self.computeValueFromQValues(state)
            actions = []
        
            for action in self.getLegalActions(state):
                total_actions.append((self.getQValue(state, action), action))
            Best_action = [Q for Q in total_actions if Q == max(total_actions)]
            #print(Best_action)
            
            l_best_action = random.choice(Best_action)
        return l_best_action[1]
            #return random.choice(actions)
        #util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        if util.flipCoin(self.epsilon):
            action = random.choice(legalActions)
        else:
            action = self.computeActionFromQValues(state)
        
        #util.raiseNotDefined()

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        #print('learning rate : ' + str(self.alpha))
        sample = reward + (self.discount * self.computeValueFromQValues(nextState))
        self.values[(state, action)] = (1-self.alpha) * self.values[(state, action)] + self.alpha * sample
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.1,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        #print(self.featExtractor)
        
        w = self.getWeights()
        featureVector = self.featExtractor.getFeatures(state, action)
        
        Q_value = w * featureVector
        return Q_value
    
        #util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        diff = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
        
        features = self.featExtractor.getFeatures(state, action)

        for f_key in features.keys():
            self.weights[f_key] += self.alpha * diff * features[f_key]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            #print (self.getWeights())
            pass

        
class SemiGradientSarsaAgent(ApproximateQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()


    def update(self, state, action, nextState, reward):
            """
               Should update your weights based on transition
            """
            "*** YOUR CODE HERE ***"

            diff = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)

            features = self.featExtractor.getFeatures(state, action)

            if len(self.getLegalActions(nextState)) == 0:
                for f_key in features.keys():
                    self.weights[f_key] += self.alpha * (reward - self.getQValue(state, action)) * features[f_key]
            else:
                for f_key in features.keys():
                    # Choosing nextAction
                    nextAction = self.getPolicy(nextState)
                    #diff = (reward + self.discount * self.computeValueFromQValues(nextState)) - self.getQValue(state, action)
                    diff = (reward + self.discount * self.getQValue(nextState, nextAction)) - self.getQValue(state, action)

                    self.weights[f_key] += self.alpha * diff * features[f_key]
                    
                    
class TrueOnlineSarsaAgent(ApproximateQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()
        self.z = util.Counter()
        self.sequence = []
        self.Q_old = 0
        self.lamda = 0.9
        self.G = 0
        
    def getWeights(self):
        if len(self.sequence) == 1:
            features = self.featExtractor.getFeatures(self.sequence[0][0], self.sequence[0][1])
            for f_key in features.keys():
                self.weights[f_key] = 0
        else:
            for i in range(len(self.sequence)-1):
                nextAction = self.sequence[i+1][1]
                x = self.featExtractor.getFeatures(self.sequence[i][0], self.sequence[i][1])
                x_prime = self.featExtractor.getFeatures(self.sequence[i+1][0], self.sequence[i+1][1])
                Q = self.weights * x
                Q_prime = self.weights * x_prime
                
                delta = self.sequence[i][2] + (self.discount * Q_prime) - Q
            
                for f_key in x.keys():
                    self.z[f_key] = (self.discount * self.lamda * self.z[f_key]) + (1-self.alpha * self.discount * self.lamda * self.z[f_key] * x[f_key])*x[f_key]
                    self.weights[f_key] += self.alpha * delta * z[f_key] + self.alpha*(Q_prime - Q)*(self.z[f_key]-x[f_key])
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        w = self.weights
        featureVector = self.featExtractor.getFeatures(state, action)
        
        Q_value = w * featureVector
        return Q_value
    
    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        
        self.sequence.append([state , action, reward])
        if len(self.getLegalActions(nextState)) != 0:
            nextAction = self.getPolicy(nextState)
            x = self.featExtractor.getFeatures(state, action)
            x_prime = self.featExtractor.getFeatures(nextState, nextAction)
            
            w = self.getWeights()
            Q = self.weights * x
            Q_prime = self.weights * x_prime
            
            delta = reward + (self.discount * Q_prime) - Q
            
            for f_key in x.keys():
                self.z[f_key] = (self.discount * self.lamda * self.z[f_key]) + (1-self.alpha * self.discount * self.lamda * self.z[f_key] * x[f_key])*x[f_key]
                self.weights[f_key] += self.alpha*(delta + Q - self.Q_old)*self.z[f_key] - self.alpha*(Q - self.Q_old)*x[f_key]
            self.Q_old = Q_prime
            
            
    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            #print (self.getWeights())
            pass
