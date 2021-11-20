# Reinforcement Learning - approximate method in Pacman 

## Introduction
In this project experimented with various Reinforcement Learning Approximate method techniques namely value Approximate Q-learning, Episode semi-gradient SARSA and True online SARSA. This is part of Pacman projects developed at <a href = 'http://ai.berkeley.edu/project_overview.html'>UC Berkeley</a>. 

## Directory Structure
-- ML_approximate_method 
    - qlearningAgents.py : It contains Q-learning, approximate Q-learning, Epsode semi-gradient SARSA and True online SARSA classes
    - learningAgents.py : This file contains training and test control. 

-- Analysis
    - Analysis.ipynb : This file analizes different algorithm with Pacman scores and execution time. 
    
## Executing
- Executing each algorithm :
  ### python pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
  : Above command line execute ApproximateQAgent, 50 times training and 10 times testing. Using environment is mediumGrid. 
  : Approximate Q-leanring algorithm => ApproximateQAgent
  : Episode semi-gradient SARSA algorithm => SemiGradientSarsaAgent
  : True online SARSA algorithm => TrueOnlineSarsaAgent
  
  : Environment 
    - smallGrid
    - mediumGrid
    - mediumClassic

- Running Analysis.ipynb
  : Put the Q-learning, Episode semi-gradient SARSA and True online SARSA algorithm result files in the same directory with the program. 
