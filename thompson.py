import numpy as np

def initialize(arm_probs: list):
    """
    Initialization of the environment for sampling

    Args:
        arm_probs (list): Probabilities of success for each machine

    Returns:
        a (list): Success counts for each machine
        b (list): Failure counts for each machine
        regret (int): Cumulative regret over past player's actions
    """    
    n_arms = len(arm_probs)
    
    a = np.ones(n_arms) #Beta(1, 1) == Uniform distribution
    b = np.ones(n_arms) 
    
    regret = 0

    return a, b, regret

def evaluate(true_prob: float):
    """
    Evaluate if the chosen action was a success or failure

    Args:
        true_prob (float): probability of succes of a chosen machine

    Returns:
        reward (int): 1 if successful action, 0 if failure
    """    
    current_result = np.random.uniform(0, 1, 1)
    reward = 1 if current_result <= true_prob else 0
    
    return reward

def update_prior(a: list, b: list, reward: int, chosen_arm: int):
    """
    Update success and failure counts 

    Args:
        a (list): Success counts for each machine
        b (list): Failure counts for each machine
        reward (int): 1 if successful action, 0 if failure
        chosen_arm (int): index of the chosen arm at the current action

    Returns:
        a (list): Success counts for each machine
        b (list): Failure counts for each machine
    """    
    a[chosen_arm] += reward #number of successes - 1
    b[chosen_arm] += 1 - reward #number of fails - 1
    
    return a, b

def update_regret(regret: int, a: list, b: list, reward: int, chosen_arm: int):
    """
    Update regret upon an action taken

    Args:
        regret (int): Cumulative regret over past player's actions
        a (list): Success counts for each machine
        b (list): Failure counts for each machine
        reward (int): 1 if successful action, 0 if failure
        chosen_arm (int): index of the chosen arm at the current action

    Returns:
        regret (int): Cumulative regret over past player's actions
    """    
    arm_theta = [a / (a + b) for a, b in zip(a, b)]
    if reward == 0:
        if arm_theta[chosen_arm] < max(arm_theta): #regret if some other arm performed better so far
            regret += 1 

    #no regrets if winning
    
    return regret

def thompson_sampl(arm_probs: list, a: list, b: list, regret: int):
    """
    Thompson sampling algorithm
    Oprimizes the exploration vs exploitation trade-off when solving
    the multi armed bandit problem.
    Chooses the next arm to pull.

    Args:
        arm_probs (list): Probabilities of success for each machine
        a (list): Success counts for each machine
        b (list): Failure counts for each machine
        regret (int): Cumulative regret over past player's actions

    Returns:
        chosen_arm (int): index of the chosen arm at the current action
        reward (int): 1 if successful action, 0 if failure
        regret (int): Cumulative regret over past player's actions
    """    
    
    samples = [np.random.beta(a[arm_num], b[arm_num]) for arm_num in range(len(arm_probs))]
    chosen_arm = np.where(samples == np.max(samples))[0][0]
    
    reward = evaluate(arm_probs[chosen_arm])
    
    update_prior(a, b, reward, chosen_arm)
    
    regret = update_regret(regret, a, b, reward, chosen_arm)
    
    return chosen_arm, reward, regret
 