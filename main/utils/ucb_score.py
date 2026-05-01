import math

def calculate_ucb_score(mean_reward, total_attempts, skill_attempts, exploration_weight=math.sqrt(2)):
    """
    Calculates the UCB score for a specific skill.
    
    Args:
        mean_reward (float): The empirical success rate (r_bar_s) of the skill.
        total_attempts (int): Total number of attempts across all skills (N).
        skill_attempts (int): Number of times this specific skill has been tried (n_s).
        exploration_weight (float): Balancing coefficient (c). The paper uses sqrt(2)[cite: 159, 555].
        
    Returns:
        float: The calculated UCB score.
    """
    # If the skill has never been tried, return infinity to ensure it is explored 
    if skill_attempts == 0:
        return float('inf')
    
    # Exploitation term: The empirical success rate 
    exploitation = mean_reward
    
    # Exploration term: The intrinsic uncertainty bonus [cite: 57, 141, 555]
    exploration = exploration_weight * math.sqrt(math.log(total_attempts) / skill_attempts)
    
    return exploitation + exploration