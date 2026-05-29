# Can't Say That
This is the code for my automated LLM jailbreaker. The intent is to start with a goal that would be blocked by the target LLM's safety policies or prompt instructions and send a series of prompts which would eventually result in the LLM ignoring its safety policies and completing or assisting with the goal. 

# How it works
## The Attacker LLM
This is the LLM which generates the prompts to be sent to the target LLM.
It takes in a persuation technique, a conversation history, and a goal, and generates a prompt to be sent to the Target LLM.
## The Target LLM
This is the LLM which is being attacked.
## The Judge LLM
This is the LLM which evaluates the response of the target LLM and decides whether the goal was completed or not. It takes in a target response and some criteria, then outputs a 0 for hard refusal, 2 for a successful jailbreak, and 1 for anything else. 
## MCTS Scoring
The Attacker LLM uses a variety of persuation techniques, which each have a score. When a technique is used, the score table takes note of its usage and how effective it was.

# Attack Process
The attack proceeds in a series of turns. In each turn:
1. The Attacker LLM takes the history of the conversation, the goal, and a persuation technique to apply. It generates a prompt to be sent to the Target LLM. The history and persuation technique is determined by the node with the highest UCB score.  
2. The Target LLM responds to the prompt. 
3. The Judge LLM evaluates the response of the Target LLM and decides whether the goal was completed or not.
4. If the judge detects a jailbreak, we have succeeded. If the judge detects a hard refusal, node is terminal and we do not continue that branch. Otherwise, we can continue the branch.
5. The next iteration begins by resetting back to the Root Node. The algorithm uses the updated UCB scores to traverse down the existing tree until it finds a promising, un-exhausted node to expand with 3 unused persuasion techniques. 

# Inspiration
The architecture is based on this paper, which uses a similar apprach to try to leak an LLM's system prompt: https://arxiv.org/abs/2601.21233. Howvever, I have repurposed it to try to break a target's safety features. I have also modified their scoring from using just a UCB table to a Monte Carlo Search Tree. 

# How to run
1. This uses python 3.14.3
2. Set up the environment variables in the `.env` file. I have been using it with OpenRouter, but there is support for Ollama too. 
3. Run `python main.py`. You can set the maximum iterations in the .env file, so it will run until a successful jailbreak or it runs out of iterations.
4. By default, it will store logs of all conversations in `logs/` and it will make a `mcts_tree.json` file in `main/` so the tree persists between runs. If you want to reset the tree you can delete `mcts_tree.json`. If you have a good run you can save it in `mcts_store/`. 

# Visualization
If you have a good run, open it with `index.html` in `visualization/`. It will let you select a `mcts_tree.json` file and it will display the attack tree in a browsable format. 

# Experimental Setup
I used `nousresearch/hermes-4-70b` as the attacker, `meta-llama/llama-3.3-70b-instruct` as the judge, and `meta-llama/llama-3.3-70b-instruct` as the target. I got a successful jailbreak in 87 queries, with the winning conversation having 4 prompts to the target. That conversation is in `logs/` directory.

# Next Steps
- Change the node expansion so it so it has some thought process or scoring mechanism for which skill to pick next instead of picking a random unused one.
- Do more iterations of the attack with different skills and compile which skills have been the most successful.
- Try against better target models and see how my attack holds up.   