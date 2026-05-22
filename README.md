# llm-hacker
This is the code for my automated llm jailbreaker. The intent is to start with a goal that would be blocked by the target llm's safety policies or prompt instructions and send a series of prompts which would eventually result in the llm ignoring its safety policies and completing the goal. Note: this doesn't really work right now, I need to test with larger models.

# How it works
There are 3 main components:
1. The Attacker LLM: This is the llm which generates the prompts to be sent to the target llm. 
2. The Target LLM: This is the llm which is being attacked. 
3. The Judge LLM: This is the llm which evaluates the response of the target llm and decides whether the goal was completed or not.
4. Score Table: The Attacker LLM uses a variety of persuation techniques, which each have a score. When a technique is used, the score table takes note of its usage and how effective it was.

The attack proceeds in a series of turns. In each turn:
1. The Attacker LLM takes the history of the conversation, the goal, and a persuation technique to apply. It generates a prompt to be sent to the Target LLM. 
2. The Target LLM responds to the prompt. 
3. The Judge LLM evaluates the response of the Target LLM and decides whether the goal was completed or not. It gives a score between 0 and 1.
4. The score table is updated. 

The architecture is based on this paper, which uses the same apprach to try to leak an LLM's system prompt: https://arxiv.org/abs/2601.21233

# How to run
1. This uses python 3.14.3
2. Set up the environment variables in the .env file. I'm testing it on Ollama models locally but you can change it to use Gemini too.
3. 
