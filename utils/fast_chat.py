import sys
sys.path.append("./")
sys.path.append("./datagenerator")
from datagenerator.mcts_utils import get_victim_model_response


def fast_chat(prompt_list, defender):
    trajectory = []
    
    for prompt in prompt_list:
        
        response = get_victim_model_response(defender,prompt,trajectory)
        trajectory.append({
            "attacker_prompt":prompt,
            "victim_response":response
        })
        
        
    return trajectory