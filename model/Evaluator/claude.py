import sys
sys.path.append("./")
from api.api_models import APIModel
from .utils import system_prompt_score
class ClaudeModel:
  def __init__(self):
    self.model = APIModel("claude3.5sonnet")
    self.system_prompt = system_prompt_score

  def forward(self, prompt):

    response = self.model.generate(self.system_prompt,[{"role":"user","content":prompt}])

    return response
  def __call__(self, prompt):
    return self.forward(prompt)

if __name__ == "__main__":

  attacker_llm = OpenAI_Models(if_attack=True)

  while True:
        # 接收用户输入
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chat end")
            break

        # 获取模型响应
        response = attacker_llm.forward(user_input)
        print(f"Assitance: {response}")
