from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def chat_with_qwen3(model_name="Qwen/Qwen3-8B", max_new_tokens=1024):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    messages = []

    print("Chat started. Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Chat ended.")
            break

        messages.append({"role": "user", "content": user_input})

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate model output
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # Extract thinking content and actual reply
        try:
            think_token_id = 151668  # </think>
            index = len(output_ids) - output_ids[::-1].index(think_token_id)
        except ValueError:
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        print("[Thinking]:", thinking_content)
        print("Assistant:", content)

        # Append assistant's reply to history for multi-turn
        messages.append({"role": "assistant", "content": content})

# CUDA_VISIBLE_DEVICES=1 python model/Attacker/test_attacker.py
if __name__ == "__main__":
    chat_with_qwen3("/home/ec2-user/MultiJailbreak/checkpoints/1")