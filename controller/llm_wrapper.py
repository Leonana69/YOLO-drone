import os
import openai

GPT3 = "gpt-3.5-turbo-16k"
GPT4 = "gpt-4"
LLAMA3 = "meta-llama/Meta-Llama-3-8B-Instruct"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
chat_log_path = os.path.join(CURRENT_DIR, "assets/chat_log.txt")

class LLMWrapper:
    def __init__(self, temperature=0.0):
        self.temperature = temperature
        # clean chat_log
        open(chat_log_path, "w").close()
        self.llama_client = openai.OpenAI(
            base_url="http://10.66.41.78:8000/v1",
            api_key="token-abc123",
        )
        self.gpt_client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

    def request(self, prompt, model_name=GPT4):
        if model_name == LLAMA3:
            client = self.llama_client
        else:
            client = self.gpt_client
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        # save the message in a txt
        with open(chat_log_path, "a") as f:
            f.write(prompt + "\n---\n")
            f.write(response.model_dump_json(indent=2) + "\n---\n")

        # print(f"LLM response: {response}")
        return response.choices[0].message.content