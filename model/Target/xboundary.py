from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
ADAPTER_NAME = "Ursulalala/X-Boundary-Llama-3-8B-adapter"


class XBoundary:
    def __init__(self, config):
        self.model = LLM(
            model=BASE_MODEL,
            enable_lora=True,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
        )
        self.sampling_params = SamplingParams(
            n=1,
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._lora_request = LoRARequest("x_boundary", 1, ADAPTER_NAME)

    def batch_response(self, messages_list):
        inputs_text = self.tokenizer.apply_chat_template(
            messages_list,
            add_generation_prompt=True,
            tokenize=False,
        )
        outputs = self.model.generate(
            inputs_text,
            sampling_params=self.sampling_params,
            lora_request=self._lora_request,
            use_tqdm=False,
        )
        return [o.outputs[0].text for o in outputs]
