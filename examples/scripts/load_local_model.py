
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel


loacl_model = AutoModelForCausalLM.from_pretrained("/home/wxt/huggingface/hub/llama2_sft_mirror/",trust_remote_code=True)
print(local_model)
