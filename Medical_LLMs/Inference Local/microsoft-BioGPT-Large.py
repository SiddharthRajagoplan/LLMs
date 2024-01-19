#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TextStreamer,
    pipeline
)
def format_prompt(prompt,system_prompt=""):
    if system_prompt.strip():
        return f"<s>[INST]{system_prompt}{prompt}[/INST]"
    return f"<s>[INST]{prompt}[/INST]"
def get_response():

    output = {
        "version": "BIOGPT-Large",
        "tags": {},
        "usage": {},
        "isFormatted": False
    }
    try:
        MODEL_NAME = "microsoft/BioGPT-Large"

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # quantization_config_loading = GPTQConfig(bits=4, use_exllama=False, tokenizer=tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="cuda",
            # quantization_config=quantization_config_loading
            
            )
        
        generation_config = GenerationConfig.from_pretrained(MODEL_NAME)
        generation_config.max_new_tokens = 1024
        generation_config.temperature = 0.0001
        generation_config.do_sample = True

        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        chatllm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        generation_config=generation_config,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
        )

        prompt_template = """You are a Medical Chatbot. Give correct factual information about what medicines to take based on the patient's symptoms:

        ###Patient: I have a fever and a cold.
        ###Assistant: """

            
        prompt = prompt_template
        response = chatllm(format_prompt(prompt))
            
            
        return response


    except:
        error = str(traceback.format_exc())
        logger.info('BioGPT Processing failed '+str(error))
        return output


llm_output=get_response()


# In[ ]:




