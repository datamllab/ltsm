from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
print(model.hf_device_map)

model.eval()

tokenizer.pad_token = tokenizer.eos_token

class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    max_tokens: int = 1024

def format_prompt_llama3(prompt: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n"
        f"{prompt}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
    )


@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    prompt = request.messages[-1]["content"] # for convenience, temporarily just use the last message.
    prompt = format_prompt_llama3(prompt)

    input_data = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = input_data["input_ids"].to(model.device)
    attention_mask = input_data["attention_mask"].to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id, 
        )

    generated = output[0][input_ids.shape[1]:]
    response_text = tokenizer.decode(generated, skip_special_tokens=True)

    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": request.model,
        "choices": [
            {
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "usage": {
            "prompt_tokens": len(input_ids[0]),
            "completion_tokens": len(output[0]),
            "total_tokens": len(input_ids[0]) + len(output[0]),
        }
    }