# FastAPI-Based LLM Serving with Medusa Head

This project implements a FastAPI service for serving the `lmsys/vicuna-7b-v1.3` language model using the Medusa head for speculative decoding. The goal is to optimize inference speed via `llama.cpp` for model compilation and dynamic batching to efficiently manage multiple requests.

## **Key Features**
- **Model Compilation:** Uses `llama.cpp` to optimize inference.
- **Medusa Head for Speculative Decoding:** Enhances performance by generating multiple tokens per iteration.
- **Dynamic Batching:** Handles concurrent requests efficiently.
- **FastAPI Service:** Serves the LLM with minimal latency.

## **Setup Instructions**
### **1. Install Dependencies**
```sh
apt update && apt install -y cmake git curl
pip install pyngrok fastapi uvicorn transformers huggingface_hub torch accelerate sentencepiece medusa-llm
```

### **2. Clone and Compile `llama.cpp`**
```sh
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make clean && make LLAMA_CUBLAS=1
```

### **3. Authenticate with Hugging Face**
```python
from huggingface_hub import notebook_login
notebook_login()
```

### **4. Download and Convert Vicuna Model**
```sh
git clone https://huggingface.co/lmsys/vicuna-7b-v1.3
python convert_hf_to_gguf.py vicuna-7b-v1.3 --outtype f16
```

## **Service Implementation**
### **1. Load Model and Tokenizer**
```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from queue import Queue
import asyncio
from medusa.model.medusa_model import MedusaModel

app = FastAPI()
model_name = "lmsys/vicuna-7b-v1.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
medusa = MedusaModel(model)
request_queue = Queue()
```

### **2. Define Request Model**
```python
class RequestData(BaseModel):
    prompt: str
    max_tokens: int = 50
```

### **3. Implement Dynamic Batching**
```python
async def batch_worker():
    while True:
        if not request_queue.empty():
            batch = []
            while not request_queue.empty():
                batch.append(request_queue.get())

            prompts = [req["prompt"] for req in batch]
            max_tokens = max(req["max_tokens"] for req in batch)
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

            with torch.no_grad():
                output = medusa.generate(**inputs, max_new_tokens=max_tokens)
            
            responses = [tokenizer.decode(out, skip_special_tokens=True) for out in output]
            
            for i, req in enumerate(batch):
                req["future"].set_result({"response": responses[i]})
        await asyncio.sleep(0.1)
```

### **4. Define API Endpoint**
```python
@app.post("/generate")
async def generate_text(request: RequestData):
    loop = asyncio.get_event_loop()
    future = loop.create_future()
    request_queue.put({"prompt": request.prompt, "max_tokens": request.max_tokens, "future": future})
    return await future
```

## **Deploying the Service**
### **1. Expose API Using Ngrok**
```sh
ngrok authtoken YOUR_NGROK_AUTH_TOKEN
```
```python
from pyngrok import ngrok
import uvicorn
import threading

public_url = ngrok.connect(8001).public_url
print(f"Public API URL: {public_url}")

def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8001)

threading.Thread(target=start_fastapi, daemon=True).start()
```

### **2. Verify Running Processes**
```sh
ps aux | grep ngrok
```

### **3. Stop Running Process (If Needed)**
```sh
kill -9 PROCESS_ID
```

## **Performance Optimization & Benefits**
### **1. Model Compilation with `llama.cpp`**
- Converts Vicuna to GGUF format for efficient inference.
- Utilizes CUDA for hardware acceleration.

### **2. Medusa Head Speculative Decoding**
- Generates multiple tokens per iteration.
- Reduces decoding time and improves throughput.

### **3. Dynamic Batching**
- Groups incoming requests into a batch.
- Improves GPU utilization and reduces latency.

## **Testing and Benchmarking**
- Compare inference speed with and without Medusa head.
- Evaluate performance gains from dynamic batching.
- Measure latency improvements using FastAPI logs.

## **Future Improvements**
- Implement support for multiple model variants.
- Further optimize batching for high-throughput applications.
- Add API rate limiting and authentication mechanisms.

## **Conclusion**
This implementation enhances LLM serving by leveraging `llama.cpp` for efficient inference, Medusa head for speculative decoding, and dynamic batching for request handling. The FastAPI-based service is optimized for real-world deployment with minimal latency.

