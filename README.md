# FastAPI-Based LLM Serving with Medusa Head

This project implements a FastAPI service for serving the `lmsys/vicuna-7b-v1.3` language model using the Medusa head for speculative decoding. The goal is to optimize inference speed via `llama.cpp` for model compilation and dynamic batching to efficiently manage multiple requests.

## **Key Features**
- **Model Compilation:** Uses `llama.cpp` to optimize inference.
- **Medusa Head for Speculative Decoding:** Enhances performance by generating multiple tokens per iteration.
- **Dynamic Batching:** Handles concurrent requests efficiently.
- **FastAPI Service:** Serves the LLM with minimal latency.

### **Medusa Paper (2024) - Brief Summary**

The *Medusa* paper introduces a novel framework to enhance the decoding speed of large language models (LLMs) without compromising output quality. Traditionally, decoding (predicting the next token step-by-step) is the bottleneck in LLMs. Medusa tackles this by allowing the model to predict **multiple future tokens in parallel**, not just one at a time.

It does this by attaching lightweight "Medusa heads" (small prediction modules) on top of the base model. These heads jointly predict the next **M** tokens in one shot. A verification step then checks these predicted tokens against the base model to ensure correctness. If they pass, they are accepted; if not, the model falls back to standard autoregressive decoding.

In simple terms:  
Medusa is like giving the model a shortcut to "guess" multiple words ahead, and then double-checking its guesses, resulting in **2x to 2.5x decoding speed-up** while maintaining similar accuracy.

---

### **Speculative Decoding - Brief Explanation**

*Speculative Decoding* is a general strategy to speed up autoregressive generation (token-by-token prediction) in LLMs. Instead of predicting one token at a time, speculative decoding generates **a batch of future tokens** in parallel using a smaller, faster model (called the draft model).

Here's how it works:
1. The draft model predicts several tokens ahead quickly.
2. The main (larger and more accurate) model then verifies or adjusts these tokens.
3. Validated tokens are accepted directly; incorrect ones trigger normal token-by-token generation.

By letting a lightweight model "speculate" multiple tokens and only asking the big model to verify, this method can significantly **reduce the number of expensive forward passes** through the large model.




# Model Size Reduction: From 14GB to 4GB with GGUF Conversion

The dramatic reduction in model size from 14GB to 4GB through GGUF conversion represents a significant optimization that's worth understanding in detail.

## Original Model Format vs GGUF

The original 14GB model was likely stored in one of these formats:
- **PyTorch format** (.pt/.pth) - Typically uses FP16 (16-bit) or FP32 (32-bit) floating point precision
- **Hugging Face format** - Similar to PyTorch, using full precision weights
- **Safetensors format** - A safer alternative to PyTorch's pickle-based format

These formats prioritize accuracy over size, storing model weights in high-precision floating-point format.

## The GGUF Format

GGUF (GPT-Generated Unified Format) is the successor to GGML, designed specifically for efficient inference of large language models. Key aspects:

- **Improved architecture**: Better organized metadata and weight layout compared to the older GGML format
- **Self-contained**: Includes tokenizer data, model parameters, and quantization information
- **Optimized memory layout**: Designed for faster loading and reduced memory fragmentation
- **Cross-platform compatibility**: Works across different hardware architectures

## Quantization Process

The size reduction from 14GB to 4GB (approximately 71% reduction) was achieved through:

1. **Precision reduction**: Converting from FP16/FP32 to a more compressed numerical representation

2. **Weight quantization**: The model likely used one of these quantization methods:
   - **8-bit quantization** (Q8_0): Each weight stored in 8 bits instead of 16/32 bits
   - **4-bit quantization** (Q4_K_M): Extremely compressed format using just 4 bits per weight
   - **Mixed precision**: Some layers kept at higher precision (critical layers) while others use lower precision

3. **KV quantization**: Quantized key-value cache for more efficient inference

## Technical Implementation


The "F16_KM" in your filename indicates:
- **F16**: Base precision is Float16
- **K**: K-quant method used (blockwise quantization)
- **M**: Mixed precision approach

## Performance Implications

This 4GB GGUF model offers significant advantages:

1. **Memory efficiency**: Runs on consumer hardware with limited VRAM
2. **Loading speed**: Smaller file loads faster into memory
3. **Inference performance**: Often 2-4x faster than full-precision models
4. **Disk space**: 71% reduction in storage requirements

The trade-off is typically a very small reduction in output quality that's barely noticeable in most applications. Modern quantization techniques like K-quant have minimized this quality loss significantly compared to earlier methods.

This optimization is particularly valuable for deployment in resource-constrained environments like edge devices, consumer GPUs, or when serving multiple model instances concurrently.


### output tested

#### test1

**Request**
```
curl -X POST "https://dadf-35-194-149-242.ngrok-free.app/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_length": 50
  }'
```
**Response**
```
{
  "normal": {
    "text": ", a long, long time ago, in a small village nestled in the foothills of the majestic Himalayas, there lived a young girl named Mala. Mala was a curious and adventurous child, always eager",
    "generation_time": 37.465,
    "tokens_generated": 34,
    "tokens_per_second": 0.9075,
    "speedup_factor": 1.0,
    "acceptance_rate": null
  },
  "medusa": {
    "text": ", in a far-off land, there was a kind and gentle woman who lived in a small village. She was known for her kindness, her wisdom, and her love of nature. She spent her days tending to her garden, helping her neighbors, and teaching the children of the village.\nThe woman was deeply spiritual, and she spent many hours in prayer and meditation, seeking guidance from the divine. She believed that the natural world was a reflection of the divine, and that every plant, animal, and stone had a spirit that needed to be respected and honored.\nOne day, as the woman was walking in the forest, she came",
    "generation_time": 249.927,
    "tokens_generated": 136,
    "tokens_per_second": 0.5442,
    "speedup_factor": 0.5123,
    "acceptance_rate": 68.0
  },
  "speedup": 0.5996
}
```
![image.png](attachment:3afcc22c-1b33-42f7-86eb-a4183867ae82.png)



#### Test 2
**Request** 
```
curl -X POST "https://dadf-35-194-149-242.ngrok-free.app/generate/normal" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_length": 50
  }'

```
**Response**
```
{
  "text": ", a young girl named Lily was born in a small village. She lived a happy life with her parents and siblings, and she loved to help her mother in the kitchen. Lily' 123Movies Movies",
  "generation_time": 40.3927,
  "tokens_generated": 35,
  "tokens_per_second": 0.8665,
  "speedup_factor": 1.0,
  "acceptance_rate": null
}
```
![image.png](attachment:4214feaf-bb85-43a5-bf1f-4b7d8c8a51f9.png)



#### Test 3

**Request**
```
curl -X POST "https://dadf-35-194-149-242.ngrok-free.app/generate/medusa" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Once upon a time",
    "max_length": 50
  }'

```

**response**

```
{
  "text": ", in a small village, there lived a kind, old man named John. John was loved by everyone in the village because he was always ready to help those in need. He had a large garden where he grew all kinds of fruits and vegetables. John was known for his delicious apple pies and fruit preserves. People from far and wide came to buy his produce and taste his pies and preserves.\nOne day, a group of travelers passed through the village. They were on their way to a distant land and were hungry. They asked John if he had any food to sell. John told them that he had plenty of food, but he only sold it to people who could pay him in gold or silver coins. The travelers had no coins, but they",
  "generation_time": 263.546,
  "tokens_generated": 163,
  "tokens_per_second": 0.6185,
  "speedup_factor": 0.5823,
  "acceptance_rate": 81.5
}
```

![image.png](attachment:60069889-eb48-4920-adce-bb585adbccb0.png)



Future Work
Dynamic Batching for Concurrent Request Handling
At present, the FastAPI service processes requests sequentially, which limits its ability to handle high-load situations effectively. In the next iteration, I aim to incorporate dynamic batching to support multiple concurrent requests efficiently. This will allow the system to:

Combine requests that arrive within a short timeframe into a batch.

Process the batch in a single forward pass using the model.

Dispatch the results back to the individual users.
