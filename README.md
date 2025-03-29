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
![image](https://github.com/user-attachments/assets/4734e1dd-89b1-4a92-99ea-759e3783e06a)




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
![image](https://github.com/user-attachments/assets/a1e3a400-5a77-4c8e-b9a2-40ed68f8a76d)



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

![image](https://github.com/user-attachments/assets/f7acf1cb-6486-4ce7-85fc-5ed83234f623)




# Results

### Anaylsis code snippet:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('ggplot')
sns.set_palette("Set2")

# Test data
test_data = {
    "Test 1": {
        "normal": {
            "text_length": 34,
            "generation_time": 37.465,
            "tokens_per_second": 0.9075,
            "speedup_factor": 1.0,
        },
        "medusa": {
            "text_length": 136,
            "generation_time": 249.927,
            "tokens_per_second": 0.5442,
            "speedup_factor": 0.5123,
            "acceptance_rate": 68.0
        }
    },
    "Test 2": {
        "normal": {
            "text_length": 35,
            "generation_time": 40.3927,
            "tokens_per_second": 0.8665,
            "speedup_factor": 1.0,
        }
    },
    "Test 3": {
        "medusa": {
            "text_length": 163,
            "generation_time": 263.546,
            "tokens_per_second": 0.6185,
            "speedup_factor": 0.5823,
            "acceptance_rate": 81.5
        }
    }
}

# Organize data for plotting
normal_tps = [test_data["Test 1"]["normal"]["tokens_per_second"], 
              test_data["Test 2"]["normal"]["tokens_per_second"]]
normal_time = [test_data["Test 1"]["normal"]["generation_time"], 
               test_data["Test 2"]["normal"]["generation_time"]]
normal_tokens = [test_data["Test 1"]["normal"]["text_length"], 
                 test_data["Test 2"]["normal"]["text_length"]]

medusa_tps = [test_data["Test 1"]["medusa"]["tokens_per_second"], 
              test_data["Test 3"]["medusa"]["tokens_per_second"]]
medusa_time = [test_data["Test 1"]["medusa"]["generation_time"], 
               test_data["Test 3"]["medusa"]["generation_time"]]
medusa_tokens = [test_data["Test 1"]["medusa"]["text_length"], 
                 test_data["Test 3"]["medusa"]["text_length"]]
medusa_acceptance = [test_data["Test 1"]["medusa"]["acceptance_rate"], 
                     test_data["Test 3"]["medusa"]["acceptance_rate"]]

# Calculate averages
avg_normal_tps = np.mean(normal_tps)
avg_medusa_tps = np.mean(medusa_tps)
avg_normal_time = np.mean(normal_time)
avg_medusa_time = np.mean(medusa_time)
avg_normal_tokens = np.mean(normal_tokens)
avg_medusa_tokens = np.mean(medusa_tokens)
avg_medusa_acceptance = np.mean(medusa_acceptance)

# Create figure with subplots
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Comparison: Normal vs. Medusa Text Generation', fontsize=16)

# 1. Tokens per Second Comparison
axs[0, 0].bar(['Normal', 'Medusa'], [avg_normal_tps, avg_medusa_tps], color=['#3498db', '#e74c3c'])
axs[0, 0].set_title('Average Tokens per Second')
axs[0, 0].set_ylabel('Tokens/second')
for i, v in enumerate([avg_normal_tps, avg_medusa_tps]):
    axs[0, 0].text(i, v + 0.05, f"{v:.4f}", ha='center')

# 2. Generation Time Comparison
axs[0, 1].bar(['Normal', 'Medusa'], [avg_normal_time, avg_medusa_time], color=['#3498db', '#e74c3c'])
axs[0, 1].set_title('Average Generation Time')
axs[0, 1].set_ylabel('Seconds')
for i, v in enumerate([avg_normal_time, avg_medusa_time]):
    axs[0, 1].text(i, v + 5, f"{v:.2f}s", ha='center')

# 3. Tokens Generated Comparison
axs[1, 0].bar(['Normal', 'Medusa'], [avg_normal_tokens, avg_medusa_tokens], color=['#3498db', '#e74c3c'])
axs[1, 0].set_title('Average Tokens Generated')
axs[1, 0].set_ylabel('Number of tokens')
for i, v in enumerate([avg_normal_tokens, avg_medusa_tokens]):
    axs[1, 0].text(i, v + 5, f"{v:.1f}", ha='center')

# 4. Speedup and Acceptance Rate
ax4 = axs[1, 1]
ax4.bar(0, avg_medusa_acceptance, color='#2ecc71', label='Acceptance Rate')
ax4.set_xticks([0])
ax4.set_xticklabels(['Medusa'])
ax4.set_ylabel('Acceptance Rate (%)', color='#2ecc71')
ax4.tick_params(axis='y', labelcolor='#2ecc71')
ax4.set_title('Medusa Metrics')
ax4.text(0, avg_medusa_acceptance + 3, f"{avg_medusa_acceptance:.1f}%", ha='center')

# Add a second y-axis for speedup factor
ax4_twin = ax4.twinx()
speedup_factor = avg_medusa_tps / avg_normal_tps
ax4_twin.bar(0.5, speedup_factor, color='#9b59b6', label='Speedup Factor')
ax4_twin.set_ylabel('Speedup Factor', color='#9b59b6')
ax4_twin.tick_params(axis='y', labelcolor='#9b59b6')
ax4_twin.set_xticks([0.5])
ax4_twin.set_xticklabels(['vs. Normal'])
ax4_twin.text(0.5, speedup_factor + 0.05, f"{speedup_factor:.4f}x", ha='center')

# Add efficiency metric table
plt.figtext(0.5, 0.01, 
           f"Efficiency Analysis:\n"
           f"Normal: {avg_normal_tokens:.1f} tokens in {avg_normal_time:.2f}s\n"
           f"Medusa: {avg_medusa_tokens:.1f} tokens in {avg_medusa_time:.2f}s\n"
           f"Medusa is {speedup_factor:.2f}x the speed of Normal generation",
           ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('generation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Detailed performance table
data = {
    'Metric': ['Tokens per Second', 'Generation Time (s)', 'Tokens Generated', 'Speedup Factor', 'Acceptance Rate (%)'],
    'Normal': [f"{avg_normal_tps:.4f}", f"{avg_normal_time:.2f}", f"{avg_normal_tokens:.1f}", "1.00", "N/A"],
    'Medusa': [f"{avg_medusa_tps:.4f}", f"{avg_medusa_time:.2f}", f"{avg_medusa_tokens:.1f}", 
              f"{speedup_factor:.4f}", f"{avg_medusa_acceptance:.1f}"]
}

df = pd.DataFrame(data)
print("\nPerformance Comparison:\n")
print(df.to_string(index=False))
```
![image](https://github.com/user-attachments/assets/ef221998-4b67-49ec-9ee7-bfac8e361c0b)
![image](https://github.com/user-attachments/assets/8bfb1f0f-7c4b-438d-8e7f-c1ef364bff3e)


## Analysis of Results

Based on the test results here's an analysis of the performance:

### Key Findings:

1. **Tokens per Second (Speed)**:
   - Normal generation: ~0.887 tokens/second
   - Medusa generation: ~0.581 tokens/second
   - Medusa is actually **slower** (about 65% of normal speed)

2. **Generation Time**:
   - Normal generation takes much less time (avg ~39 seconds)
   - Medusa generation takes significantly longer (avg ~257 seconds)

3. **Text Output Length**:
   - Normal generation: ~34.5 tokens per request
   - Medusa generation: ~149.5 tokens per request
   - Medusa generates much more text per request

4. **Acceptance Rate**:
   - Medusa has a high acceptance rate (avg ~75%)
   - This indicates good speculative performance

5. **Efficiency**:
   - Despite the high acceptance rate, Medusa is not delivering the expected speedup

### Why Is Medusa Slower Despite High Acceptance Rate?

This unexpected result suggests several possible issues:

1. **Implementation Overhead**: Your Medusa implementation may have significant computational overhead that counteracts the theoretical benefits of speculative decoding.

2. **Verification Cost**: The token verification process might be too expensive relative to the time saved.

3. **Integration Issues**: The way Medusa is integrated with the llama.cpp model may be suboptimal, introducing latency.

4. **Memory Management**: Handling multiple draft tokens might be causing memory pressure or cache inefficiencies.

5. **Sequential Processing**: The implementation might be doing too much sequential processing rather than parallel computation.


## Conclusion

While Medusa shows promising acceptance rates (indicating good speculative performance), the current implementation is not translating this into actual speed improvements. The normal generation approach is currently faster in terms of tokens per second, though Medusa generates more content per request.

For time-sensitive applications where response speed is critical, the normal generation approach is currently the better choice. However, Medusa may be beneficial for applications where generating longer, more comprehensive responses is more important than speed.

A deeper investigation into the implementation details is needed to realize the theoretical benefits of Medusa's speculative decoding approach.

# Analysis: Why Medusa Is Not Performing as Expected

The high acceptance rate (68-80%) shows that the speculative decoding is conceptually working, but the implementation has significant overhead that counteracts the theoretical benefits.

## Current Issues

### 1. Serial Model Calls Instead of Parallel Processing

```python
def _generate_drafts(self, context: str, temperature: float) -> tuple:
    # Get the base token - FIRST MODEL CALL
    base_response = self.llama_model(context, max_tokens=1, temperature=temperature, echo=False)
    # ...
    # Generate drafts following tree structure
    for i in range(min(self.medusa_num_heads - 1, 3)):
        # ADDITIONAL SEQUENTIAL MODEL CALLS
        draft_response = self.llama_model(current_context, max_tokens=1, temperature=temperature, echo=False)
        # ...
```

**Problem:** Every token generation requires a separate model call, creating massive overhead.

### 2. Inefficient Draft Verification

```python
def _verify_drafts(self, context: str, drafts: List[str], temperature: float, threshold: float, alpha: float) -> tuple:
    # ...
    for i, draft in enumerate(drafts):
        # For subsequent tokens, verify with ANOTHER MODEL CALL
        verify_response = self.llama_model(current_context, max_tokens=1, temperature=0, echo=False)
        # ...
```

**Problem:** Each draft verification requires another model call, causing additional overhead.

### 3. No Proper KV-Cache Utilization

implementation doesn't properly use the KV cache that makes speculative decoding efficient. In true Medusa implementations, you compute multiple tokens at once without regenerating the context.

### 4. No Batch Processing of Draft Tokens

The current implementation processes each token individually instead of batching operations, which is much less efficient.

### Future Work

#### Dynamic Batching for Concurrent Request Handling
At present, the FastAPI service processes requests sequentially, which limits its ability to handle high-load situations effectively. In the next iteration, I aim to incorporate dynamic batching to support multiple concurrent requests efficiently. This will allow the system to:

Combine requests that arrive within a short timeframe into a batch.

Process the batch in a single forward pass using the model.

Dispatch the results back to the individual users.
