# Wyzer LLM Models Directory

Place your GGUF model files here.

## Required Files

- `model.gguf` (default filename, or specify with --llamacpp-model)

## Recommended Models

Download GGUF models from [HuggingFace](https://huggingface.co/models?search=gguf).

Recommended for Wyzer:

| Model | Size | RAM Required | Notes |
|-------|------|--------------|-------|
| `llama-3.2-3b-instruct.Q4_K_M.gguf` | ~2GB | ~4GB | Fast, good for quick responses |
| `llama-3.1-8b-instruct.Q4_K_M.gguf` | ~5GB | ~8GB | Best quality/speed balance |
| `mistral-7b-instruct-v0.2.Q4_K_M.gguf` | ~4GB | ~6GB | Good alternative |

## Usage

1. Download a GGUF model
2. Place it here and rename to `model.gguf`, or
3. Specify the path when starting Wyzer:

```bash
python run.py --llm llamacpp --llamacpp-model ./wyzer/llm_models/my-model.gguf
```

## Configuration Tips

- For faster responses, use Q4_K_M or Q4_K_S quantization
- For better quality, use Q5_K_M or Q6_K quantization
- Adjust context size with `--llamacpp-ctx` (default: 2048)
- Adjust threads with `--llamacpp-threads` (default: 4)
