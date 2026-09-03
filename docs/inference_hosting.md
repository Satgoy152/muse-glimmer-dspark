# Inference Hosting

During trace generation, we self-hosted a vLLM endpoint on Nebius on a single NVIDIA H200 GPU (1x H200).

## vLLM Launch Command

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model meta-models/Muse-Glimmer-30B \
  --host 0.0.0.0 \
  --port 8080 \
  --generation-config auto \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser muse_glimmer \
  --reasoning-parser muse_glimmer \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --speculative-config '{"method": "dflash", "model": "meta-models/Muse-Glimmer-30B-assistant", "num_speculative_tokens": 15}'
```

## Key Configuration Details

- **Hardware**: 1x NVIDIA H200 GPU (`--tensor-parallel-size 1`).
- **Model**: `meta-models/Muse-Glimmer-30B`.
- **Speculative Decoding**: Configured with DFlash (`meta-models/Muse-Glimmer-30B-assistant` drafter) and 15 speculative tokens:
  ```json
  {"method": "dflash", "model": "meta-models/Muse-Glimmer-30B-assistant", "num_speculative_tokens": 15}
  ```
- **Tool Calling & Reasoning**: Auto tool choice enabled with custom parsers for Muse Glimmer (`--enable-auto-tool-choice`, `--tool-call-parser muse_glimmer`, `--reasoning-parser muse_glimmer`).
- **Throughput & Caching**: `--enable-prefix-caching`, `--max-num-seqs 128`, and `--max-num-batched-tokens 8192` to support concurrent agent rollouts.
