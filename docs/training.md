# SFT and RL workflow

## SFT

`sft/train_tc2.py` implements completion-only, weighted single-GPU QLoRA training. Prompt labels are masked, completion tokens contribute to loss, and per-sample weights are applied. The reference setup uses NF4 double quantization, bfloat16 compute, and LoRA on attention/MLP projections (rank 32, alpha 64, dropout 0.05).

Provide your own stage dataset and fast tokenizer. `sft/prepare_dataset.py` defines the stage encoding contract. The source/tokenizer manifests are integrity checks, not permission to redistribute private training data. Trainer help is available with:

```bash
python -B sft/train_tc2.py --help
```

## RL

Start from the pretrained backbone plus a pinned, completed SFT LoRA adapter. RL updates that shared adapter; SFT and RL evaluation adapters are alternatives, not two adapters merged together.

```text
Freeze behavior policy
→ collect four rollouts per training question
→ stage Judge and incremental evidence coverage
→ independent semantic audit
→ authority/token/group validation
→ compile rewards and advantages
→ clipped policy optimization
→ save/check checkpoint
→ held-out paired evaluation
```

`training/train.py` requires a bound batch, model path, starting adapter and separate trusted authority artifact. Inspect its CLI rather than constructing ad-hoc records. The `--preflight` path validates compilation without loading model weights. Real training additionally checks behavior replay parity.

The reference defaults are learning rate `1e-6`, clipping epsilon `0.2`, target KL `0.02` and parity tolerance `0.05`. These are starting parameters, not an empirically optimal prescription. GPU trainers use Linux locking primitives; use Linux/WSL and install a CUDA-compatible PyTorch build first. Optional dependencies are listed in `pyproject.toml`; exact deployment versions must be pinned and tested separately.

## Planned collection schedule

First accept a small complete batch, then target **50 questions × 4 rollouts = 200 trajectories per collection/scoring cycle**. Preserve per-trajectory capture receipts and resumable collection state; never mix policy identities within a behavior batch. Incomplete question groups remain blocked.

Collection cycles are not optimizer steps. The exported trainer currently steps by question group. A true 50-question macro-update needs explicit gradient accumulation, loss normalization, memory scheduling and checkpoint semantics; do not label it implemented merely because 50 questions were collected.

Auditing can happen after collection and scoring. A correction must be versioned under a trusted scorer/audit authority and trigger recompilation/revalidation; manually editing scores in a batch is not an authorized training signal. Changing prompts, adapters or decoding processors also changes identities and invalidates inappropriate cache/replay reuse.
