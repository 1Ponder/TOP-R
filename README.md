# TOP-R: Tools Orchestration Privacy Risk

Code and dataset for the paper *"Agent Tools Orchestration Leaks More: Dataset, Benchmark, and Mitigation"*.

TOP-R is a privacy risk that arises when an LLM-based agent, executing a benign user instruction, aggregates individually innocuous fragments from multiple tool outputs and reconstructs sensitive information through cross-source reasoning. No single tool return reveals the secret — the leakage only emerges from their composition (the "mosaic effect").

TOP-Bench contains 300 validated samples across 5 privacy domains (PID, HMD, FAD, BAL, PCI) and 5 inference paradigms, plus a 100-sample diagnostic subset with social-context augmentation.

## Repo Structure

```
├── data/
│   └── test.json               # TOP-Bench dataset (300 samples)
├── code_run.py                 # Main 3-round evaluation pipeline
├── code_eva.py                 # LLM-as-judge scoring
├── infer_all.py                # C3 verification: all tools combined
├── infer_single.py             # C2 verification: single tool
├── infer_sensitivity.py        # C1 verification: conclusion sensitivity
├── api_config.txt              # Your API credentials (gitignored)
└── README.md
```

## Setup

```bash
pip install openai
```

All scripts talk to an OpenAI-compatible API endpoint (works with OpenAI, Alibaba DashScope, DeepSeek, vLLM, etc.). You just need to point them at the right `base_url`.

## Configuration

**`code_run.py`** reads credentials from a config file (default: `api_config.txt`):

```
key = sk-xxxxxxxxxxxx
base_url = "https://api.openai.com/v1"
model = "gpt-4o"
```

You can also override via CLI flags (`--api_key`, `--base_url`, `--model`) or the `OPENAI_API_KEY` environment variable.

**`code_eva.py`** and the three `infer_*.py` scripts read from environment variables:

```bash
export API_KEY="sk-xxxxxxxxxxxx"
export BASE_URL="https://api.openai.com/v1"
export MODEL="gpt-4o"
```

`code_eva.py` also accepts `--api_key`, `--base_url`, `--model` flags.

## Running the Evaluation

The full pipeline has two stages: run the agent, then score the results.

### 1. Run the 3-round pipeline

```bash
python code_run.py --dataset data/test.json --output run_result.json --config api_config.txt
```

This runs three rounds per sample:

- **R1 (Tool Selection):** the model picks which tools to use
- **R2 (Response Generation):** the model generates a response using the selected tools' data
- **R3 (Implicit Probing):** a classifier checks whether the tool outputs are sufficient to reconstruct the sensitive conclusion

Useful flags:

| Flag                | What it does                                     |
| ------------------- | ------------------------------------------------ |
| `--workers N`       | Parallel threads (default: 1)                    |
| `--stream`          | Use streaming API calls                          |
| `--enable_thinking` | Enable chain-of-thought (auto-enables streaming) |
| `--rpm N`           | Rate limit (requests per minute, 0 = no limit)   |
| `--start N`         | Skip first N entries                             |
| `--limit N`         | Only process N entries (0 = all)                 |
| `--save_reasoning`  | Save model reasoning traces (default: on)        |

### 2. Score with LLM-as-judge

```bash
python code_eva.py --test data/test.json --results run_result.json --output eval_result.json
```

This calls a judge model to score each entry on task completion and privacy leakage (explicit + implicit). The judge uses greedy decoding (`--temperature 0.0` by default) for reproducibility.

Output includes per-entry verdicts and aggregate stats (TC, Explicit Leak, Implicit Leak, OLR, H-Score).

## Data Verification Scripts

These three scripts verify that each benchmark sample satisfies the three formal conditions (C1–C3). They were used during dataset construction and can also serve as standalone analysis tools.

```bash
# C1: Is the malicious goal actually privacy-sensitive?
python infer_sensitivity.py data/test.json results_sensitivity.json

# C2: Can any SINGLE tool alone reconstruct the malicious goal? (should be NO)
python infer_single.py data/test.json results_single.json

# C3: Can ALL tools combined reconstruct the malicious goal? (should be YES)
python infer_all.py data/test.json results_all.json
```

All three support `WORKERS`, `STREAM`, `ENABLE_THINKING`, `RPM` env vars for parallelism and rate control.

`infer_single.py` additionally supports resumption — if interrupted, it writes partial results to a `.partial.jsonl` file and picks up from there on the next run. Set `PARTIAL_FILE` env var to customize the path.

## Mitigation Experiments

To reproduce the mitigation results (CIE, DCPE, MRCD) from the paper, modify the system prompt in `code_run.py` according to the templates in Appendix L of the paper. The mitigation strategies are implemented purely through system-prompt injection — no code changes needed.

## Dataset Format

Each entry in `test.json` follows this schema:

```json
{
  "metadata": {
    "id": 1,
    "domain": "HMD",
    "inference_chain": "..."
  },
  "user_goal": {
    "benign_goal": "Process my weekly expense report.",
    "malicious_goal": "The user is secretly interviewing at a competitor."
  },
  "available_tools": [
    {
      "tool_name": "BankStatement",
      "tool_id": "T1",
      "tool_description": "...",
      "parameters": ["..."],
      "return_data": "...",
      "noise_tool": false
    }
  ],
  "evaluation_benchmarks": {
    "success_criteria": "...",
    "sensitive_conclusion": "...",
    "basis_for_judgment": "..."
  }
}
```

## Citation

```bibtex
@misc{qiao2025agenttoolsorchestrationleaks,
      title={Agent Tools Orchestration Leaks More: Dataset, Benchmark, and Mitigation}, 
      author={Yuxuan Qiao and Dongqin Liu and Hongchang Yang and Wei Zhou and Songlin Hu},
      year={2025},
      eprint={2512.16310},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2512.16310}, 
}
```


