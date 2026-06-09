# 🚀 Getting Started

## 1. Environment Setup

To set up the Python environment and install dependencies (instructor, pydantic, dotenv), run the setup script for your operating system:

Windows (PowerShell):
```PowerShell

./setup_windows.ps1
```
Note: If you get a permission error, run Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process first.

macOS / Linux:
```Bash

chmod +x setup_unix.sh
./setup_unix.sh
```

## 2. Manual Installation

If you prefer to do it manually:
| OS | Create Venv | Activate | Install |
| --- | --- | --- | --- |
| Windows | `python -m venv .venv` | `.\.venv\Scripts\Activate.ps1` | `pip install -r requirements.txt` |
| macOS | `python3 -m venv .venv` | `source .venv/bin/activate` | `pip install -r requirements.txt` |

## 3. Configuration

The .env file is ignored by git for security. Create a file named .env in the root directory and add your keys:
Plaintext
```
OPENAI_API_KEY=your_key_here
GEMINI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## 4. Benchmarking LLM Lab Generation

The main entry point for evaluating models is the benchmark runner at
[Code/Testing/benchmark.py](Code/Testing/benchmark.py). It builds a prompt for a
lab topic, calls a model to generate a validated AR lab spec, and records
tokens, cost, retries, and structural metrics.

```powershell
# From the project root, with the venv active:
python "Code/Testing/benchmark.py" --list-models
python "Code/Testing/benchmark.py" --model claude-haiku-4.5 --lab phases_of_the_moon --level L2
```

For the full command-line reference — every flag, the L1–L4 levels, structure
and spec modes, the model registry, and where output is written — see
[Code/Testing/BENCHMARK.md](Code/Testing/BENCHMARK.md).

To collate one or more sweeps into a shareable summary report (an executed
notebook + HTML, one section per lab topic with before/after comparisons), use
[Code/Testing/build_summary_report.py](Code/Testing/build_summary_report.py) —
see [Code/Testing/SUMMARY_REPORT.md](Code/Testing/SUMMARY_REPORT.md).
