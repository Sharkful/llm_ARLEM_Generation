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

Windows (Conda):
```
git clone ...
cd llm_ARLEM_Generation
conda create -n ar-experiments  python=3.11
conda activate ar-experiments              
pip install -r requirements.txt            
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
