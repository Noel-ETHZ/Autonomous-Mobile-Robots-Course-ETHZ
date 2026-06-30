# Installation
## Using `uv` (Recommended)
Simply run
```bash
uv sync
uv pip install -e .
```

## Using `venv` (only tested with python 3.12)
Run
```bash
pyhton -m venv venv
source venv/bin/activate
pip install -e .
```


# Running the scripts
## Using `uv`

```bash
uv run amr_ex_12/e_12_task_*.py
```

## Using `venv` (only tested with python 3.12)

```bash
source venv/bin/activate && python amr_ex_12/e_12_task_*.py
```
