# Neural Networks: Zero to Hero

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python nanogpt/gpt_2.py
torchrun --standalone --nproc_per_node=8 nanogpt/gpt_2.py # for 8 GPUs
```
