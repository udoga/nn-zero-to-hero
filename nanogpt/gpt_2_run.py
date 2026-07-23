from pathlib import Path
import tiktoken
import torch
from gpt_2 import GPT, GPTConfig


def main():
    config = GPTConfig(ctx_length=1024, emb_size=768, block_count=12, head_count=12, vocab_size=50304)
    model = GPT(config)
    model.load_state_dict(torch.load(Path(__file__).resolve().parents[1] / 'gpt_2.pt', map_location='cpu'))
    encoder = tiktoken.get_encoding('gpt2')
    input_token_ids = torch.tensor(encoder.encode("Hello, I'm a language model,"), dtype=torch.long)
    output_token_ids = model.generate(input_token_ids.unsqueeze(0), new_token_count=300)[0]
    print(encoder.decode(output_token_ids.tolist()))


if __name__ == '__main__':
    main()
