# Kenchi 1

Kenchi (謙知, "humble knowledge") is a small language model I made, with about
500 million parameters. The goal of Kenchi is that it is the smallest model that
can use tools and tell you when it does not know.

It is only ~500M parameters so it has many limitations.

Made by Rex Hall (rexmhall09) with Lor, using Fable 5.2, Astra 6.2, and Fern 0.1.
Code and weights are MIT.

## Run it

The weights are on [Hugging Face](https://huggingface.co/rexmhall09/kenchi).
You need Python 3.10 or later.

```sh
git clone https://github.com/rexmhall09/kenchi.git
python -m pip install torch numpy sentencepiece safetensors huggingface_hub
hf download rexmhall09/kenchi --local-dir kenchi-hf
cd kenchi
python run.py --weights ../kenchi-hf/model.safetensors --config ../kenchi-hf/config.json
```

This opens a chat in the terminal. Kenchi searches Wikipedia, so it needs the
internet. `/new` starts a new conversation and `--verbose` shows what it read.

On a Mac with Apple Silicon you can use MLX instead
(`pip install mlx numpy sentencepiece`, then `./chat.sh`).

`serve.py` takes the same `--weights` and `--config` and runs a small server
that speaks the OpenAI chat format, so other programs can give Kenchi their own
tools.

## Limitations

- It can be wrong, and wrong answers still come with a source. Open the link.
- Follow-up questions ("where was he born?") mostly fail. Ask the full
  question again.
- It is not a good general chatbot.
- English only.
- 2,048 tokens of context.

## Training

The training code and configs are in `kenchi/` and `train/`. Run
`python -m train.pretrain_run --help` or `python -m train.sft_run --help`
for the entry points. Training data and pretrained checkpoints are not included.

## Data

Built with Qwen: some of the training data was written by Qwen models. The
search results used in training are Wikipedia text (CC BY-SA 3.0).
[DATA.md](DATA.md) has the full sources and citations.
