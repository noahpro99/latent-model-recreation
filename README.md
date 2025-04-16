# latent-model-recreation

## Running instructions

```bash
cp .env.example .env # fill in the env variables
uv sync --extra cu118 # or uv sync --extra cpu
uv run src/main.py -h
# to run training even when ssh is closed
nohup uv run src/main.py train -e 50 > output.log &
```

# Task Assignments

## dataset

umur12

- some dataset for pretraining simple one big ish one
  - https://huggingface.co/datasets/nampdn-ai/tiny-strange-textbooks
- https://huggingface.co/datasets/nvidia/OpenMathInstruct-1

## train latent model

noah & shanb

- pretrain once with no recurrence
- benchmark it with recurrence in latent space vs recurrence in input/output space

## Slides

phamlt

- explain what we are doing
- explain what we have done so far
  - dataset
  - model code so far (maybe pre training results)

## pitching

kingsley
