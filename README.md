# latent-model-recreation

## Running instructions

```bash
cp .env.example .env # fill in the env variables
uv sync --extra cu124 # or uv sync --extra cpu
uv run src/main.py -h
# to run training even when ssh is closed
nohup uv run src/main.py train -e 50 > output.log &
```

# Report

```bash
cd report
pandoc report.md -o report.pdf -C
```
