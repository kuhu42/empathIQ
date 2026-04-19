# README

This thing kind of works, but I don't like it.

## Get the project

1. Clone the repo

```bash
git clone https://github.com/kuhu42/empathIQ.git empath-iq
cd empath-iq
```

2. Switch branches

```bash
git checkout dev
```

3. Pull latest changes

```bash
git pull origin dev
```

## Please use uv

```bash
uv init
uv venv
source .venv/bin/activate
uv add -r requirements.txt
uv run app.py
```

## Or if you are using pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```
