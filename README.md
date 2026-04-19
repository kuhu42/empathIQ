# README

This thing kind of works, but I don't like it.

## To run

---

```bash
cd empath-iq
```

### Please use uv

---

```bash
uv init
uv venv
source .venv/bin/activate
uv add -r requirements.txt
uv run app.py
```

### Or if you are using pip

---

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 app.py
```
