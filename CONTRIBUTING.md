# Contributing to LipRead

Thank you for your interest in contributing to **LipRead**! We welcome contributions to preprocessing, neural architectures, decoding strategies, evaluation, and documentation.

---

## 1. Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/LipRead.git
   cd LipRead
   ```

2. **Create a Python 3.10+ virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify your environment:**
   ```bash
   python -m src.utils.check_environment
   pytest -q
   ```

---

## 2. Code Quality & Standards

- **Python Version:** Python 3.10+ (tested on Python 3.12).
- **Type Annotations:** Use Python type hints (`typing` module) across all public interfaces and methods.
- **Formatting & Style:** Follow PEP 8 guidelines. Keep line lengths reasonable (typically $\le 100$ characters).
- **Docstrings:** Document all modules, classes, and functions using standard Google-style or NumPy-style docstrings.
- **Error Handling:** Avoid bare `except:`. Provide descriptive logging and custom exception types where appropriate.

---

## 3. Running Automated Tests

All changes must pass the test suite before being merged:

```bash
# Run the complete test suite
pytest -v

# Run a specific test module
pytest tests/test_model.py -v
```

CI workflows run on GitHub Actions in a CPU-only environment without requiring an external GPU, external drives, or proprietary datasets. Ensure any new unit tests use synthetic tensors or fixtures.

---

## 4. Dataset & Checkpoint Safety Rules

To keep the repository clean, legal, and lightweight:

1. **NEVER Commit Raw or Processed Video Data:**
   - Do NOT commit `.mpg`, `.mp4`, `.jpg`, `.png`, or `.pt` tensor caches to git.
   - Raw datasets (such as EGCLLC or GRID) must remain outside version control.
2. **NEVER Commit Model Checkpoint Weights:**
   - Do NOT commit `.pt`, `.pth`, `.bin`, `.onnx`, or `.ckpt` weights.
   - Use external storage or release artifacts for model weights.
3. **NEVER Commit Secrets or Environment Files:**
   - Never commit `.env` or files containing API keys, tokens, or personal paths.
   - Use `.env.example` as a template for configurable variables.

---

## 5. Pull Request Guidelines

1. **Branch Naming:**
   - Feature: `feature/your-feature-name`
   - Bugfix: `fix/issue-description`
   - Documentation: `docs/summary-of-changes`
2. **Commit Messages:**
   - Write clear, imperative commit messages (e.g., `Add in-memory cache lookup to dataset adapter`).
   - Do not reference deprecated project version tags in commit headers.
3. **Pull Request Checklist:**
   - [ ] Unit tests added or updated for new functionality.
   - [ ] All 130+ unit tests pass locally (`pytest`).
   - [ ] Code is formatted and typed.
   - [ ] Documentation updated if CLI flags, APIs, or configurations changed.
   - [ ] No large files, checkpoints, or secrets committed.
