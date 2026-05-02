# Medical & Legal Document Summarizer

An AI summarizer for medical research papers and legal documents, built by fine-tuning T5-Small with LoRA (Low-Rank Adaptation). Trained on PubMed and BillSum datasets.

## Features

- Summarizes both medical and legal text
- Lightweight model (~10MB adapter weights)
- Gradio web interface
- Runs on CPU or GPU

## Project Structure

| File | Description |
|------|-------------|
| `app.py` | Gradio web app |
| `train_model.py` | Training script (T5 + LoRA) |
| `requirements.txt` | Python dependencies |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

Place the trained model folder (`adapter_model.bin` + `adapter_config.json`) inside `model_files/`.

## Usage

**Run the web app:**
```bash
python app.py
```

**Train from scratch** (~3–4 hours on GPU):
```bash
python train_model.py
```

> If not using Google Colab, update `OUTPUT_DIR` in `train_model.py` to a local path.

## Contributing

Open an issue or submit a pull request.

