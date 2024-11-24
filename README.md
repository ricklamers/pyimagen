# PyImagen
PyImagen is a desktop GUI app for generating images using a variety of models via the Fal.ai API, with prompt generation by Groq's fast LLM inference.

# Inspiration
[@rikarends](https://x.com/rikarends) [Makepad](https://github.com/makepad/makepad) based version. 

[@ronaldmannak](https://x.com/ronaldmannak)'s Flux AI Studio.

<img alt="UI" src="https://github.com/user-attachments/assets/3c6748f1-b0ad-4c49-8baa-ce9d6e7d4ee6">

## Features

- Generate images from text prompts utilizing multiple AI models
- Create random prompts using Groq's LLM
- Enhance prompts with additional creative details using Groq's LLM
- Maintain an image history with thumbnail previews
- Copy images directly to the clipboard
- Save images and their metadata locally for future reference

## Available Models

- FLUX 1 Schnell
- Recraft V3 
- AuraFlow
- FLUX 1.1 Pro Ultra
- FLUX 1.1 Pro
- Stable Cascade
- Fast Turbo Diffusion
- Fast LCM Diffusion
- And more...

## Requirements

- Python 3.10 or higher
- Groq API key
- Fal.ai API key

## API keys
Set `GROQ_API_KEY` and `FAL_KEY` in `.env` alternatively, enter them when the application starts.

## Installation

### PYPI
```bash
pip install pyimagen
pyimagen
```

### From Source

1. Clone the repository and `cd` into it
2. Install the necessary dependencies:

```bash
# Optional
python -m venv .venv && source .venv/bin/activate 
pip install -e .
python app.py
```
