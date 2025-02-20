# Beam Search vs Greedy Search Demo

Welcome to the Beam Search vs Greedy Search Demo repository! This educational demo illustrates two fundamental text generation strategies in natural language processing using the Qwen2.5-0.5B model from Hugging Face.

## Overview

This demo compares:
- **Greedy Search**: Picks the most probable token at every generation step (deterministic).
- **Beam Search**: Explores multiple beams concurrently and returns the top candidate, often achieving more coherent or creative outputs.

The demo is implemented using [Gradio](https://gradio.app/), which provides an interactive web interface to explore these methods side by side.

## Features

- **Interactive Interface:** Enter a prompt, adjust generation parameters (maximum new tokens, number of beams), and view results for both approaches.
- **Side-by-Side Comparison:** See generation details such as computation time and strategy explanation for both greedy and beam search.
- **Educational:** Gain insights into the strengths and trade-offs between different text generation strategies.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/cavit99/beam-search-demo.git
    cd beam-search-demo
    ```

2. **Setup a virtual environment (recommended):**

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use: env\Scripts\activate
    ```

3. **Install the required packages:**

    If a `requirements.txt` file is provided, use:
    ```bash
    pip install -r requirements.txt
    ```

    Otherwise, manually install the dependencies:
    ```bash
    pip install transformers gradio
    ```

## Usage

To run the demo locally, execute:

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
