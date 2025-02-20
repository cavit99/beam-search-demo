import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr

# Load model and tokenizer once at startup
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
print(f"Model loaded on: {model.device}")

# Define the generation function
def generate_text(prompt, max_new_tokens, num_beams):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[-1]  
    # Greedy search
    start_time = time.time()
    outputs_greedy = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        num_beams=1,
        do_sample=False,
    )
    greedy_time = time.time() - start_time
    # Remove the prompt tokens from the output
    generated_tokens_greedy = outputs_greedy[0][input_length:]
    generated_text_greedy = tokenizer.decode(generated_tokens_greedy, skip_special_tokens=True)
    
    # Beam search
    start_time = time.time()
    outputs_beam = model.generate(
        **inputs,
        num_beams=int(num_beams),
        num_return_sequences=1,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
    )
    beam_time = time.time() - start_time
    # Remove the prompt tokens as above
    generated_tokens_beam = outputs_beam[0][input_length:]
    generated_text_beam = tokenizer.decode(generated_tokens_beam, skip_special_tokens=True)
    
    # Prepare outputs for better display formatting
    greedy_details = (
        f"**Strategy:** Picks the most probable token at each step (deterministic).\n\n"
        f"**Time:** {greedy_time:.2f} seconds"
    )
    
    beam_details = (
        f"**Strategy:** Explores {num_beams} beams concurrently and returns the top candidate.\n\n"
        f"**Time:** {beam_time:.2f} seconds"
    )
    
    return greedy_details, generated_text_greedy, beam_details, generated_text_beam

with gr.Blocks() as demo:
    # Informational header to help users understand the demo
    gr.Markdown(
        "# Beam Search Demo\n\n"
        "This demo shows how two different text generation strategies work using the Qwen2.5-0.5B model. "
        "The left side uses **greedy search**, which picks the most probable token at every generation step (deterministic), "
        "while the right side uses **beam search**, which explores multiple beams concurrently to choose the most likely "
        "sequence of tokens.\n\n"
        "**Important:** This model works best with prompts that need completion rather than question-answering. For example, "
        "instead of 'What is the capital of France?', use prompts like 'The capital of France is' or 'Here is a story about:'\n\n"
        "Use the controls below to enter your prompt, adjust the maximum number of newly generated tokens, and set the "
        "number of beams for beam search. The results for both strategies are displayed side by side for easy comparison.\n\n"
    )

    # Input components in a single column at the top
    with gr.Column():
        gr.Markdown("## Input")
        prompt_input = gr.Textbox(label="Prompt", value="Here is a funny love letter for you:")
        max_tokens_input = gr.Slider(minimum=1, maximum=100, step=1, label="Max new tokens", value=50)
        num_beams_input = gr.Slider(minimum=1, maximum=20, step=1, label="Number of beams", value=10)
        generate_btn = gr.Button("Generate")
    
    with gr.Row():
        with gr.Column():
            greedy_details_output = gr.Markdown(label="Greedy Search Details")
            greedy_textbox_output = gr.Textbox(label="Greedy Search Generated Text", lines=10)
        with gr.Column():
            beam_details_output = gr.Markdown(label="Beam Search Details")
            beam_textbox_output = gr.Textbox(label="Beam Search Generated Text", lines=10)
    
    # Connect the button click event to the generation function
    generate_btn.click(
        generate_text,
        inputs=[prompt_input, max_tokens_input, num_beams_input],
        outputs=[greedy_details_output, greedy_textbox_output, beam_details_output, beam_textbox_output]
    )

if __name__ == "__main__":
    demo.launch()