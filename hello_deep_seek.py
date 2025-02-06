from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Prepare input text
input_text = "In which continent do we have Brazil"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text
outputs = model.generate(**inputs, max_length=50)  # Adjust max_length as needed

# Decode the generated text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)