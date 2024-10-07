import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    return model, tokenizer

# Function to generate text
def generate_text(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs.input_ids, max_length=100, do_sample=True)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Streamlit interface
st.title("GPT-Neo 125M Text Generator")
st.write("Enter a prompt, and the model will generate a continuation:")

# Input prompt
prompt = st.text_area("Prompt:", value="Once upon a time", height=150)

# Generate text when the button is clicked
if st.button("Generate"):
    with st.spinner("Generating text..."):
        model, tokenizer = load_model()
        result = generate_text(prompt, model, tokenizer)
        st.write(result)
