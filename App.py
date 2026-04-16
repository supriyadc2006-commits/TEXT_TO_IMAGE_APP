import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.set_page_config(page_title="Text to Image Generator")

st.title("🖼️ Text to Image Generator")

@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        pipe.to("cuda")
    return pipe

pipe = load_model()

prompt = st.text_input("Enter your prompt:")

if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            image = pipe(prompt).images[0]
            st.image(image, caption=prompt)
    else:
        st.warning("Please enter a prompt")
