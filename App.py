import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.title("🖼️ Text to Image Generator")

@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32   # ✅ safer for CPU (Streamlit Cloud)
    )

    pipe = pipe.to("cpu")  # ✅ force CPU (important for deployment)
    pipe.safety_checker = None  # ✅ avoid safety-related crashes

    return pipe

pipe = load_model()

prompt = st.text_input("Enter your prompt:")

if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            result = pipe(prompt, num_inference_steps=20)  # ✅ safer steps
            image = result.images[0]
            st.image(image, caption=prompt)
    else:
        st.warning("Please enter a prompt")
