import streamlit as st
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from datetime import datetime
import tempfile
import torch
import time
import fitz
import docx
from docx import Document
from fpdf import FPDF
import anthropic
from dotenv import load_dotenv
import os

# --- Load environment variables ---
load_dotenv()

# --- Setup and Config ---
st.set_page_config(page_title="VIDA Multi-Model Chatbot", layout="wide")

# --- Sidebar Image ---
st.logo("https://logodix.com/logo/951536.png",
    size = "large",
    icon_image = "https://logodix.com/logo/951515.png"
)

st.title("VIDA Chatbot Prototype")
st.header("Multi-Model Chatbot Interface")

device = "cuda" if torch.cuda.is_available() else "cpu"
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY")

# --- Sidebar Model Selector ---
model_choice = st.sidebar.selectbox(
    "Choose a Model",
    ["SmolVLM2 - Video & Image Analysis", "SmolDocling - Document Handler", "Claude"]
)

model_labels = {
    "SmolVLM2 - Video & Image Analysis": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "SmolDocling - Document Handler": "ds4sd/SmolDocling-256M-preview",
    "Claude": "claude-3-haiku-20240307"
}

# --- Loaders ---
@st.cache_resource
def load_smolvlm2():
    processor = AutoProcessor.from_pretrained(model_labels["SmolVLM2 - Video & Image Analysis"])
    model = AutoModelForImageTextToText.from_pretrained(
        model_labels["SmolVLM2 - Video & Image Analysis"],
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    return model, processor

@st.cache_resource
def load_docling():
    processor = AutoProcessor.from_pretrained(model_labels["SmolDocling - Document Handler"])
    model = AutoModelForImageTextToText.from_pretrained(
        model_labels["SmolDocling - Document Handler"],
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    return model, processor

# --- Save Functions ---
def save_as_docx(text, filename="generated_document.docx"):
    doc = Document()
    doc.add_paragraph(text)
    doc.save(filename)
    return filename

def save_as_pdf(text, filename="generated_document.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in text.split("\n"):
        pdf.cell(200, 10, txt=line, ln=True)
    pdf.output(filename)
    return filename

# --- Claude Setup ---
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def call_claude(prompt, model="claude-3-haiku-20240307"):
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system="You are a helpful assistant.",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# --- Text Extraction ---
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

# --- Docling Response Gen ---
def generate_response_docling(text_input=None, image_input=None, model=None, processor=None):
    content = []
    if image_input:
        image = Image.open(image_input)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp, format="PNG")
            content.append({"type": "image", "url": tmp.name})
    if text_input:
        content.append({"type": "text", "text": text_input})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=1024)
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]

# --- SmolVLM2 ---
def generate_response_smolvlm2(text_input=None, image_input=None, video_input=None, model=None, processor=None):
    content = []
    if video_input:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_input.read())
            content.append({"type": "video", "path": tmp.name})
    if image_input:
        image = Image.open(image_input)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            image.save(tmp, format="PNG")
            content.append({"type": "image", "url": tmp.name})
    if text_input:
        content.append({"type": "text", "text": text_input})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=1024)
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]

# --- Interface and Chat ---
if "multimodal_messages" not in st.session_state:
    st.session_state["multimodal_messages"] = [{
        "role": "assistant",
        "content": "Hello! I'm here to help. Select a model and send your instructions.",
        "timestamp": datetime.now().strftime("%I:%M %p"),
    }]
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

for msg in st.session_state["multimodal_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        st.caption(f"Sent at {msg['timestamp']}")

user_text = st.chat_input("Type your instructions here:")

# --- Model-specific inputs ---
image_file = None
video_file = None

if model_choice == "Claude":
    st.info("⚠️ Claude currently only supports text input. Image and video uploads are hidden.")

elif model_choice == "SmolDocling - Document Handler":
    st.info("Currently testing this model. Please upload the picture of a document: PNG/JPG")
    st.info("Previous model was microsoft/phi-2")
    image_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"], key=f"img_{st.session_state['uploader_key']}")

elif model_choice == "SmolVLM2 - Video & Image Analysis":
    image_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"], key=f"img_{st.session_state['uploader_key']}")
    video_file = st.file_uploader("Upload a video (optional)", type=["mp4", "avi", "mov"], key=f"vid_{st.session_state['uploader_key']}")

if user_text and user_text.strip():
    ts = datetime.now().strftime("%I:%M %p")
    st.session_state["multimodal_messages"].append({"role": "user", "content": user_text, "timestamp": ts})

    with st.spinner("Generating response..."):
        if model_choice == "SmolVLM2 - Video & Image Analysis":
            model, processor = load_smolvlm2()
            response = generate_response_smolvlm2(user_text, image_file, video_file, model, processor)

        elif model_choice == "SmolDocling - Document Handler":
            model, processor = load_docling()
            response = generate_response_docling(user_text, image_file, model, processor)

        elif model_choice == "Claude":
            try:
                response = call_claude(user_text)
            except Exception as e:
                response = f"❌ Claude API Error: {e}"

    st.session_state["multimodal_messages"].append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().strftime("%I:%M %p")
    })

    st.session_state["uploader_key"] += 1
    st.rerun()

if st.button("🗑️ Clear Chat"):
    st.session_state["multimodal_messages"] = []
    st.rerun()
