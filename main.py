import streamlit as st
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModelForCausalLM, AutoTokenizer
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


# --- Setup and Config ---
st.set_page_config(page_title="VIDA Multi-Model Chatbot", layout="wide")

st.title("VIDA Chatbot Prototype")
st.header("Multi-Model Chatbot Interface")

device = "cuda" if torch.cuda.is_available() else "cpu"
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]

# --- Sidebar Model Selector ---
model_choice = st.sidebar.selectbox(
    "Choose a Model",
    [
        "SmolVLM2 - Video & Image Analysis",
        "Phi-2 - Document Handler",
        "Claude"
    ]
)

model_labels = {
    "SmolVLM2 - Video & Image Analysis": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "Phi-2 - Document Handler": "microsoft/phi-2",
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
def load_phi2():
    model = AutoModelForCausalLM.from_pretrained(
        model_labels["Phi-2 - Document Handler"],
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_labels["Phi-2 - Document Handler"])
    return model, tokenizer

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

# --- Phi-2 Text Gen ---
def generate_response_phi2(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    lines = raw_text.splitlines()
    cleaned = [line for line in lines if not line.strip().startswith("INPUT")]
    return "\n".join(cleaned).replace("##OUTPUT", "").replace("OUTPUT:", "").replace("OUTPUT####", "").strip()

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

# --- Text Extraction ---
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)

# --- Session State ---
if "multimodal_messages" not in st.session_state:
    st.session_state["multimodal_messages"] = [{
        "role": "assistant",
        "content": "Hello! I'm here to help. Select a model and send your instructions.",
        "timestamp": datetime.now().strftime("%I:%M %p"),
    }]
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "document_text" not in st.session_state:
    st.session_state["document_text"] = ""

# --- Chat History ---
for msg in st.session_state["multimodal_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        st.caption(f"Sent at {msg['timestamp']}")

# --- Inputs ---
user_text = st.chat_input("Type your instructions here:")
image_file = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg"], key=f"image_{st.session_state['uploader_key']}")
video_file = st.file_uploader("Upload a video (optional)", type=["mp4", "avi", "mov"], key=f"video_{st.session_state['uploader_key']}")
document_file = None

if model_choice == "Phi-2 - Document Handler":
    document_file = st.file_uploader("Upload a document (PDF, DOCX, or TXT)", type=["pdf", "txt", "docx"], key=f"doc_{st.session_state['uploader_key']}")
    if document_file:
        if document_file.type == "application/pdf":
            extracted_text = extract_text_from_pdf(document_file)
        elif document_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            extracted_text = extract_text_from_docx(document_file)
        else:
            extracted_text = document_file.read().decode("utf-8")
        st.session_state["document_text"] = extracted_text
        st.success("Document uploaded and text extracted!")

# --- Generate Response ---
if user_text and user_text.strip():
    ts = datetime.now().strftime("%I:%M %p")
    summary = f"Instructions: {user_text}, Image: {bool(image_file)}, Video: {bool(video_file)}"
    st.session_state["multimodal_messages"].append({"role": "user", "content": summary, "timestamp": ts})

    with st.spinner("Processing your input..."):
        start = time.time()

        if model_choice == "SmolVLM2 - Video & Image Analysis":
            model, processor = load_smolvlm2()
            response = generate_response_smolvlm2(user_text, image_file, video_file, model, processor)

        elif model_choice == "Phi-2 - Document Handler":
            model, tokenizer = load_phi2()
            if st.session_state["document_text"]:
                filtered = "\n".join(
                    line for line in st.session_state["document_text"].splitlines()
                    if not line.strip().endswith("?") and not line.strip().lower().startswith("q:")
                )
                context = (
                    "The following text is ONLY background information. "
                    "Do not answer questions from within the document. "
                    "Answer only the specific user question below.\n\n"
                    f"Document:\n{filtered[:1000]}\n\n"
                    f"Question:\n{user_text}\n\n"
                    "Answer:"
                )
                response = generate_response_phi2(context, model, tokenizer)
            else:
                response = generate_response_phi2(user_text, model, tokenizer)

        elif model_choice == "Claude":
            try:
                response = call_claude(user_text)
            except Exception as e:
                response = f"❌ Claude API Error: {e}"

        elapsed = time.time() - start

    st.session_state["multimodal_messages"].append({
        "role": "assistant",
        "content": (
            f"**Model: {model_labels[model_choice]}**\n\n"
            f"{response}\n\n"
            f"_Processing time: {elapsed:.2f} seconds._"
        ),
        "timestamp": datetime.now().strftime("%I:%M %p")
    })

    st.session_state["last_response"] = response

    st.session_state["uploader_key"] += 1
    st.rerun()

# --- Clear Chat ---
st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
with st.container():
    if st.button("🗑️ Clear Chat"):
        st.session_state["multimodal_messages"] = []
        st.session_state["document_text"] = ""
        st.rerun()


