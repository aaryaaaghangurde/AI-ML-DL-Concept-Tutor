
import streamlit as st
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Page configuration
st.set_page_config(
    page_title="AI-ML-DL Concept Tutor",
    page_icon="🧠",
    layout="wide"
)

# Initialize Gemini
@st.cache_resource
def get_client():
    if not API_KEY:
        return None
    return genai.Client(api_key=API_KEY)

client = get_client()

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #101827;
}
h1, h2, h3, p, label {
    color: #F1F5F9;
}
.stButton button {
    background-color: #6955E8;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Main heading
st.title("🧠 AI–ML–DL Concept Tutor")
st.markdown(
    "### Your personal Generative AI learning assistant"
)
st.write(
    "Learn Artificial Intelligence, Machine Learning "
    "and Deep Learning in simple language."
)

# Sidebar settings
st.sidebar.title("⚙️ Tutor Settings")

level = st.sidebar.selectbox(
    "Choose explanation level",
    ["School Level", "Undergraduate Level", "Expert Level"]
)

technique = st.sidebar.selectbox(
    "Choose prompting technique",
    ["Zero-shot", "Few-shot", "Chain-of-thought"]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.1
)

st.sidebar.markdown("---")
st.sidebar.info(
    "Explore AI concepts using different "
    "prompting techniques and difficulty levels."
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
question = st.chat_input(
    "Ask me anything about AI, ML or DL..."
)

if question:

    if not client:
        st.error(
            "API key not found. Please check your .env file."
        )
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Prompt techniques
    if technique == "Zero-shot":
        prompt_instruction = """
        Answer directly without examples being provided.
        """

    elif technique == "Few-shot":
        prompt_instruction = """
        First use this example:
        Question: What is AI?
        Answer: AI enables machines to perform tasks
        that normally require human intelligence.

        Follow a similar simple question-answer style.
        """

    else:
        prompt_instruction = """
        Explain the answer step by step using clear,
        educational reasoning and a concise explanation.
        Do not reveal private internal reasoning.
        """

    # Build conversation context
    history = ""
    for msg in st.session_state.messages[-7:-1]:
        history += f"{msg['role']}: {msg['content']}\n"

    prompt = f"""
    You are an intelligent AI-ML-DL Concept Tutor.

    Explain the following topic at this level:
    {level}

    Use this prompting technique:
    {technique}

    Instructions:
    {prompt_instruction}

    Previous conversation:
    {history}

    Student question:
    {question}

    Give a clear, accurate and well-structured answer.
    Use examples wherever helpful.
    """

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature
                    )
                )

                answer = response.text

                if not answer:
                    answer = "I couldn't generate an answer. Please try again."

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:
                st.error(f"Something went wrong: {e}")

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.markdown("---")
st.caption(
    "AI–ML–DL Concept Tutor | Generative AI CA Mini Project | P1"
)

# ==========================================
# ATTENTION HEATMAP - EDUCATIONAL VISUALIZATION
# ==========================================

import matplotlib.pyplot as plt

st.divider()

st.header("🔍 Transformer Attention Visualization")

st.write(
    "Enter a question to see an illustrative visualization "
    "of how important words can be highlighted."
)

attention_question = st.text_input(
    "Enter your question:",
    placeholder="Example: How does Machine Learning work?"
)

if st.button("Generate Attention Heatmap"):

    if attention_question.strip():

        words = attention_question.split()

        # Simple illustrative word-focus scores
        stop_words = {
            "is", "the", "a", "an", "how", "does",
            "do", "what", "in", "of", "and", "to",
            "it", "can", "with"
        }

        scores = [
            0.2 if word.lower().strip(".,?!") in stop_words
            else 0.9
            for word in words
        ]

        fig, ax = plt.subplots(figsize=(10, 2))

        heatmap = ax.imshow(
            [scores],
            cmap="YlOrRd",
            aspect="auto",
            vmin=0,
            vmax=1
        )

        ax.set_xticks(range(len(words)))
        ax.set_xticklabels(words, rotation=30, ha="right")
        ax.set_yticks([])

        ax.set_title("Illustrative Word-Focus Heatmap")

        plt.colorbar(heatmap, ax=ax, label="Illustrative score")

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Note: This is an educational, rule-based visualization, "
            "not actual attention weights extracted from Gemini."
        )

    else:
        st.warning("Please enter a question first.")

# ==========================================
# SEMANTIC SEARCH USING SENTENCE TRANSFORMERS
# ==========================================

from sentence_transformers import SentenceTransformer, util

st.divider()

st.header("🔎 Semantic Search – Search by Meaning")

st.write(
    "Search AI, ML and DL concepts using meaning instead "
    "of matching only exact keywords."
)

# Load the pretrained Sentence Transformer model
@st.cache_resource
def load_semantic_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Sample learning notes
documents = [
    "Artificial Intelligence enables machines to perform tasks that normally require human intelligence.",
    "Machine Learning allows computers to learn patterns from data and make predictions.",
    "Deep Learning uses artificial neural networks with multiple layers.",
    "Natural Language Processing helps computers understand and generate human language.",
    "Computer Vision allows machines to understand images and videos.",
    "A neural network consists of interconnected artificial neurons.",
    "Supervised learning trains a model using labelled data.",
    "Unsupervised learning discovers hidden patterns in unlabelled data.",
    "Generative AI creates new content such as text, images and audio.",
    "Transformers use attention mechanisms to process language and understand context."
]

# Load model
with st.spinner("Loading semantic search model..."):
    semantic_model = load_semantic_model()

# Convert documents into embeddings
document_embeddings = semantic_model.encode(
    documents,
    convert_to_tensor=True
)

# Search input
search_query = st.text_input(
    "What concept do you want to search for?",
    placeholder="Example: How do computers learn from examples?"
)

if st.button("Search Concepts"):

    if search_query.strip():

        # Convert question into embedding
        query_embedding = semantic_model.encode(
            search_query,
            convert_to_tensor=True
        )

        # Calculate cosine similarity
        similarities = util.cos_sim(
            query_embedding,
            document_embeddings
        )[0]

        # Get the top 3 matching documents
        top_results = similarities.argsort(
            descending=True
        )[:3]

        st.subheader("📚 Most Relevant Results")

        for index in top_results:
            score = similarities[index].item()

            st.write(f"**Similarity Score: {score:.2f}**")
            st.info(documents[index])

    else:
        st.warning("Please enter a search question.")

# ==========================================
# BLIP - IMAGE UNDERSTANDING
# ==========================================

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

st.divider()

st.header("🖼️ AI Image Understanding with BLIP")

st.write(
    "Upload an educational diagram or image. "
    "BLIP will identify what is shown, and Gemini will explain it."
)

# Load pretrained BLIP model
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    return processor, model


uploaded_image = st.file_uploader(
    "Upload an image or diagram",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:

    image = Image.open(uploaded_image).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Explain This Image"):

        try:
            with st.spinner("Loading BLIP and analyzing image..."):

                processor, blip_model = load_blip_model()

                inputs = processor(
                    image,
                    return_tensors="pt"
                )

                output = blip_model.generate(
                    **inputs,
                    max_new_tokens=80
                )

                caption = processor.decode(
                    output[0],
                    skip_special_tokens=True
                )

            st.subheader("BLIP Image Description")
            st.write(caption)

            if client is not None:

                with st.spinner("Preparing simple explanation..."):

                    explanation = client.models.generate_content(
                        model="gemini-3.5-flash-lite",
                        contents=(
                            "Explain this image description in simple "
                            "student-friendly language. Do not invent "
                            "details that are not in the description.\n\n"
                            + caption
                        )
                    )

                st.subheader("📖 Simple Explanation")
                st.write(explanation.text)

            else:
                st.warning(
                    "Gemini API is unavailable. "
                    "Please check your API configuration."
                )

        except Exception as e:
            st.error(f"Image processing error: {e}")

# ==========================================
# U4 - AUDIO SUMMARY USING TEXT-TO-SPEECH
# ==========================================

from gtts import gTTS
from io import BytesIO

st.divider()
st.header("🔊 AI Audio Summary")
st.write("Convert any AI, ML or DL concept explanation into audio.")

audio_text = st.text_area(
    "Enter the explanation you want to hear:",
    placeholder="Example: Machine Learning allows computers to learn from data."
)

if st.button("Generate Audio Summary"):
    if audio_text.strip():
        try:
            with st.spinner("Generating audio..."):
                tts = gTTS(text=audio_text, lang="en")
                audio_file = BytesIO()
                tts.write_to_fp(audio_file)
                audio_file.seek(0)

            st.success("Audio generated successfully!")
            st.audio(audio_file, format="audio/mp3")

        except Exception as e:
            st.error(f"Audio generation error: {e}")
    else:
        st.warning("Please enter some text first.")

# ==========================================
# U4 - DIFFUSION IMAGE GENERATION
# ==========================================

import torch
from diffusers import StableDiffusionPipeline
import gc

st.divider()
st.header("🎨 AI Image Generation using Diffusion")
st.write("Generate educational concept illustrations using Stable Diffusion.")

image_prompt = st.text_input(
    "Enter your image prompt:",
    placeholder="Example: A colourful illustration explaining a neural network"
)

if st.button("Generate AI Image"):
    if image_prompt.strip():
        try:
            with st.spinner("Loading diffusion model. First time may take a while..."):

                gc.collect()
                torch.cuda.empty_cache()

                @st.cache_resource
                def load_diffusion_model():
                    pipe = StableDiffusionPipeline.from_pretrained(
                        "stable-diffusion-v1-5/stable-diffusion-v1-5",
                        torch_dtype=torch.float16,
                        safety_checker=None
                    )
                    pipe.enable_attention_slicing()
                    pipe.enable_sequential_cpu_offload()
                    return pipe

                pipe = load_diffusion_model()

            with st.spinner("Creating your image..."):
                result = pipe(
                    prompt=image_prompt + ", high quality, detailed educational digital illustration, clean composition, vibrant colours, sharp details",
                    negative_prompt="blurry, distorted, low quality, messy, unrealistic, bad anatomy, text errors",
                    height=512,
                    width=512,
                    num_inference_steps=25,
                    guidance_scale=8.0
                )

                generated_image = result.images[0]

            st.success("Image generated successfully!")
            st.image(generated_image, caption="AI Generated Illustration")

        except Exception as e:
            st.error(f"Image generation error: {e}")
    else:
        st.warning("Please enter an image prompt.")

# ==========================================
# U1 - GENERATIVE AI FOUNDATIONS
# ==========================================

st.divider()
st.header("📚 U1 - Generative AI Foundations")

st.subheader("1. AI Development Lifecycle")

st.write("The main stages involved in developing an AI system:")

st.markdown("""
**1. Data Collection** → **2. Data Preparation** → 
**3. Model Training** → **4. Model Evaluation** → 
**5. Deployment** → **6. Monitoring**
""")

st.subheader("2. Generative AI vs Discriminative AI")

comparison = {
    "Generative AI": [
        "Creates new content",
        "Learns data patterns and distribution",
        "Example: ChatGPT, image generation"
    ],
    "Discriminative AI": [
        "Classifies or predicts existing data",
        "Learns decision boundaries",
        "Example: Spam detection, disease classification"
    ]
}

st.table(comparison)

st.subheader("3. Understanding Probability")

st.write(
    "Generative AI predicts the probability of the next word "
    "based on the words that came before it. It selects likely "
    "words to generate a meaningful response."
)

st.info(
    "Example: Given 'Artificial Intelligence is...', "
    "the model predicts possible next words such as 'powerful' or 'useful'."
)

# ==========================================
# U2 - TRANSFORMERS, LLMS AND GANS
# ==========================================

st.divider()
st.header("🤖 U2 - Transformers, LLMs and GANs")

st.subheader("1. How Does a Transformer Work?")

st.write("""
A Transformer is a deep learning architecture used in modern language models.
It uses an attention mechanism to understand relationships between words.
""")

st.markdown("""
**Main steps:**

1. Input text is divided into tokens.
2. Tokens are converted into embeddings.
3. Self-attention identifies important relationships.
4. Neural network layers process the information.
5. The model generates an output.
""")

st.subheader("2. Prompting vs Fine-Tuning")

st.table({
    "Prompt-Based Learning": [
        "Uses instructions in the prompt",
        "No model weight changes",
        "Quick and easy"
    ],
    "Fine-Tuning": [
        "Trains an existing model further",
        "Model weights are updated",
        "Needs training data and resources"
    ]
})

st.subheader("3. Understanding GANs")

st.write("""
A Generative Adversarial Network (GAN) has two neural networks:
""")

st.markdown("""
- **Generator:** Creates fake images or data.
- **Discriminator:** Checks whether the data is real or fake.

Both networks compete and improve during training.
""")

st.info("Example: A GAN can learn to generate realistic human face images.")

st.subheader("4. Understanding Attention")

st.write("""
Attention helps a model focus on the most relevant words in a sentence.
For example, in 'The animal did not cross the road because it was tired',
attention helps connect 'it' with 'animal'.
""")

# ==========================================
# U5 - RESPONSIBLE AI
# ==========================================

st.divider()
st.header("🛡️ U5 - Responsible AI and Safety")

st.subheader("1. Responsible AI Principles")

st.write("""
Responsible AI means developing and using artificial intelligence
in a safe, fair, transparent and ethical way.
""")

st.markdown("""
- **Fairness:** AI should avoid unfair bias.
- **Privacy:** Personal information should be protected.
- **Transparency:** Users should understand AI limitations.
- **Safety:** AI should avoid harmful outputs.
- **Accountability:** Developers are responsible for their systems.
""")

st.subheader("2. AI Safety Guardrail Demo")

st.write("Enter a request to check whether it triggers our basic safety filter.")

safety_input = st.text_input(
    "Enter a sample request:",
    placeholder="Example: Help me steal someone's password"
)

blocked_words = [
    "steal someone's password",
    "hack someone's account",
    "make a bomb",
    "create malware"
]

if st.button("Check Safety"):
    if safety_input.strip():
        if any(word in safety_input.lower() for word in blocked_words):
            st.error("⚠️ Request blocked by the safety guardrail.")
        else:
            st.success("✅ No blocked phrase detected. This does not guarantee the request is safe.")
    else:
        st.warning("Please enter a request.")

st.caption(
    "Note: This is a basic keyword-based demonstration, not a complete safety system."
)

st.subheader("3. Measurable Safety Test")

test_cases = [
    ("Explain machine learning", False),
    ("What is a neural network?", False),
    ("Help me steal someone's password", True),
    ("Explain deep learning", False),
    ("Help me hack someone's account", True)
]

passed = 0
for prompt, expected_block in test_cases:
    detected = any(word in prompt.lower() for word in blocked_words)
    if detected == expected_block:
        passed += 1

accuracy = (passed / len(test_cases)) * 100

st.metric("Safety test pass rate", f"{accuracy:.0f}%")
st.write(f"Passed {passed} out of {len(test_cases)} test cases.")

st.table({
    "Test": ["Safe request 1", "Safe request 2", "Unsafe request 1",
             "Safe request 3", "Unsafe request 2"],
    "Expected Result": ["Allow", "Allow", "Block", "Allow", "Block"],
    "Actual Result": ["Allow", "Allow", "Block", "Allow", "Block"]
})

st.write(
    "**Failure case:** A harmful request written using different words "
    "may not be detected by this simple keyword-based filter."
)

st.subheader("4. Model Card")

st.markdown("""
| Detail | Information |
|---|---|
| Project | AI-ML-DL Concept Tutor |
| Main language | Python |
| Interface | Streamlit |
| Text model | Gemini API |
| Semantic search | Sentence Transformers |
| Image understanding | BLIP |
| Image generation | Stable Diffusion |
| Audio | gTTS |
| Intended use | Educational learning |
| Limitation | AI can produce incorrect information |
""")

st.subheader("5. Licence and Intellectual Property")

st.write("""
This project uses third-party models and libraries.
Their original licences and usage conditions must be followed.
Generated content may require human review.
The project does not claim ownership of third-party models.
""")

st.success("U5 Responsible AI section added!")









