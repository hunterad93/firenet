import openai
import streamlit as st
from openai import OpenAI
import time

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
assistant_id = "asst_fNOMurA8WDwUTCQyo9gkzmET"

def ensure_single_thread_id():
    if "thread_id" not in st.session_state:
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id
    return st.session_state.thread_id

def get_filename(file_id):
    try:
        # Retrieve the file metadata from OpenAI
        file_metadata = client.files.retrieve(file_id)
        # Extract the filename from the metadata
        filename = file_metadata.filename
        return filename
    except Exception as e:
        print(f"Error retrieving file: {e}")
        return None
    
def format_citation(annotation):
    file_id = annotation.file_citation.file_id
    filename = get_filename(file_id)
    if filename:
        citation_info = f"[Citation from {filename} "
    else:
        citation_info = "[Citation from an unknown file"
    return citation_info

def stream_generator(prompt, thread_id):
    # Create the initial message
    message = client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=prompt
    )

    # Start streaming the response
    with st.spinner("Wait... Generating response..."):
        stream = client.beta.threads.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            stream=True,
            max_prompt_tokens=20000
        )
        partial_response = ""
        for event in stream:
            if event.data.object == "thread.message.delta":
                for content in event.data.delta.content:
                    if content.type == 'text':
                        text_value = content.text.value
                        annotations = content.text.annotations
                        if annotations:
                            for annotation in annotations:
                                citation_info = format_citation(annotation)
                                indexes = f"from index {annotation.start_index} to {annotation.end_index}]"
                                text_value = f" **{citation_info + indexes}**"
                        partial_response += text_value
                        words = partial_response.split(' ')
                        for word in words[:-1]:  # Yield all but the last incomplete word
                            yield word + ' '
                        partial_response = words[-1]  # Keep the last part for the next chunk
            else:
                pass
        if partial_response:
            yield partial_response  # Yield any remaining text

# Streamlit interface
st.set_page_config(page_icon="🔥")
st.title("🔥 Discuss Firenet With ChatGPT 🔥")

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Enter your message")
# Streamlit interface
if prompt:
    thread_id = ensure_single_thread_id()
    with st.chat_message("user"):
        st.write(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response_container = st.empty()  # Create an empty container for the response
        full_response = ""
        for chunk in stream_generator(prompt, thread_id):
            full_response += chunk
            # Update the container with the latest full response, adding fire emojis
            response_container.markdown("🔥 " + full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})