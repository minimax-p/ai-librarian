import gradio as gr


def search_query(query, history, sources, pre_prompts):
    # This is a placeholder for search functionality.
    response = f"Searching for: '{query}'"
    history.append(f"User: {query}")
    sources.append("Source: Local Document 1")
    sources.append("Source: Local Document 2")
    return response, history, sources


def handle_file_upload(files):
    # This is a placeholder for file upload handling.
    uploaded_files = [file.name for file in files]
    return f"Uploaded files: {', '.join(uploaded_files)}"


def show_prompt_examples():
    # Show pre-made prompts for the user to try.
    return [
        "Example 1: Find all books about AI.",
        "Example 2: What is the meaning of life?",
        "Example 3: Search for books on quantum computing.",
    ]


with gr.Blocks() as demo:
    gr.Markdown("# 🧑‍💼 Local Librarian AI Chatbot")

    # File upload section
    with gr.Tab("File Upload"):
        gr.Markdown("## Upload Your Files to the System")
        file_upload = gr.File(label="Upload Documents", file_count="multiple", type="file")
        file_output = gr.Markdown(label="Uploaded Files")
        file_upload.upload(fn=handle_file_upload, inputs=file_upload, outputs=file_output)

    # Chat and search interface
    with gr.Tab("Search & Query"):
        gr.Markdown("## Ask the Librarian Anything")

        # Input fields for query and conversation history
        query_input = gr.Textbox(label="Search Query", placeholder="Enter your search query here...")
        search_button = gr.Button("Search")
        conversation_history = gr.Textbox(label="Conversation History", interactive=False, lines=10)
        source_display = gr.Textbox(label="Source Citations", interactive=False, lines=5)

        # Show example prompts
        prompt_examples = gr.Button("Show Prompt Examples")
        examples_output = gr.Markdown()

        search_button.click(fn=search_query,
                            inputs=[query_input, conversation_history, source_display, examples_output],
                            outputs=[conversation_history, source_display])
        prompt_examples.click(fn=show_prompt_examples, outputs=examples_output)

    # Instructions
    gr.Markdown("### How to use this app:")
    gr.Markdown("""
        1. In the 'File Upload' tab, upload your documents to make them available for search.
        2. In the 'Search & Query' tab, input your query in the 'Search Query' field.
        3. Press the 'Search' button to see the results and the source citations.
        4. You can review the conversation history as it evolves.
        5. Use the 'Show Prompt Examples' button to see some predefined prompts to guide you.
    """)

    with gr.Accordion("About this demo"):
        gr.Markdown(
            """
            This demo showcases several Gradio components for a local librarian AI chatbot:
            - File upload functionality to import documents.
            - Search functionality with conversation history and source citation.
            - Premade prompt examples to help users get started.
            - Simple Markdown and interactive components for a user-friendly experience.
            """
        )

demo.launch()
