import os
import shutil
import gradio as gr

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    SummaryIndex
)
from CustomCitationEngine import CitationQueryEngine

from llama_index.core.settings import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.schema import IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM Config
from llama_index.llms.openai import OpenAI

from llama_index.llms.ollama import Ollama

llm = Ollama(model="qwen2.5:7b", request_timeout=60.0)


# Global variables
# llm = OpenAI(
#     model="gpt-4o-mini",
#     api_key=OPENAI_API_KEY,
# )

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
Settings.llm = llm
Settings.chunk_size = 512

# Global list to store nodes
objects = []
saved_files = {}

# Global variable to store query engine
global_query_engine = None


def search_query(query, history):
    """Handle user query with the query engine"""
    global global_query_engine

    prompt = (f"\n\n---Chat history---: {str(history)}\n ---User's New Input---: {query}"
              f"\nANSWER MUST INCLUDE THE FULL PASSAGE OF CITATIONS!!!!!!!!!!!!"
              f"\nANSWER MUST INCLUDE THE FULL PASSAGE OF CITATIONS!!!!!!!!!!!!"
              f"Please include citation in this format:"
              f"---CITATION---:"
              f"[1] **[BOOK TITLE]**, [PAGE NUMBER]")

    # Check if query engine is available
    if global_query_engine is None:
        return {
            'role': "assistant",
            'content': "Upload documents first! Ready to assist you. 📚"
        }

    try:
        # Use the query engine to process the query
        response = global_query_engine.query(prompt)
        print(response.source_nodes)

        return {
            'role': "assistant",
            'content': response.response
        }
    except Exception as e:
        return {
            'role': "assistant",
            'content': f"An error occurred: {str(e)}"
        }


def handle_file_upload(files):
    global objects
    global global_query_engine
    global saved_files

    # Create the upload directory if it doesn't exist
    upload_dir = "upload"
    os.makedirs(upload_dir, exist_ok=True)

    # List to store paths of saved file
    file_paths = []

    # Process each uploaded file
    for file in files:
        # Get the original filename
        original_filename = os.path.basename(file.name)

        # Create the destination path
        destination_path = os.path.join(upload_dir, original_filename)

        # Copy the file to the upload directory
        shutil.copy(file.name, destination_path)

        # Add the destination path to saved files list
        file_paths.append(destination_path)

        docs = SimpleDirectoryReader(input_files=[destination_path]).load_data()
        vector_index = VectorStoreIndex.from_documents(docs)
        summary_index = SummaryIndex.from_documents(docs)

        # define query engines
        vector_query_engine = CitationQueryEngine.from_args(vector_index, similarity_top_k=3, citation_chunk_size=1024,)
        summary_query_engine = summary_index.as_query_engine()
        # summary_query_engine = CitationQueryEngine.from_args(summary_index, similarity_top_k=3, citation_chunk_size=1024)

        # define tools
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=(
                        f"Use this index if you need to lookup specific facts related '{original_filename}'"
                        f"MUST INCLUDE ALL CITATIONS IN THE ANSWER!!!!!!!!! THIS IS VERY IMPORTANT!!!!"
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=summary_query_engine,
                metadata=ToolMetadata(
                    name="summary_tool",
                    description=(
                        f"Useful for summarization questions related to {file}"
                    ),
                ),
            ),
        ]
        agent = ReActAgent.from_tools(
            query_engine_tools,
            llm=llm,
            verbose=True,
        )
        book_summary = (f"This content contains the full book '{original_filename}'."
                        f"Use this index if you need to lookup specific facts related '{original_filename}'")
        node = IndexNode(text=book_summary, index_id=file, obj=agent)
        objects.append(node)

        vector_index = VectorStoreIndex(
            objects=objects,
        )

        global_query_engine = vector_index.as_query_engine(similarity_top_k=1, verbose=True)

        saved_files[original_filename] = destination_path

    # Return a message with the saved file paths
    return f"Uploaded: {', '.join(saved_files.keys())}"


def list_uploaded_files():
    """List all files in the upload directory"""
    upload_dir = "upload"

    # Check if upload directory exists
    if not os.path.exists(upload_dir):
        return "No upload directory found."

    # Get list of files
    files = os.listdir(upload_dir)

    if not files:
        return "No files uploaded yet."

    return ", ".join(files)

def check_query_engine_status():
    """Check the status of the global query engine"""
    global objects

    available_engine = ""
    for obj in objects:
        available_engine += f"{obj}\n"

    if global_query_engine is None:
        return "Query Engine: Not Ready 🚫"
    else:
        return f"Query Engine: Ready to Query! 🚀\n{available_engine}"


def delete_file(filename):
    """Delete a specific file from the upload directory"""
    global saved_files
    global objects
    global global_query_engine

    upload_dir = "upload"
    file_path = os.path.join(upload_dir, filename)

    try:
        # Remove the file from the filesystem
        if os.path.exists(file_path):
            os.remove(file_path)

        # Remove from saved_files dictionary
        if filename in saved_files:
            del saved_files[filename]

        # Reset query engine and objects if no files remain
        if not saved_files:
            objects = []
            global_query_engine = None
        else:
            # Recreate the index without the deleted file
            new_objects = [obj for obj in objects if obj.index_id != filename]
            objects = new_objects

            # Recreate the vector index
            vector_index = VectorStoreIndex(objects=objects)
            global_query_engine = vector_index.as_query_engine(similarity_top_k=1, verbose=True)

        return f"Deleted: {filename}"
    except Exception as e:
        return f"Error: {str(e)}"


def clear_upload_folder():
    """Clear all files from the upload folder and reset global variables"""
    global saved_files
    global objects
    global global_query_engine

    upload_dir = "upload"

    try:
        # Remove all files in the upload directory
        for filename in os.listdir(upload_dir):
            file_path = os.path.join(upload_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Reset global variables
        saved_files = {}
        objects = []
        global_query_engine = None

        return "Cleared all files 🗑️"
    except Exception as e:
        return f"Error clearing files: {str(e)}"


# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🧑‍💼 Local Librarian AI")

    with gr.Tab("Document Management"):
        gr.Markdown("## Manage Your Documents")

        file_upload = gr.File(label="Upload Documents", file_count="multiple", type="filepath")
        file_output = gr.Markdown(label="Uploaded Files")
        file_upload.upload(fn=handle_file_upload, inputs=file_upload, outputs=file_output)



    # Chat Interface
    with gr.Tab("AI Chat"):
        chat_interface = gr.ChatInterface(
            fn=search_query,
            title="Ask About Your Documents",
            theme="default"
        )
    gr.Markdown('## Citations')

    gr.Markdown("## Manage Your Documents")
    with gr.Row():
        # File List Button
        list_files_btn = gr.Button("List Files", variant="primary")
        # Query Engine Status Button
        check_engine_btn = gr.Button("Check Engine", variant="secondary")
        # Clear Files Button
        clear_files_btn = gr.Button("Clear All", variant="stop")

    files_status = gr.Markdown(label="Files Status")
    engine_status = gr.Markdown(label="Engine Status")

    list_files_btn.click(fn=list_uploaded_files, outputs=files_status)
    check_engine_btn.click(fn=check_query_engine_status, outputs=engine_status)
    clear_files_btn.click(fn=clear_upload_folder, outputs=files_status)

    with gr.Row():
        # Delete File
        file_to_delete = gr.Textbox(label="File to Delete", scale=3)
        delete_btn = gr.Button("Delete", variant="stop", scale=1)

    delete_output = gr.Markdown(label="Delete Status")
    delete_btn.click(
        fn=delete_file,
        inputs=file_to_delete,
        outputs=delete_output
    )
    # About Section
    with gr.Accordion("How to use 💡"):
        gr.Markdown("""
        ## Upload:
        - Select one or multiple BOOKS (.epub/.pdf) to upload.
        - Wait a bit ⏳ for the documents to be processed.
        ## Status Check:
        - Click the 'List Files' button to see the uploaded BOOKS.
        - Click the 'Check Engine' button to see the query engine status.
        - Click the 'Clear All' button to remove all uploaded BOOKS and ENGINES.
        ## Chat:
        - Type your question in the chat interface.
        - Citations are included for specific questions ONLY. Not included for summary, since it will be too much for the LLM to handle.
        ## Citations:
        - Includes the book name and page number.
        """)

# Launch the Gradio app
demo.launch()