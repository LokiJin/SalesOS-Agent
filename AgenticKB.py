# """
# Agent Implementation with Multiple Tools for llama.cpp

# Installation:
# pip install langgraph langchain-openai langchain-community chromadb sentence-transformers requests

# Setup:
# 1. Start llama.cpp server with a tool-enabled model, gpt-oss-20b-Q4_K_M.gguf is a confirmed example that works
   
# 2. Run this script
# """

from pathlib import Path
import os
import requests
from typing import List, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

# RAG setup 
from rag_metadata import RAGMetadataManager
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredHTMLLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================================
# Configuration
# ============================================================================
SUPPORTED_FILE_TYPES = {
    '.txt': TextLoader,
    '.pdf': PyPDFLoader,
    '.docx': Docx2txtLoader,
    '.md': UnstructuredMarkdownLoader,
    '.csv': CSVLoader,
    '.json': JSONLoader,
    '.html': UnstructuredHTMLLoader,
    '.htm': UnstructuredHTMLLoader,
}

os.environ['OPENAI_API_KEY'] = "not_a_real_key"  # Some models required a variable here, but is not actually used

RAG_AVAILABLE = True
LLAMA_SERVER_URL = "http://localhost:8080/v1/"
MODEL_NAME = "gpt-oss-20b-Q4_K_M.gguf"  # Change to your model name
CHROMA_DB_PATH = "Agentic KB/chroma_db"  # Path for RAG database
DOCS_PATH = "Agentic KB/kb/"  # Path to your documents for RAG


VECTORSTORE = None

def init_vectorstore():
    global VECTORSTORE
    if VECTORSTORE is None:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        VECTORSTORE = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
    return VECTORSTORE


# ============================================================================
# Tool Definitions
# ============================================================================

@tool
def wiki_summary(query: str) -> str:
    """
    Fetches a Wikipedia summary for general knowledge questions.
    
    Use for: historical facts, biographies, scientific concepts, general knowledge.
    DO NOT use for: company-specific info, current events, calculations.
    
    Args:
        query: The topic to search on Wikipedia (e.g., "Nikola Tesla", "Quantum Computing")
    """
    try:
        # Replace spaces with underscores for Wikipedia API
        query_formatted = query.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query_formatted}"
        
        headers = {'User-Agent': 'Agent/1.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return f"Sorry, no Wikipedia page found for '{query}'."
        
        data = response.json()
        extract = data.get("extract", "No summary available.")
        
        # Add URL for reference
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        if page_url:
            extract += f"\n\nSource: {page_url}"
            
        return extract
    except Exception as e:
        return f"Error fetching Wikipedia data: {str(e)}"



@tool
def search_local_docs(query: str) -> str:
    """
    Search your local RAG database for relevant information based on the query. 
    Always use this tool FIRST for any question, as it contains your specific knowledge base.
    Args:
        query: The search query for your local knowledge base
    """
    
    try:
        if not os.path.exists(CHROMA_DB_PATH):
            return f"Local knowledge base not found at {CHROMA_DB_PATH}. Please set up your RAG database first."
        
        #print(f"[DEBUG] Searching for: '{query}'")
        vectorstore = init_vectorstore()
        
        # DEBUG: Check collection size
        collection = vectorstore._collection
        
        # Search for relevant documents
        docs = vectorstore.similarity_search(query, k=3)
  
        
        # # DEBUG: Show what was found
        # print(f"[DEBUG] Found {len(docs)} documents for query: '{query}'")
        # for i, doc in enumerate(docs):
        #     print(f"[DEBUG] Doc {i+1} source: {doc.metadata.get('source', 'Unknown')}")
        #     print(f"[DEBUG] Doc {i+1} preview: {doc.page_content[:150]}...")
        
        if not docs:
            return "No relevant documents found in local knowledge base."
        
        # Format results
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content
            results.append(f"[Document {i}] From {source}:\n{content}")
        
        final_result = "\n\n---\n\n".join(results)
        #print(f"[DEBUG] Returning {len(final_result)} characters of results")
        return final_result
    
    except Exception as e:
      return f"Error searching local documents: {str(e)}"


@tool
def web_search_duckduckgo(query: str) -> str:
    """ Search the web using DuckDuckGo for current information and recent events. 
    Use for: current events, recent news, up-to-date information not in Wikipedia. 
    Args: query: Search query for DuckDuckGo """
    try:
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': 1
        }
        headers = {'User-Agent': 'Agent/1.0'}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            return "Error: Unable to search the web at this time."
        
        data = response.json()
        abstract = data.get('AbstractText')
        if abstract:
            source = data.get('AbstractURL', '')
            return f"{abstract}\n\nSource: {source}" if source else abstract
        
        # Handle RelatedTopics recursively
        related = data.get('RelatedTopics', [])
        results = []

        def extract_text(topics):
            for item in topics:
                if 'Text' in item:
                    results.append(item['Text'])
                elif 'Topics' in item:
                    extract_text(item['Topics'])

        extract_text(related)

        if results:
            print(results[:3])
            return "Related information:\n" + "\n\n".join(results[:3])
        

        return "No results found. Try a different search query."

    except Exception as e:
        return f"Error performing web search: {str(e)}"


# ============================================================================
# RAG Setup Functions (Optional)
# ============================================================================

def get_loader_for_file(file_path: str):
    """
    Get appropriate loader for a file based on extension.
    
    Args:
        file_path: Path to file
        
    Returns:
        Loader class or None if unsupported
    """
    ext = Path(file_path).suffix.lower()
    return SUPPORTED_FILE_TYPES.get(ext)

def setup_rag_database_incremental(
    documents_path: str, 
    db_path: str,
    metadata_db_path: str = None,
    force_rebuild: bool = False
):
    """
    Setup or update RAG database with incremental change detection.
    
    Args:
        documents_path: Path to folder containing your text documents
        db_path: Path where the Chroma database will be saved
        metadata_db_path: Path to metadata SQLite database (defaults to db_path + '/metadata.db')
        force_rebuild: If True, delete everything and rebuild from scratch
        
    Returns:
        Dictionary with update statistics
    """
    if not RAG_AVAILABLE:
        print("RAG dependencies not installed. Skipping RAG setup.")
        return None
    
    # Setup paths
    if metadata_db_path is None:
        metadata_db_path = os.path.join(db_path, "metadata.db")
    
    os.makedirs(db_path, exist_ok=True)
    
    # Initialize metadata manager
    metadata_mgr = RAGMetadataManager(metadata_db_path)
    
    # Force rebuild if requested
    if force_rebuild:
        print("🔄 Force rebuild requested. Deleting existing database...")
        import shutil
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
        os.makedirs(db_path, exist_ok=True)
        metadata_mgr = RAGMetadataManager(metadata_db_path)  # Reinit
    
    # Check if documents folder exists
    if not os.path.exists(documents_path):
        print(f"Documents folder not found: {documents_path}")
        print("Creating example documents folder...")
        os.makedirs(documents_path, exist_ok=True)
        return {"status": "no_documents"}
    
   # Scan current files (ALL supported types)
    print(f"📂 Scanning documents in {documents_path}...")
    current_files = {}
    supported_extensions = list(SUPPORTED_FILE_TYPES.keys())
    
    # for file_path in Path(documents_path).rglob("*.txt"):
    #     file_path_str = str(file_path)
    #     file_hash = metadata_mgr.compute_file_hash(file_path_str)
    #     if file_hash:
    #         current_files[file_path_str] = file_hash
    for ext in supported_extensions:
        for file_path in Path(documents_path).rglob(f"*{ext}"):
            file_path_str = str(file_path)
            file_hash = metadata_mgr.compute_file_hash(file_path_str)
            if file_hash:
                current_files[file_path_str] = file_hash
    
    if not current_files:
        print(f"No supported files found. Supported types: {', '.join(supported_extensions)}")
        return {"status": "no_documents"}
    
    print(f"Found {len(current_files)} files")
    
    # Get tracked files from metadata
    tracked_files = metadata_mgr.get_all_tracked_files()
    
    # Determine what needs to be updated
    new_files = set(current_files.keys()) - tracked_files
    deleted_files = tracked_files - set(current_files.keys())
    potentially_changed = tracked_files & set(current_files.keys())
    
    changed_files = set()
    for file_path in potentially_changed:
        current_hash = current_files[file_path]
        stored_hash = metadata_mgr.get_stored_hash(file_path)
        if current_hash != stored_hash:
            changed_files.add(file_path)
    
    # Summary
    stats = {
        "total_files": len(current_files),
        "new_files": len(new_files),
        "changed_files": len(changed_files),
        "deleted_files": len(deleted_files),
        "unchanged_files": len(current_files) - len(new_files) - len(changed_files)
    }
    
    print(f"\n📊 Change Detection:")
    print(f"   New files: {stats['new_files']}")
    print(f"   Changed files: {stats['changed_files']}")
    print(f"   Deleted files: {stats['deleted_files']}")
    print(f"   Unchanged files: {stats['unchanged_files']}")
    
    # If nothing to update, return early
    if not new_files and not changed_files and not deleted_files:
        print("\n✓ Database is up to date. No changes needed.")
        return stats
    
    # Initialize embeddings and vectorstore
    print(f"\n🔧 Loading vectorstore...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Handle deletions
    if deleted_files:
        print(f"\n🗑️  Deleting {len(deleted_files)} removed files...")
        for file_path in deleted_files:
            chunk_ids = metadata_mgr.delete_file_metadata(file_path)
            if chunk_ids:
                try:
                    vectorstore.delete(ids=chunk_ids)
                    print(f"   ✓ Deleted {len(chunk_ids)} chunks from {Path(file_path).name}")
                except Exception as e:
                    print(f"   ✗ Error deleting chunks for {file_path}: {e}")
    
    # Handle updates (changed files)
    files_to_process = list(new_files | changed_files)
    
    if files_to_process:
        print(f"\n📥 Processing {len(files_to_process)} files...")
        
        # Delete old chunks for changed files
        for file_path in changed_files:
            chunk_ids = metadata_mgr.get_chunk_ids(file_path)
            if chunk_ids:
                try:
                    vectorstore.delete(ids=chunk_ids)
                    print(f"   🔄 Updating {Path(file_path).name} (removed {len(chunk_ids)} old chunks)")
                except Exception as e:
                    print(f"   ✗ Error deleting old chunks for {file_path}: {e}")
        
        # Load and split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )
        
        total_chunks_added = 0
        
        for file_path in files_to_process:
            try:
                # Get appropriate loader for this file type
                loader_class = get_loader_for_file(file_path)
                if not loader_class:
                    print(f"   ⚠️  Skipping {Path(file_path).name} (unsupported type)")
                    continue
                
                # Load single file with appropriate loader
                loader = loader_class(file_path)
                docs = loader.load()
                
                # Split into chunks
                splits = text_splitter.split_documents(docs)
                
                if not splits:
                    continue
                
                # Generate unique chunk IDs
                chunk_ids = [f"{file_path}_{i}" for i in range(len(splits))]
                
                # Add to vectorstore
                vectorstore.add_documents(documents=splits, ids=chunk_ids)
                
                # Get file type
                file_type = Path(file_path).suffix.lower().replace('.', '')
                
                # Update metadata
                metadata_mgr.store_file_metadata(
                    file_path,
                    current_files[file_path],
                    chunk_ids,
                    file_type  # NEW: track file type
                )
                
                total_chunks_added += len(splits)
                status = "✓" if file_path in new_files else "🔄"
                print(f"   {status} {Path(file_path).name} ({file_type}): {len(splits)} chunks")
                
            except Exception as e:
                print(f"   ✗ Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
    
    # Final stats
    final_stats = metadata_mgr.get_stats()
    stats.update(final_stats)
    
    print(f"\n✓ Database updated successfully!")
    print(f"   Total files tracked: {final_stats['file_count']}")
    print(f"   Total chunks: {final_stats['total_chunks']}")
    
    return stats

# ============================================================================
# Agent Setup
# ============================================================================

def create_agent_a(
    model_name: str = MODEL_NAME,
    base_url: str = LLAMA_SERVER_URL,
    temperature: float = 0.1,
    tools_to_use: Optional[List] = None
):
    """
    Create the agent with specified tools.
    
    Args:
        model_name: Name of the model in llama.cpp server
        base_url: URL of the llama.cpp server
        temperature: Sampling temperature (0.0 - 1.0)
        tools_to_use: List of tools to give the agent (defaults to all available tools)
    """
    # Setup LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        streaming=False,
        max_tokens=2500
    )
    
    # Default tools
    if tools_to_use is None:
        tools_to_use = [
            search_local_docs,
            wiki_summary,
            web_search_duckduckgo
        ]
    
    #Add an in‑memory checkpointer 
    checkpointer = InMemorySaver()

    # Create agent
    agent = create_agent(
        llm,
        tools_to_use,
        system_prompt="""You are a highly capable AI assistant with access to multiple tools. Always use the tools instead of guessing or making up information.

        You are a highly capable AI assistant with access to multiple tools.

**CRITICAL TOOL USAGE RULES:**
1. ONLY call search_local_docs with keywords FROM THE USER'S CURRENT QUESTION
2. DO NOT search for topics from previous conversations
3. DO NOT search for random topics not mentioned by the user

Available tools and when to use them:

1. search_local_docs:
   - Use **first** for most questions.
   - Contains the most authoritative, curated, context-specific knowledge.
   - Use whenever the question could be answered by your local knowledge base.

2. wiki_summary:
   - Use for general knowledge, historical facts, biographies, concepts, and definitions.
   - Use only if search_local_docs does not provide relevant information.

3. web_search_duckduckgo:
   - Use **last**, only if neither local docs nor wiki can answer.
   - Only use for factual reference-style information about current events or entities.
   - Do **not** use for live news, stock prices, sports scores, or unknown upcoming events.
   - Transform the user's question into a concise reference-style query suitable for DuckDuckGo.
     - Extract main entity or topic
     - Append keywords like "summary", "definition", "overview"
   - If multiple interpretations exist, include the 2-3 most relevant keywords.

Guidelines:
- Always check tools in the following priority: search_local_docs → wiki_summary → web_search_duckduckgo.
- Synthesize information from tools into a clear, concise, and factual answer.
- If a tool returns no results or an error, try the next tool in the hierarchy.
- Do not guess or fabricate answers.""", 
        checkpointer=checkpointer,
    )
    
    return agent


# ============================================================================
# Helper Functions
# ============================================================================

def ask(agent, question: str, verbose: bool = True) -> str:
    """
    Ask the agent a question and return the answer.
    
    Args:
        agent: The agent
        question: The question to ask
        verbose: Whether to print intermediate steps
    
    Returns:
        The agent's final answer
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"{'='*60}\n")
    
    try:
        
        thread_config = {"configurable": {"thread_id": "my_thread"}} #TODO: make this more robust for user

        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            thread_config
            )
        


        # Print tool calls if verbose
        if verbose:
            messages = result["messages"]
            for msg in messages[1:-1]:  # Skip initial question and final answer
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        print(f"🔧 Tool used: {tool_call['name']}")
                        print(f"   Args: {tool_call['args']}")
                        print()
        
        answer = result["messages"][-1].content
    
        if verbose:
            print(f"{'='*60}")
            print(f"Answer: {answer}")
            print(f"{'='*60}\n")
        
        return answer
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if verbose:
            print(f"❌ {error_msg}")
        return error_msg


# ============================================================================
# Main
# ============================================================================

def main():
    """Main function with example usage"""

    # Setup/Update RAG database 
    if RAG_AVAILABLE:
        print("\n📚 Checking RAG database...")
        setup_rag_database_incremental(DOCS_PATH, CHROMA_DB_PATH)  
        print()
    
    # Create agent
    agent = create_agent_a()
    print("✓ Agent ready!\n")
    
    # Interactive mode
    print("\n" + "="*60)
    print("Interactive Mode - Type 'quit' to exit")
    print("="*60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            ask(agent, user_input)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()