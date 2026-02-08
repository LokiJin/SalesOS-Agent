# """
# Agent Implementation with Multiple Tools for llama.cpp

# Installation:
# pip install langgraph langchain-openai langchain-community chromadb sentence-transformers requests

# Setup:
# 1. Start llama.cpp server with a tool-enabled model, gpt-oss-20b-Q4_K_M.gguf is a confirmed example that works
   
# 2. Run this script
# """

import os
import requests
from typing import List, Optional
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# RAG setup 

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ============================================================================
# Configuration
# ============================================================================

os.environ['OPENAI_API_KEY'] = "not_a_real_key"  # Some models required a variable here, but is not actually used

RAG_AVAILABLE = True
LLAMA_SERVER_URL = "http://localhost:8080/v1/"
MODEL_NAME = "gpt-oss-20b-Q4_K_M.gguf"  # Change to your model name
CHROMA_DB_PATH = "chroma_db"  # Path for RAG database
DOCS_PATH = "kb/"  # Path to your documents for RAG


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
        
      
        vectorstore = init_vectorstore()

        docs = vectorstore.similarity_search(query, k=3, distance_metric="cosine")
        # Search for relevant documents
      
        
        if not docs:
            return "No relevant documents found in local knowledge base."
        
        # Format results
        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            results.append(f"[Document {i}] {source}\n{doc.page_content}")
        
        return "\n\n---\n\n".join(results)
    
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
            print("DUCK DUCK DUCK GO", query, response)
           
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

def setup_rag_database(documents_path: str, db_path: str):
    """
    One-time setup to create RAG database from your documents.
    
    Args:
        documents_path: Path to folder containing your text documents
        db_path: Path where the Chroma database will be saved
    """
    if not RAG_AVAILABLE:
        print("RAG dependencies not installed. Skipping RAG setup.")
        return False
    
    if not os.path.exists(documents_path):
        print(f"Documents folder not found: {documents_path}")
        print("Creating example documents folder...")
        os.makedirs(documents_path, exist_ok=True)
        
    
    print(f"Loading documents from {documents_path}...")
    loader = DirectoryLoader(
        documents_path,
        glob="**/*.txt",
        loader_cls=lambda path: TextLoader(path, encoding="utf-8")
    )
    documents = loader.load()
    
    if not documents:
        print("No documents found. Add .txt files to the documents folder.")
        return False
    
    print(f"Loaded {len(documents)} documents. Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(documents)
    
    print(f"Creating embeddings and storing in {db_path}...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=db_path, 
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    print(f"✓ RAG database created successfully with {len(splits)} chunks!")
    return True


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
    

    # Create agent
    agent = create_agent(
        llm,
        tools_to_use,
        system_prompt="""You are a highly capable AI assistant with access to multiple tools. Always use the tools instead of guessing or making up information.

Available tools and when to use them:

1. search_local_docs:
   - Use **first** for any question.
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
- Do not guess or fabricate answers."""
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
        result = agent.invoke({"messages": [HumanMessage(content=question)]})
        
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
    
    print("🤖 Multi-Tool Agent for llama.cpp")
    print("="*60)
    
    # Optional: Setup RAG database if it doesn't exist
    if RAG_AVAILABLE and not os.path.exists(CHROMA_DB_PATH):
        print("\n📚 RAG database not found. Setting up...")
        setup_rag_database(DOCS_PATH, CHROMA_DB_PATH)
        print()
    
    # Create agent
    print("🔧 Initializing agent...")
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