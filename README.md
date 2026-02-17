# SalesOS-Agent 🤖

**Learn how AI agents use multiple tools to answer questions by building a working sales assistant.**

This project provides a baseline to explore a few core concepts of agentic AI: how an LLM decides which tools to use, how vector search finds relevant documents, and how everything connects together. Not prod ready, just built for learning.

---

## How It Works: Step-by-Step 

### 1. You Ask a Question
```
User: "Did we hit our Q1 2025 sales target?"
```
The agent receives the prompt as plain text input.

### 2. LLM Decides What Tools to Use
The agent sends the prompt and tool descriptions to the LLM. The LLM **reasons** about what information it needs: actual Q1 sales (database) and the Q1 target (documents). The LLM outputs which tools to call and with what parameters.

Agent (LangChain)          LLM (AI Model)
     |                          |
     |---(Question + Tools)---->|
     |                          | [Thinks: I need SQL tool first]
     |<---(Call SQL tool)-------|
     |                          |
[Executes SQL tool]             |
     |                          |
     |---(SQL results)--------->|
     |                          | [Thinks: Now I need RAG tool]
     |<---(Call RAG tool)-------|
     |                          |
[Executes RAG tool]             |
     |                          |
     |---(RAG results)--------->|
     |                          | [Thinks: Now I can answer]
     |<---(Final answer)--------|

### 3. Agent Calls the SQL Tool
```python
@tool
def query_sales_database(question: str) -> str:
    # Agent passes: "What were Q1 sales?"
```
The agent decides to use the SQL tool first. The tool receives the question and generates a SQL query to get the actual sales number. The tool makes a second LLM call to generate the SQL.

### 4. SQL Generation 
```python
sql_query = _generate_sql_with_llm(question, schema)
# Generates for example: SELECT SUM(total_amount) FROM sales WHERE ...
```
The SQL tool asks the LLM to write the query based on the database schema. The schema is cached after the first call so it doesn't query the DB structure every time which works fine for a small local setup like this one.

### 5. Query Execution
```python
cursor.execute(sql_query)
results = cursor.fetchall()
# Returns: [{'total': 271680.50}]
```
The generated SQL runs against the SQLite database. 

### 6. Agent Calls the RAG Tool
```python
@tool
def search_local_docs(query: str) -> str:
    # Agent passes: "Q1 sales target"
```
The agent realizes it also needs the sales target, so it calls the document search tool. This tool uses RAG (Retrieval Augmented Generation) to find relevant info.

### 7. Vector Search (Semantic Similarity)
```python
docs = vectorstore.similarity_search_with_score(query, k=6)
```
Documents in the "kb/" folder were converted to **vector embeddings** (arrays of numbers that represent meaning) when you ran `setup_knowledge_base.py`. The query is also converted to a vector, then ChromaDB finds the "TOP_K" most similar document chunks using **vector distance** In other words it measures how close vectors are in mathematical space, which corresponds to semantic similarity. The closer the distance between the points, the more they are similar.

### 8. Retrieval Augmented Generation (RAG) Returns Relevant Chunks
```
[Document 1] From: Q1_2025_Sales_Priorities.md (Score: 0.22)
Q1 2025 Company Goals

Revenue Targets

New Business Goal:** $12M  
```
The tool returns the most relevant document sections with their similarity scores. Lower scores mean more relevant (closer in vector space).

### 9. Agent Synthesizes the Answer
```python
# Agent now has:
# - Actual sales: $271,680
# - Target: $15M
# Agent reasons about both pieces
```
The LLM combines both tool outputs to formulate a complete answer. It can do math, make comparisons, and provide context.

### 10. Response to User
```
A: Q1 sales were $271K vs target of $15M (1.8% of goal).
   You fell short by $14.73M.
```
The agent returns a synthesized answer that addresses your original question. With the `viz_tool.py` the agent may create a graph from SQL data in your charts/ folder. 

---

## Core Components Explained

### Agent 
The LLM that orchestrates everything. It reads tool descriptions, decides which tools to call, interprets results, and generates responses. Uses **conversation memory** to remember context within a session.

### Tools (Python Functions)
Python functions decorated with `@tool` that the agent can call. Each tool has a name, description, and parameters that the agent reads to decide when to use it. Tools return strings that the agent can reason about.

### SQL Tool (Text-to-SQL)
Takes a natural language question, generates SQL using an LLM, validates it (blocks DROP, DELETE, etc.), executes it on SQLite, and returns results. 

### RAG Tool (Vector Search)
Searches your company documents using semantic similarity. Documents are split into chunks (500 chars), converted to vectors (arrays of 384 numbers from `sentence-transformers/all-MiniLM-L6-v2`), and stored in ChromaDB. When you search, your query becomes a vector and ChromaDB finds the closest matching document vectors.

### Vector Database (ChromaDB)
Stores document embeddings and enables fast similarity search. Uses **HNSW algorithm** (Hierarchical Navigable Small World) to find nearest neighbors in high-dimensional space without comparing every single vector.

### Embeddings (Sentence Transformers)
Converts text to vectors. Similar meanings produce similar vectors. Example: "sales target" and "revenue goal" have close vectors even though they share no words. The model (`all-MiniLM-L6-v2`) was trained on millions of sentences to learn these semantic relationships.

### Visualization Tool
Non-LLM tool that takes data (typically SQL results), converts to charts using matplotlib, and saves PNG files. Shows that tools don't have to call LLMs—they can be simple functions.

### System Prompt
Instructions that tell the agent how to use tools. Explains which tools have what data, gives examples of multi-tool workflows, and sets behavior guidelines. The agent follows these instructions but isn't perfect—more capable LLMs follow prompts better.

### Checkpointer (Memory)
Stores conversation history in RAM using `InMemorySaver`. Allows follow-up questions like "What about just Europe?" after asking about customers. Memory resets when you restart the program.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install langchain langchain-openai langchain-community langgraph
pip install chromadb sentence-transformers matplotlib pandas
# etc...
```

### 2. Choose Your LLM

**Local (Ollama - easiest):**
```bash
ollama pull llama3.2:3b
ollama serve
```

**Or use cloud API:**
```bash
export OPENAI_API_KEY="sk-..."
```

Edit `config.py` with your endpoint and model name.

### 3. Set Up Data
```bash
# Create fake sales database (200 customers, 500 transactions)
python setup_sales_db.py

# Add documents to kb/ folder
mkdir -p kb
echo "Q1 Sales Target: $15M" > kb/targets.txt

# Convert documents to vectors
python setup_knowledge_base.py
```

### 4. Run Agent
```bash
python agent.py
```

---

## What You'll Learn

**Agentic AI:** How LLMs decide which actions to take rather than just generating text.

**Tool Calling:** How to make Python functions available to an agent and how the agent decides when to use them.

**RAG (Vector Search):** How semantic search works—why "sales target" finds documents about "revenue goals" even without exact keyword matches.

**Text-to-SQL:** How LLMs can write database queries from natural language questions.

**Multi-Tool Reasoning:** How agents combine information from multiple sources (database + documents) to answer complex questions.

**LangChain Patterns:** How the ReAct agent pattern works, how tools are defined with `@tool`, and how conversation memory is managed.

---

## Example Queries to Try

```bash
# Single tool (SQL only)
"What were total sales last quarter?"
"Who are our top 5 customers?"

# Single tool (RAG only)
"What is our discount policy?"
"What are the Q1 sales targets?"

# Multi-tool (SQL + RAG)
"Did we hit our Q1 target?"
"Should we offer discounts to our top customers?"

# Visualization
"Show me a bar chart of top customers by revenue"
"Create a line chart of monthly sales trends"

# Follow-up questions
"What about just Europe?"  # After asking about customers
"Show that as a pie chart"  # After getting regional data
```

---

## Project Structure

```
SalesOS-Agent/
├── agent.py                 # agent setup and main loop
├── config.py                # LLM endpoint, paths, settings
│
├── setup_sales_db.py        # Creates SQLite with fake data
├── setup_knowledge_base.py  # Converts docs → vectors
│
├── tools/
│   ├── sales_tool.py        # SQL query tool (nested LLM)
│   ├── knowledge_tool.py    # RAG search with vector similarity
│   ├── viz_tool.py          # Chart creation (matplotlib)
│   └── web_tools.py         # Wikipedia lookup
│
├── kb/                      # Put your documents here
├── sales_db/                # Generated SQLite database
├── chroma_db/               # Generated vector database
└── charts/                  # Generated PNG charts
```

**Start reading here:**
1. `agent.py` (system prompt - how agent uses tools)
2. `tools/knowledge_tool.py` (simplest tool - RAG search)
3. `tools/sales_tool.py` (LLM SQL calls)

---

## Key Design Choices (Why It's Built This Way)

**Why single agent, not multi-agent?**  
Simpler to learn. Multi-agent adds complexity that obscures the core concepts.

**Why SQLite?**  
No server setup needed. Easy to inspect with `sqlite3 sales_db/sales_data.db`.

**Why all-MiniLM-L6-v2 embeddings?**  
Fast, small, good enough for learning. 

**Why ChromaDB?**  
Simple Python library, no separate server needed. Good for learning, can scale if needed.

---

## Common Issues

**"No data in database"** → Run `python setup_sales_db.py` first

**"Connection refused"** → Make sure your LLM server is running (`ollama serve` or `./llama.cpp server`)

**"No documents found"** → Put files in `kb/` folder and run `python setup_knowledge_base.py`

**Agent gives wrong answers** → Try a different model (8B vs 3B parameters) or simplify your question

---

## Next Steps

Maybe you can:
- Add your own tools (email sender, API caller, etc.)
- Experiment with different embedding models
- Try LangGraph for multi-agent workflows
- Implement proper error handling and logging
- Build a web UI with Streamlit or FastAPI

---

## License

MIT - Use this to learn and build your own projects.

**Built for learning agentic AI concepts, not for production use.** 
