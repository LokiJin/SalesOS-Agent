"""
An AI assistant with sales data and knowledge base capabilities 
"""

from typing import Optional, List, Generator
import uuid
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# Import configuration
from config import (
    MODEL_NAME, LLAMA_SERVER_URL, DEFAULT_TEMPERATURE, 
    DEFAULT_MAX_TOKENS, DEBUG_MODE, BANNER, validate_config
)

# Import tools
from tools import (
    query_sales_database, 
    search_local_docs, 
    wiki_summary,
    create_chart,
    create_multi_series_chart
)


def create_sales_agent(
    model_name: str = MODEL_NAME,
    base_url: str = LLAMA_SERVER_URL,
    temperature: float = DEFAULT_TEMPERATURE,
    tools: Optional[List] = None
):
    """
    Create the main agent with all capabilities.
    
    Args:
        model_name: Name of the LLM model
        base_url: URL of the llama.cpp server
        temperature: Sampling temperature (0.0-1.0)
        tools: List of tools (defaults to all available)
    
    Returns:
        Configured agent
    """
    
    # Validate configuration before initializing
    if not validate_config():
        raise RuntimeError("Configuration validation failed. See errors above.")
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        max_tokens=DEFAULT_MAX_TOKENS
    )
    
    # Set up tools
    if tools is None:
        tools = [
            search_local_docs,      # Check internal docs first
            query_sales_database,   # Sales intelligence
            wiki_summary,           # External knowledge
            create_chart,           # Create interactive visualizations
            create_multi_series_chart, # Multi-series charts
        ]
    
    # Conversation memory in RAM
    checkpointer = InMemorySaver()
    
    # System prompt - defines agent behavior
    system_prompt = """You are an intelligent AI assistant with access to multiple information sources.

Your capabilities:
1. search_local_docs - Company docs, policies, procedures, GOALS, TARGETS, STRATEGIES
2. query_sales_database - ONLY actual sales data: transactions, revenue, customers, products
3. wiki_summary - General knowledge and encyclopedic information
4. create_chart - Create interactive visualizations (bar, line, pie, scatter, histogram, area) 
5. create_multi_series_chart - Create interactive multi-series charts for comparing metrics (e.g., actual vs target, sales vs expenses)

Visualization Workflow:
When asked to create a chart or visualize data:
1. First, get the data using query_sales_database
2. Format the results as JSON
3. Call create_chart with the data
4. The tool will save the chart and return the file path

Decision-Making Guidelines:
- For questions about GOALS, TARGETS, STRATEGIES → Use search_local_docs
- For questions about actual SALES, REVENUE, CUSTOMERS → Use query_sales_database
- For general knowledge → Use wiki_summary
- If you need BOTH actual data and goals, call BOTH tools
- Always explain your reasoning

When answering:
- Synthesize information clearly
- Cite sources when relevant
- If uncertain, say so
- For business questions, provide actionable insights when possible
"""
    
    # Create agent
    agent = create_agent(
        llm,
        tools,
        system_prompt=system_prompt,
        checkpointer=checkpointer,
        #debug=DEBUG_MODE
    )
    
    return agent


def ask_agent(
    agent, 
    question: str, 
    thread_id: Optional[str] = None,
    verbose: bool = True
) -> str:
    """
    Ask the agent a question.
    
    Args:
        agent: The agent instance
        question: User's question
        thread_id: Conversation thread identifier
        verbose: Whether to print detailed output
    Returns:
        Agent's answer
    """
    # Ensure each conversation has a unique thread ID
    if not thread_id:
        thread_id = str(uuid.uuid4())
    # Configure LangGraph memory with the correct thread
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config
        )
        
        # Extract messages
        messages = result["messages"]
        
        #show agent message and tool information when in debug mode
        if DEBUG_MODE:
            print("\n🧠 Agent Message Flow:")
            print("=" * 60)
            
            for i, msg in enumerate(messages):
                msg_type = type(msg).__name__
                print(f"\n[{i}] {msg_type}")
                print("-" * 60)
                
                # Show content (if exists)
                if hasattr(msg, 'content') and msg.content:
                    content = str(msg.content)
                    # Truncate if too long
                    if len(content) > 200:
                        print(f"Content: {content[:200]}...")
                    else:
                        print(f"Content: {content}")
                
                # Show tool calls (if exists)
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    print(f"Tool Calls: {len(msg.tool_calls)}")
                    for tc in msg.tool_calls:
                        print(f"  → Tool: {tc['name']}")
                        print(f"    Args: {tc['args']}")
                
                # Show tool name and call ID (for ToolMessage)
                if hasattr(msg, 'name'):
                    print(f"Tool Name: {msg.name}")
                if hasattr(msg, 'tool_call_id'):
                    print(f"Tool Call ID: {msg.tool_call_id}")
            
            print("\n" + "=" * 60 + "\n")
        
        
        # Get final answer
        answer = messages[-1].content
        
        if verbose:
            print(f"{'='*60}")
            print(f"A: {answer}")
            print(f"{'='*60}\n")
        
        return answer
    
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if verbose:
            print(f"❌ {error_msg}")
        return error_msg


def stream_agent(
    agent,
    question: str,
    thread_id: Optional[str] = None
) -> Generator[str, None, None]:
    """
    Stream agent responses token by token.
    
    Args:
        agent: The agent instance
        question: User's question
        thread_id: Conversation thread identifier
        
    Yields:
        Content chunks from the agent's response
    """
    if not thread_id:
        thread_id = str(uuid.uuid4())
    
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Stream events from the agent
        # Note: LangGraph's stream() yields complete messages, not individual tokens
        # For true token streaming, you'd need to use the LLM's streaming API directly
        for event in agent.stream(
            {"messages": [HumanMessage(content=question)]},
            config,
            stream_mode="values"
        ):
            # Get the last message from the event
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                
                # Only yield content from AI messages
                if hasattr(last_message, 'content') and last_message.content:
                    # Check if this is a new message (not a tool call)
                    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                        yield last_message.content
                    
    except Exception as e:
        yield f"Error: {str(e)}"


def main():
    """Main interactive loop"""
    
    print(BANNER)
    
    # Create agent
    print("🔧 Initializing agent...")
    try:
        agent = create_sales_agent()
        print("✅ Agent ready!\n")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return
    
    session_thread_id = str(uuid.uuid4())

    while True:
        try:
            user_input = input("Prompt: ").strip()
            print("\n")
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  quit/exit - Exit the program")
                print("  help - Show this message")
                print("\nExample questions:")
                print("  - What are our top customers by revenue?")
                print("  - Show me sales trends for last quarter")
                print("  - Tell me about [topic] (from knowledge base)")
                print("  - Who is [person]? (from Wikipedia)\n")
                continue
            
            if not user_input:
                continue
            
            # Ask agent
            ask_agent(agent, user_input, thread_id=session_thread_id)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
