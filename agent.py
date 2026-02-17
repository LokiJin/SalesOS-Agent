"""
An AI assistant with sales data and knowledge base capabilities 
"""

from typing import Optional, List
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

# Import configuration
from config import (
    MODEL_NAME, LLAMA_SERVER_URL, DEFAULT_TEMPERATURE, 
    DEFAULT_MAX_TOKENS, DEFAULT_THREAD_ID, DEBUG_MODE, BANNER
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
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
        streaming=True,
        max_tokens=DEFAULT_MAX_TOKENS
    )
    
    # Set up tools
    if tools is None:
        tools = [
            search_local_docs,      # Check internal docs first
            query_sales_database,   # Sales intelligence
            wiki_summary,           # External knowledge
            create_chart,           # Create visualizations
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
4. create_chart - Create visualizations (bar, line, pie, scatter, histogram)
5. create_multi_series_chart - Create multi-series charts for comparing metrics

Visualization Workflow:
When asked to create a chart or visualize data:
1. First, get the data using query_sales_database
2. Format the results as JSON
3. Call create_chart with the data
4. Tell the user the relative path to the chart

Example:
Q: "Show me a bar chart of top 5 customers by revenue"

Step 1: query_sales_database("SELECT c.company, SUM(s.total_amount) as revenue 
                               FROM sales s JOIN customers c ON s.customer_id=c.customer_id 
                               WHERE s.status='Completed' 
                               GROUP BY c.company 
                               ORDER BY revenue DESC LIMIT 5")
→ Returns: [{'company': 'Acme Corp', 'revenue': 50000}, ...]

Step 2: create_chart(
    data='[{"company": "Acme Corp", "revenue": 50000}, ...]',
    chart_type="bar",
    title="Top 5 Customers by Revenue",
    x_label="Customer",
    y_label="Revenue ($)"
)
→ Returns: "Chart saved to: charts/bar_20250216_143022.png"

Step 3: Respond to user with chart location relative path and summary

Tool Usage Guidelines:
- Sales DATABASE contains: transactions, revenue, customers, products, quantities
- Sales DATABASE does NOT contain: goals, targets, quotas, strategies, plans
- For goals/targets/quotas → ALWAYS use search_local_docs
- For actual sales numbers → use query_sales_database
- If question asks for BOTH (e.g., "sales vs goal") → use BOTH tools

Multi-tool examples:
- "Sales in Q1 and our goal" → query_sales_database + search_local_docs
- "Did we hit our target?" → query_sales_database + search_local_docs
- "Compare actual to budget" → query_sales_database + search_local_docs

Tool Usage Limits:
- Call each tool at most one per question unless new information is required.
- If a search does not return relevant new information, do NOT retry with similar wording.
- If monthly targets are not explicitly found, state that no monthly targets are defined.
- Do not rephrase and repeat the same search query.
- If results are similar to a previous tool call, proceed to final answer.

Example workflows:

Q: "Who are our top customers and should we offer them discounts?"
Step 1: query_sales_database("Top customers by revenue") → Get customer list
Step 2: search_local_docs("discount policy") → Get approval guidelines
Step 3: Synthesize: "Top customers are X, Y, Z. Per policy, Gold tier gets 10-20% off"
Step 4: Create graph reporting on top customers 

Always use multiple tools when questions have multiple components.

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
    thread_id: str = DEFAULT_THREAD_ID,
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
    
    try:
        # Configure conversation thread
        config = {"configurable": {"thread_id": thread_id}}
        
        # Invoke agent
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
                        print(f"Content: {content}...")
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


def main():
    """Main interactive loop"""
    
    print(BANNER)
    
    # Create agent
    print("🔧 Initializing agent...")
    agent = create_sales_agent()
    print("✅ Agent ready!\n")
    
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
            ask_agent(agent, user_input)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
