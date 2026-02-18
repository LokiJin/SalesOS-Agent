"""
Sales Database Tool
Handles all sales data queries using LLM-powered text-to-SQL
"""
import sqlite3

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Import config
import sys
from pathlib import Path
from config import SALES_DB_PATH, LLAMA_SERVER_URL, MODEL_NAME, DEBUG_MODE, SQL_PRINTING_ENABLED

sys.path.append(str(Path(__file__).parent.parent))

# Cache schema at module level (loaded once)
_SCHEMA_CACHE = None

def _get_schema_cached():
    """Get database schema (cached after first call)"""
    global _SCHEMA_CACHE
    
    if _SCHEMA_CACHE is None:
        
        conn = sqlite3.connect(f'file:{SALES_DB_PATH}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        _SCHEMA_CACHE = _get_database_schema(cursor)
        conn.close()
          
    return _SCHEMA_CACHE

@tool
def query_sales_database(question: str) -> str:
    """
    Query the company sales database for business intelligence using natural language.
    
    Use for questions about: 
    - Sales performance, revenue, trends
    - Customer data, orders, purchasing patterns  
    - Product performance, bestsellers
    - Regional performance
    - Sales rep performance
    - Time-based analysis (monthly, quarterly, YTD)
    
    Examples:
    - "What were total sales last quarter?"
    - "Who are our top 5 customers by revenue?"
    - "Which products sell best in Europe?"
    - "Show me monthly revenue trend for 2024"
    
    Args:
        question: Natural language question about sales data
    """
    
    try:
        # Connect to database
        conn = sqlite3.connect(f'file:{SALES_DB_PATH}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        schema = _get_schema_cached()  # ← Uses cache
        
        # Use LLM to generate SQL
        sql_query = _generate_sql_with_llm(question, schema)
        
        if sql_query.startswith("ERROR:"):
            return sql_query  # Return the error message to agent

        if SQL_PRINTING_ENABLED or DEBUG_MODE:
            print(f"[SQL QUERY] {sql_query}")
        
        # Validate SQL for safety
        _validate_sql(sql_query)

        # Execute query
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        conn.close()
        
        if not results:
            return f"Query executed but returned no results for: {question}"
        
        rows = [dict(row) for row in results]
        return "\n".join(str(r) for r in rows)
        # Format and return results
        #return _format_results(results, question) todo: figure out if we want to format results or just return raw data for agent to interpret
    
    except sqlite3.Error as e:
        return f"Database error: {str(e)}\nThe generated SQL may be invalid. Try rephrasing."
    except Exception as e:
        return f"Error: {str(e)}"



def _get_database_schema(cursor) -> str:
    """Get detailed database schema with sample data"""
    
    schema_parts = []
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    
    for table in tables:
        # Get column info
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        
        schema_parts.append(f"\nTable: {table}")
        schema_parts.append("Columns:")
        for col in columns:
            schema_parts.append(f"  - {col[1]} ({col[2]})")
        
        # Get sample row
        cursor.execute(f"SELECT * FROM {table} LIMIT 1")
        sample = cursor.fetchone()
        if sample:
            schema_parts.append("Sample data:")
            sample_dict = dict(sample)
            for key, value in list(sample_dict.items())[:5]:
                schema_parts.append(f"  {key}: {value}")
    
    schema_text = "\n".join(schema_parts)
    
    # Add usage guidelines
    guidelines = """

Key Relationships:
- customers.region_id → regions.region_id
- sales.customer_id → customers.customer_id
- sales_items.sale_id → sales.sale_id
- sales_items.product_id → products.product_id

SQL Guidelines:
- The sales.total_amount column already includes all discounts (pre-calculated)
- For revenue/totals: Use SUM(sales.total_amount) - fast and accurate
- For product-level analysis: Join sales_items and calculate item revenue
- Monthly grouping: strftime('%Y-%m', sale_date)
- Last quarter: date('now', '-3 months')
- This year: strftime('%Y', sale_date) = strftime('%Y', 'now')
- Customer tiers: Bronze, Silver, Gold, Platinum
- Always use explicit JOINs

Query Pattern Examples:
1. Revenue questions → Use sales.total_amount:
   SELECT SUM(total_amount) FROM sales WHERE status='Completed'
   
2. Product questions → Join to sales_items:
   SELECT p.product_name, SUM(si.quantity * si.unit_price * (1 - si.discount))
   FROM sales_items si JOIN products p ON si.product_id = p.product_id
   JOIN sales s ON si.sale_id = s.sale_id
   WHERE s.status='Completed'
   
3. Customer revenue → Use sales.total_amount:
   SELECT c.company, SUM(s.total_amount)
   FROM sales s JOIN customers c ON s.customer_id = c.customer_id
   WHERE s.status='Completed'
"""
    
    return schema_text + guidelines

def _validate_sql(query: str) -> None:
    """Validate SQL query for security"""
    
    query_upper = query.upper()
    
    # Block dangerous keywords
    forbidden = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 
                 'CREATE', 'TRUNCATE', 'REPLACE']
    
    for keyword in forbidden:
        if keyword in query_upper:
            raise ValueError(f"Query contains forbidden keyword: {keyword}")
        

def _generate_sql_with_llm(question: str, schema: str) -> str:
    """Use LLM to convert natural language to SQL"""
    
    sql_llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.0,
        base_url=LLAMA_SERVER_URL,
        max_tokens=3500
    )
    prompt = f"""You are an expert SQL query generator. Convert the question into a valid SQLite query.

DATABASE SCHEMA:
{schema}

QUESTION: {question}

RULES:
1. Generate ONE valid SQLite query
2. Use explicit JOINs
3. Filter completed sales: status = 'Completed'
4. Return ONLY the SQL query (no markdown, no explanations)
5. Use aggregations appropriately (SUM, COUNT, AVG)
6. IMPORTANT: Only query for data that exists in the schema above
7. If the question asks for data NOT in the schema (like goals, targets, quotas), 
   return ONLY: "ERROR: Cannot answer - [missing data] not in database"
8. Do NOT add LIMIT for queries that compute totals, averages, or group by time periods. Only add LIMIT for queries that retrieve individual rows (top N customers, recent sales, etc.)
9. Use ORDER BY for meaningful sorting

Examples:
- Question asks for "sales goal" but no goal column exists → 
  Return: "ERROR: Cannot answer - sales goals not in database"
- Question asks for "customer satisfaction" but no satisfaction data → 
  Return: "ERROR: Cannot answer - satisfaction data not in database"

  IMPORTANT - Customer Name Convention:
- When asked about "customers", "top customers", "which customers", etc. 
  → Use customers.company (the business/organization name)
- Only use customers.customer_name (contact person name) when specifically asked for:
  * "contact name", "who is the contact", "point of contact", "account manager name"
- Examples:
  * "Who are our top customers?" → SELECT company, SUM(total_amount)...
  * "Best customers by revenue?" → SELECT company, SUM(total_amount)...
  * "What's the contact name for Acme?" → SELECT customer_name FROM customers WHERE company = 'Acme'

SQL Query:"""
    
    try:
        response = sql_llm.invoke([HumanMessage(content=prompt)])
        sql_query = response.content.strip()
        
        # Clean up formatting
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        sql_query = sql_query.rstrip(';')
        
        return sql_query
    
    except Exception as e:
        if DEBUG_MODE:
            print(f"[ERROR] SQL generation failed: {e}")
        # Safe fallback
        return "SELECT * FROM sales WHERE status = 'Completed' ORDER BY sale_date DESC LIMIT 10"


# def _format_results(results, question: str) -> str: #todo: figure out if we want to format results or just return raw data for agent to interpret
#     """Format query results for readability"""
    
#     if not results:
#         return "No results found."
    
#     rows = [dict(row) for row in results]
    
#     # Single aggregate result (one row, few columns)
#     if len(rows) == 1 and len(rows[0]) <= 3:
#         parts = []
#         for key, value in rows[0].items():
#             if isinstance(value, (int, float)):
#                 # Format money
#                 if any(word in key.lower() for word in ["revenue", "amount", "spent", "value", "price"]):
#                     parts.append(f"{key}: ${value:,.2f}")
#                 # Format large numbers
#                 elif value > 1000:
#                     parts.append(f"{key}: {value:,}")
#                 else:
#                     parts.append(f"{key}: {value}")
#             else:
#                 parts.append(f"{key}: {value}")
#         return "\n".join(parts)
    
#     # Multiple rows - format as table
#     output = []
#     headers = list(rows[0].keys())
    
#     # Header row
#     output.append(" | ".join(headers))
#     output.append("-" * 60)
    
#     # Data rows (limit to 40)
#     for row in rows[:40]:
#         formatted_cells = []
#         for key, value in row.items():
#             if value is None:
#                 formatted_cells.append("N/A")
#             elif isinstance(value, float):
#                 if any(word in key.lower() for word in ["revenue", "amount", "spent", "value", "price", "cost"]):
#                     formatted_cells.append(f"${value:,.2f}")
#                 else:
#                     formatted_cells.append(f"{value:.2f}")
#             elif isinstance(value, int) and value > 1000:
#                 formatted_cells.append(f"{value:,}")
#             else:
#                 formatted_cells.append(str(value))
#         output.append(" | ".join(formatted_cells))
    
#     if len(rows) > 40:
#         output.append(f"\n... and {len(rows) - 40} more rows")
    
#     return "\n".join(output)
