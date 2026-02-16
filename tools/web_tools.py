"""
Web Tools
External information retrieval (Wikipedia summaries for now)
"""
import requests
from langchain_core.tools import tool

# Import config
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import DEBUG_MODE


@tool
def wiki_summary(query: str) -> str:
    """
    Fetch Wikipedia summary for general knowledge questions.
    
    Use for: historical facts, biographies, scientific concepts, definitions, company information, 
    and other encyclopedic information to augment the local knowledge base.
    
    Use as a secondary source for company-specific info, current events.
    
    Args:
        query: Topic to search on Wikipedia (e.g., "Quantum Computing", "Nikola Tesla", "ACME Inc.")
    
    Returns:
        Wikipedia summary or error message
    """
    
    try:
        if DEBUG_MODE:
            print(f"[WIKIPEDIA] Searching for: {query}")
        
        # Format query for Wikipedia API
        query_formatted = query.replace(" ", "_")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query_formatted}"
        
        headers = {'User-Agent': 'AgenticKB/1.0'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            return f"No Wikipedia page found for '{query}'. Try rephrasing."
        
        data = response.json()
        extract = data.get("extract", "No summary available.")
        
        # Add source URL
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        if page_url:
            extract += f"\n\nSource: {page_url}"
        
        if DEBUG_MODE:
            print(f"[WIKIPEDIA] Found summary ({len(extract)} chars)")
        
        return extract
    
    except requests.Timeout:
        return "Wikipedia request timed out. Try again later."
    except Exception as e:
        return f"Error fetching Wikipedia data: {str(e)}"
