"""
Google search open tool.
Opens the system default browser to a Google search for a given query.
"""
import webbrowser
from typing import Dict, Any
from urllib.parse import urlencode
from wyzer.tools.tool_base import ToolBase


class GoogleSearchOpenTool(ToolBase):
    """Tool to open a Google search in the default browser"""
    
    def __init__(self):
        """Initialize google_search_open tool"""
        super().__init__()
        self._name = "google_search_open"
        self._description = "Open a Google search in the default browser for the given query"
        self._args_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The search query to look up on Google"
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Open a Google search for the given query.
        
        Args:
            query: Search query string
            
        Returns:
            Dict with ok status and url, or error
        """
        query = kwargs.get("query", "").strip()
        
        if not query:
            return {
                "error": {
                    "type": "invalid_query",
                    "message": "Query cannot be empty"
                }
            }
        
        try:
            # URL-encode the query and build the Google search URL
            encoded_query = urlencode({"q": query})
            url = f"https://www.google.com/search?{encoded_query}"
            
            # Open URL in a new browser tab (new=2)
            webbrowser.open(url, new=2)
            
            return {
                "ok": True,
                "url": url
            }
        except Exception as e:
            return {
                "error": {
                    "type": "execution_error",
                    "message": str(e)
                }
            }
