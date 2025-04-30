"""
Database module for interacting with Supabase.

This module handles all database connections and operations for the CharityFinder application.
"""

import pandas as pd
from supabase import create_client
from typing import Dict, List, Any, Optional

class SupabaseClient:
    """A client class for interacting with Supabase."""
    
    def __init__(self, url: str, key: str, table_name: str = "charities", schema: str = "public"):
        """
        Initialize the Supabase client.
        
        Args:
            url (str): Supabase URL
            key (str): Supabase API key
            table_name (str): Name of the table to query (default: "charities")
            schema (str): Database schema (default: "public")
        """
        self.client = create_client(url, key)
        self.table_name = table_name
        self.schema = schema
        
    def get_all_charities(self) -> pd.DataFrame:
        """
        Retrieve all charities from the database.
        
        Returns:
            pd.DataFrame: DataFrame containing all charity records
        """
        response = self.client.table(self.table_name).select("*").execute()
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(response.data)
        return df
    
    def search_charities(self, query: str, limit: int = 10) -> pd.DataFrame:
        """
        Search for charities based on a text query.
        
        Args:
            query (str): Search query text
            limit (int): Maximum number of results to return
            
        Returns:
            pd.DataFrame: DataFrame containing matching charity records
        """
        # Using text search if your Supabase has text search capabilities
        # This is a basic implementation and might need adaptation based on your schema
        response = self.client.table(self.table_name).select("*").textsearch(
            "name", query  # Assuming there's a name column to search in
        ).limit(limit).execute()
        
        return pd.DataFrame(response.data)
    
    def get_charity_by_id(self, charity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific charity by its ID.
        
        Args:
            charity_id (str): The ID of the charity to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Charity data or None if not found
        """
        response = self.client.table(self.table_name).select("*").eq("id", charity_id).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        return None
    
    def get_charities_by_category(self, category: str, limit: int = 10) -> pd.DataFrame:
        """
        Get charities filtered by category.
        
        Args:
            category (str): Category to filter by
            limit (int): Maximum number of results to return
            
        Returns:
            pd.DataFrame: DataFrame containing matching charity records
        """
        # Assuming there's a category column
        response = self.client.table(self.table_name).select("*").eq("category", category).limit(limit).execute()
        
        return pd.DataFrame(response.data)
    
    def get_context_for_rag(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get relevant charity data to use as context for RAG.
        
        Args:
            query (str): The query to use for finding relevant context
            limit (int): Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of charity data that can be used as context
        """
        # Basic implementation - in production, you might want more sophisticated retrieval
        df = self.search_charities(query, limit=limit)
        
        if df.empty:
            return []
        
        # Convert DataFrame back to list of dictionaries
        return df.to_dict(orient="records")