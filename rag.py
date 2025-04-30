"""
RAG (Retrieval-Augmented Generation) module.

This module integrates the database and LLM components to implement
a RAG system that enhances LLM responses with retrieved data.
"""

from typing import Dict, List, Any

from database import SupabaseClient
from llm import GeminiClient

class RAGProcessor:
    """
    A class that implements Retrieval-Augmented Generation by combining
    database retrieval with LLM generation.
    """
    
    def __init__(self, db_client: SupabaseClient, llm_client: GeminiClient):
        """
        Initialize the RAG processor.
        
        Args:
            db_client (SupabaseClient): Instance of the Supabase client
            llm_client (GeminiClient): Instance of the Gemini LLM client
        """
        self.db_client = db_client
        self.llm_client = llm_client
    
    def process_query(self, query: str, max_context_items: int = 5) -> str:
        """
        Process a user query through the RAG pipeline.
        
        This function retrieves relevant context from the database and then
        uses that context to enhance the LLM's response to the query.
        
        Args:
            query (str): The user's query
            max_context_items (int): Maximum number of context items to retrieve
            
        Returns:
            str: The RAG-enhanced response
        """
        # Step 1: Retrieve relevant context from the database
        context = self.db_client.get_context_for_rag(query, limit=max_context_items)
        
        # Step 2: Generate a response using the LLM with the retrieved context
        response = self.llm_client.generate_with_context(query, context)
        
        return response
    
    def process_query_with_category(self, query: str, category: str) -> str:
        """
        Process a user query filtered by charity category.
        
        Args:
            query (str): The user's query
            category (str): Category to filter charities by
            
        Returns:
            str: The RAG-enhanced response focused on a specific category
        """
        # Get charities in the specified category
        charities_df = self.db_client.get_charities_by_category(category)
        
        if charities_df.empty:
            return f"I couldn't find any charities in the '{category}' category. Please try another category."
        
        # Convert DataFrame to list of dictionaries for context
        context = charities_df.to_dict(orient="records")
        
        # Generate response with this category-specific context
        response = self.llm_client.generate_with_context(query, context)
        
        return response