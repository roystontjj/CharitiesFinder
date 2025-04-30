"""
LLM module for interacting with Google's Gemini API.

This module handles the connection to Google's Generative AI models and provides
functions for generating text responses.
"""

import google.generativeai as genai
from typing import List, Dict, Any, Optional

class GeminiClient:
    """Client for interacting with Google's Gemini LLM."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.0-pro"):
        """
        Initialize the Gemini client.
        
        Args:
            api_key (str): Gemini API key
            model (str): Model name to use (default: "gemini-1.0-pro")
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
    
    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Generate a text response from the LLM based on a prompt.
        
        Args:
            prompt (str): The prompt to send to the LLM
            temperature (float): Controls randomness in the output (0.0-1.0)
            
        Returns:
            str: The generated response
        """
        response = self.model.generate_content(
            prompt,
            generation_config={"temperature": temperature}
        )
        
        return response.text
    
    def generate_with_context(self, 
                           query: str, 
                           context: List[Dict[str, Any]],
                           temperature: float = 0.7) -> str:
        """
        Generate a response with additional context provided.
        
        Args:
            query (str): The user's query
            context (List[Dict[str, Any]]): List of context items to enhance the response
            temperature (float): Controls randomness in the output (0.0-1.0)
            
        Returns:
            str: The generated response
        """
        # Format context into a string
        context_str = self._format_context(context)
        
        # Create a prompt that includes the context and query
        prompt = f"""
You are an assistant for a charity finding application called CharityFinder. 
You help users find and learn about charities based on their interests.

Here is the user's query: "{query}"

I'll provide you with relevant information about charities in our database that might be helpful:

{context_str}

Based on the above context and the user's query, provide a helpful, informative response.
Mention specific charities from the context when relevant. If the user is looking for a 
charity recommendation, suggest appropriate ones from the provided context.
If the context doesn't contain information relevant to the user's query, acknowledge that
and provide general information about the topic if possible.
        """
        
        # Generate the response
        return self.generate_response(prompt, temperature)
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """
        Format context items into a string for inclusion in the prompt.
        
        Args:
            context (List[Dict[str, Any]]): List of context items
            
        Returns:
            str: Formatted context string
        """
        if not context:
            return "No relevant charity information available."
            
        context_parts = []
        
        for i, item in enumerate(context, start=1):
            # Format will depend on what fields your charity data contains
            # This is just an example
            charity_info = f"""
Charity {i}:
Name: {item.get('name', 'Unknown')}
Category: {item.get('category', 'Unknown')}
Mission: {item.get('mission', 'No mission statement available')}
Website: {item.get('website', 'No website available')}
"""
            context_parts.append(charity_info)
            
        return "\n".join(context_parts)