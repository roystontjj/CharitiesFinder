"""
Main Streamlit application for CharityFinder.

This is the entry point for the Streamlit web application that integrates
all the components together to provide the RAG-powered charity finder.
"""

import streamlit as st
import pandas as pd

from config import get_config
from database import SupabaseClient
from llm import GeminiClient
from rag import RAGProcessor

def initialize_components():
    """
    Initialize all the components needed for the application.
    
    Returns:
        tuple: (SupabaseClient, GeminiClient, RAGProcessor)
    """
    # Get configuration values
    config = get_config()
    
    # Initialize Supabase client
    db_client = SupabaseClient(
        url=config["supabase_url"],
        key=config["supabase_key"],
        table_name=config["supabase_table"],
        schema=config["supabase_schema"]
    )
    
    # Initialize Gemini LLM client
    llm_client = GeminiClient(api_key=config["gemini_api_key"])
    
    # Initialize RAG processor
    rag_processor = RAGProcessor(db_client=db_client, llm_client=llm_client)
    
    return db_client, llm_client, rag_processor

def display_charity_data(charities_df):
    """
    Display charity data in the Streamlit app.
    
    Args:
        charities_df (pd.DataFrame): DataFrame containing charity data
    """
    if charities_df.empty:
        st.info("No charities found matching your criteria.")
        return
    
    st.subheader("Charities")
    
    # Display as a table
    st.dataframe(charities_df)
    
    # You could also create expandable sections for each charity
    for i, row in charities_df.iterrows():
        with st.expander(f"{row.get('name', 'Charity')}"):
            cols = st.columns(2)
            with cols[0]:
                st.write(f"**Category:** {row.get('category', 'N/A')}")
                st.write(f"**Location:** {row.get('location', 'N/A')}")
            with cols[1]:
                st.write(f"**Website:** {row.get('website', 'N/A')}")
                st.write(f"**Contact:** {row.get('contact', 'N/A')}")
            
            st.write("**Mission:**")
            st.write(row.get('mission', 'No mission statement available.'))

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="CharityFinder", page_icon="🌟")
    
    st.title("🌟 CharityFinder")
    st.subheader("Find the perfect charity aligned with your interests and values")
    
    # Initialize components
    db_client, llm_client, rag_processor = initialize_components()
    
    # Sidebar for filtering options
    st.sidebar.title("Filters")
    
    # Get all categories for the dropdown
    try:
        all_charities = db_client.get_all_charities()
        categories = all_charities['category'].unique() if 'category' in all_charities.columns else []
        selected_category = st.sidebar.selectbox("Category", ["All"] + list(categories))
    except Exception as e:
        st.sidebar.error(f"Error loading categories: {str(e)}")
        selected_category = "All"
    
    # Search bar
    query = st.text_input("What kind of charity are you looking for?")
    
    # Process the query when submitted
    if query:
        with st.spinner("Finding charities that match your interests..."):
            try:
                if selected_category and selected_category != "All":
                    # Use category-specific RAG processing
                    response = rag_processor.process_query_with_category(query, selected_category)
                    
                    # Get charities in this category for display
                    charities_df = db_client.get_charities_by_category(selected_category)
                else:
                    # Use general RAG processing
                    response = rag_processor.process_query(query)
                    
                    # Get charities for display
                    charities_df = db_client.search_charities(query)
                
                # Display the LLM's response
                st.write("### AI Assistant Recommendations")
                st.write(response)
                
                # Display charity data
                display_charity_data(charities_df)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Display all charities when no search is performed
    else:
        try:
            if selected_category and selected_category != "All":
                charities_df = db_client.get_charities_by_category(selected_category)
            else:
                charities_df = db_client.get_all_charities()
            
            display_charity_data(charities_df)
        except Exception as e:
            st.error(f"Error loading charity data: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ | Powered by Gemini LLM")

if __name__ == "__main__":
    main()