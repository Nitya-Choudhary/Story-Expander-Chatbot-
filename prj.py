import streamlit as st
import os
from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage
import traceback

# 🔐 Load API keys securely from Streamlit secrets
os.environ["groq_api_key"] = st.secrets["groq_api_key"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
groq_api_key = os.environ["groq_api_key"]

# Supported model
MODEL_NAME = "llama3-70b-8192"

# Initialize tools (optional)
tool_tavily = TavilySearchResults(max_results=2)
tools = [tool_tavily]

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: Past Narratives
st.sidebar.title("🕛 Chat History")
with st.sidebar.expander("Past Narratives"):
    for idx, chat in enumerate(reversed(st.session_state.chat_history)):
        st.markdown(f"**{idx+1}. Genres:** {', '.join(chat['genre'])}")
        st.markdown(f"**Prompt:** {chat['prompt']}")

# Title and Inputs
st.title("Inkpot ✒️")
genre = st.multiselect("Genres:", ['Thriller', 'Romance', 'Fantasy', 'Self Help', 'Sci-Fi', 'Horror'])
prompt = st.text_area("Story Arc / Chapter Concept:", placeholder="Enter your story 🖊️", height=100)
story_length = "moderate"

# Handle "Narrate" button
if st.button("Narrate"):
    if prompt.strip():
        if not genre:
            st.warning("Please select at least one genre.")
        else:
            try:
                # System prompt for genre-aware story generation
                system_prompt = (
                    "You are a storytelling assistant. Based on the user's input, generate a complete short story "
                    "with a structured narrative: engaging opening, character development, plot twists, climax, and resolution. "
                    f"Ensure the writing reflects genre elements and matches the selected story length: {story_length}."
                )

                # Prepare messages
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Story idea: {prompt}. Genres: {', '.join(genre)}. Write a full story based on this arc."}
                ]

                # Initialize model and agent
                llm = ChatGroq(groq_api_key=groq_api_key, model_name=MODEL_NAME)
                agent = create_react_agent(llm, tools=tools)

                # Run the agent
                result = agent.invoke({"messages": messages})

                # Extract AI messages
                ai_response = [
                    msg.content
                    for msg in result["messages"]
                    if isinstance(msg, AIMessage)
                ]

                if ai_response:
                    story = ai_response[-1]
                    st.subheader("📖 Your Story")
                    st.markdown(story)

                    # Save to history
                    st.session_state.chat_history.append({
                        "genre": genre,
                        "prompt": prompt,
                        "response": story
                    })
                else:
                    st.warning("No story was generated. Try a different prompt or genre.")
            except Exception as e:
                traceback.print_exc()
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a story arc or chapter concept.")

