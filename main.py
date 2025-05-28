import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Dict, Any, Literal

# Import CORSMiddleware
from fastapi.middleware.cors import CORSMiddleware

# LangChain/LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
# CHANGE 1: Import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI # New import
# from langchain_deepseek import ChatDeepSeek # Remove or comment out this line
from langgraph.prebuilt import create_react_agent

# Example tool for web search
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

app = FastAPI()

# --- CORS Configuration ---
origins = [
    "https://safina-aichat.onrender.com",
    "https://safina-aichat.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
# CHANGE 2: Change variable name to GOOGLE_API_KEY
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# CHANGE 3: Update check for Google API key
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")
if not TAVILY_API_KEY:
    print("Warning: TAVILY_API_KEY not found in .env file. Tavily web search tool will not function.")
    tavily_search_enabled = False
else:
    tavily_search_enabled = True

# CHANGE 4: Instantiate ChatGoogleGenerativeAI
# Using 'gemini-1.5-flash' for its good balance of speed, capability, and cost in the free tier.
# 'gemini-pro' is also an option.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0)


# --- Define Tools (No changes here, as tools are standard LangChain tools) ---
tools = []
if tavily_search_enabled:
    @tool
    def tavily_search(query: str) -> str:
        """Perform a web search using Tavily and return relevant snippets."""
        try:
            tavily = TavilySearchResults(api_key=TAVILY_API_KEY, max_results=5)
            results = tavily.invoke({"query": query})
            return str(results)
        except Exception as e:
            return f"Error performing Tavily search: {e}"
    tools.append(tavily_search)
else:
    @tool
    def placeholder_search(query: str) -> str:
        """A placeholder tool for web search. Tavily API key is not configured."""
        return "Search functionality is currently unavailable (Tavily API key not set)."
    tools.append(placeholder_search)

# --- LangGraph Agent Setup using create_react_agent ---
app_runnable = create_react_agent(llm, tools)


# --- FastAPI Models for API Request/Response (No changes) ---
class ChatMessage(BaseModel):
    sender: str
    content: str
    tool_calls: List[Dict[str, Any]] = []

class ChatRequest(BaseModel):
    message: str
    chat_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    response: str
    chat_history: List[ChatMessage]

# --- FastAPI Endpoint (No significant changes needed) ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_message_content = request.message
    history_messages: List[BaseMessage] = []

    for msg in request.chat_history:
        if msg.sender == "user":
            history_messages.append(HumanMessage(content=msg.content))
        elif msg.sender == "bot":
            history_messages.append(AIMessage(content=msg.content, tool_calls=msg.tool_calls))

    full_chat_history = history_messages + [HumanMessage(content=user_message_content)]

    try:
        final_state = await app_runnable.ainvoke({"messages": full_chat_history})

        ai_response_message = None
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                ai_response_message = msg
                break
            elif isinstance(msg, HumanMessage):
                break
            elif isinstance(msg, ToolMessage):
                continue

        if ai_response_message:
            ai_response_content = ai_response_message.content
        else:
            ai_response_content = "Thinking... (AI is processing tool output or waiting for more context.)"

        new_chat_history: List[ChatMessage] = []
        for msg in final_state["messages"]:
            if isinstance(msg, HumanMessage):
                new_chat_history.append(ChatMessage(sender="user", content=msg.content))
            elif isinstance(msg, AIMessage):
                tool_calls_for_pydantic = [tc.dict() for tc in msg.tool_calls] if msg.tool_calls else []
                new_chat_history.append(ChatMessage(sender="bot", content=msg.content, tool_calls=tool_calls_for_pydantic))
            elif isinstance(msg, ToolMessage):
                new_chat_history.append(ChatMessage(sender="bot", content=f"*(Tool Output for {msg.tool_call_id}): {msg.content}*"))

        return ChatResponse(response=ai_response_content, chat_history=new_chat_history)

    except Exception as e:
        print(f"Error during chat processing: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# To run the FastAPI app:
# uvicorn main:app --reload