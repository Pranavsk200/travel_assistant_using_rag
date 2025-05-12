from typing import Optional
from langchain_core.runnables import Runnable, RunnableConfig
from customer_support_chat.app.core.state import State
from pydantic import BaseModel
from customer_support_chat.app.core.settings import get_settings
from langchain_openai import ChatOpenAI
from langchain_google_genai import chat_models
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from google.genai import types
from google import genai
settings = get_settings()

# llm = GenerativeModel(model_name="gemini-2.0-flash")
client = genai.Client(api_key="AIzaSyBJj2tNQ3JF9Opet6YKMk0EgGBfkXe38E0")
# llm = client.models
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=settings.GEMINI_API_KEY)
# Initialize the language model (shared among assistants)
# llm = ChatOpenAI(
#     model="gpt-4",
#     openai_api_key=settings.OPENAI_API_KEY,
#     temperature=1,
# )

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: Optional[RunnableConfig] = None):
        while True:
            result = self.runnable.invoke(state, config)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

# Define the CompleteOrEscalate tool
class CompleteOrEscalate(BaseModel):
    """A tool to mark the current task as completed or to escalate control to the main assistant."""
    cancel: bool = True
    reason: str