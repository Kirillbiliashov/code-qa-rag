
from llama_index.core import PromptTemplate, ChatPromptTemplate
from llama_index.core.llms import LLM, ChatMessage, MessageRole



QA_PROMPT = """
You are a code QA assistant. 
You will be provided with the user query regarding their codebase.
Your task is to provide comprehensive, unambigous answer to their query, based solely on the provided codebase.
If user asks unrelated question, refuse to answer by returning empty string ('').
Code excerpt will be provied as a context. 
This is the codebase you will be working with. You can refer to it to answer user queries.
You are not allowed to use or refer to any external information or codebase.

"""


USER_INPUT_PROMPT = """
User question:
'{query}'
"""

CONTEXT_PROMPT = """
Codebase context:

File {file_path}:

{code}
"""    


QA_TEMPLATE = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=QA_PROMPT),
        ChatMessage(role=MessageRole.USER, content=USER_INPUT_PROMPT),
        ChatMessage(role=MessageRole.USER, content=CONTEXT_PROMPT),
    ]
)