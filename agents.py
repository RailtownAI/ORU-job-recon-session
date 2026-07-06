import asyncio

import railtracks as rt
from tavily import TavilyClient
from pydantic import BaseModel

client = TavilyClient()

# rt.enable_logging()

SYSTEM_PROMPT = """You are a powerful job searching assistant. 
You will leverage your web search tool to extract information from the web. 
At every step make sure you log the steps you are completing. You thoughts should be short and concise details outlining your thoughts.
"""

class WebResult(BaseModel):
    title: str | None
    url: str
    content: str

@rt.function_node
def web_search(query: str) -> list[WebResult]:
    """
    Perform a web search using extract information from the Tavily API.

    Args:
        query (str): The search query.

    Returns:
        list[WebResult]: The search results.
    """
    results = client.search(query)
    
    return [WebResult(title=result["title"], url=result["url"], content=result["content"]) for result in results["results"]]

@rt.function_node
def extract_from_url(url: str) -> WebResult:
    """
    Extract information from a given URL using the Tavily API.

    Args:
        url (str): The URL to extract information from.

    Returns:
        WebResult: The extracted information.
    """
    result = client.extract(url)
    if len(result["results"]) == 0:
        raise ValueError(f"No results found for URL: {url}")
    
    extracted_result = result["results"][0]
    return WebResult(title=extracted_result.get("title"), url=extracted_result["url"], content=extracted_result["raw_content"])




JobAgent = rt.agent_node(
    name="Job Agent",
    system_message=SYSTEM_PROMPT,
    llm=rt.llm.OpenAILLM("gpt-5.1"),
    tool_nodes=[web_search, extract_from_url],
)

async def broadcast_hook(message: str):
    for char in message:
        print(char, end="")
        await asyncio.sleep(0.02)


job_agent_flow = rt.Flow(
    name="Job Agent",
    entry_point=JobAgent,
    broadcast_callback=broadcast_hook
)



if __name__ == "__main__":
    result = job_agent_flow.invoke("Tell me about some positions at AMII (alberta machine intelligence institute) and what they are looking for in a candidate.")
    print('--------------------------------')

    print(result.content)

