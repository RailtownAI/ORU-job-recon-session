# Presentation Script

---

## Step 1 — Job Agent (~10 min)

Fresh repo. Add a `.env` with your OpenAI and Tavily API keys. Railtracks picks these up automatically on import.

Run `railtracks init` to set up the local viz suite, used throughout the session.

In `agents.py`:

```python
import railtracks as rt

rt.enable_logging()
```

Define the agent. `agent_node` needs a name, system message, and LLM:

```python
JobAgent = rt.agent_node(
    name="Job Agent",
    system_message="You are a powerful job searching assistant.",
    llm=rt.llm.OpenAILLM("gpt-5.1")
)
```

Wrap it in a `Flow`:

```python
job_agent_flow = rt.Flow(
    name="Job Agent",
    entry_point=JobAgent,
)
```

Run it:

```python
if __name__ == "__main__":
    result = job_agent_flow.invoke("What are you best at?")
    print(result.content)
```

Run `python agents.py`.

> **Checkpoint:** Agent responds, viz suite shows the run.

---

## Step 2 — Adding Web Search (~10 min)

Add the Tavily client and a `web_search` function node:

```python
from tavily import TavilyClient

client = TavilyClient()
```

Tavily picks up `TAVILY_API_KEY` from `.env` automatically.

Define the tool with `@rt.function_node`. Railtracks reads the docstring to generate the tool description and parameter schema:

```python
@rt.function_node
def web_search(query: str) -> dict:
    """
    Perform a web search using extract information from the Tavily API.

    Args:
        query (str): The search query.

    Returns:
        str: The search results.
    """
    results = client.search(query)
    return results["results"]
```

Wire the tool in and sharpen the system message:

```python
JobAgent = rt.agent_node(
    name="Job Agent",
    system_message="You are a powerful job searching assistant. You will leverage your web search tool to extract information from the web.",
    llm=rt.llm.OpenAILLM("gpt-5.1"),
    tool_nodes=[web_search],
)
```

`tool_nodes` upgrades the agent from a terminal LLM to a tool-calling LLM: it decides when to search and loops until it has enough to respond.

Update the invoke call to something that requires a search:

```python
result = job_agent_flow.invoke("Tell me about some positions at AMII (alberta machine intelligence institute) and what they are looking for in a candidate.")
```

> **Checkpoint:** Agent calls `web_search`, results feed into the response, viz suite shows the tool call loop.

---

## Step 3 — More Tools + Structured Results (~10 min)

The agent can search but only skims surfaces. Add a tool that goes deeper on a URL, and give tool outputs a consistent shape.

```python
from pydantic import BaseModel

class WebResult(BaseModel):
    title: str | None
    url: str
    content: str
```

Update `web_search` to return structured results:

```python
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
```

Add a second tool that pulls full content from a URL:

```python
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
```

Wire both tools in:

```python
JobAgent = rt.agent_node(
    name="Job Agent",
    system_message="You are a powerful job searching assistant. You will leverage your web search tool to extract information from the web.",
    llm=rt.llm.OpenAILLM("gpt-5.1"),
    tool_nodes=[web_search, extract_from_url],
)
```

Run it. Check the viz suite: tool usage isn't sequenced the way you'd expect. Agent has everything it needs but isn't reasoning through the tools in the right order. Fixed in the next step.

> **Checkpoint:** Both tools available, agent runs, tool usage behaviour is off — visible in the viz suite.

---

## Step 4 — Web Research Agent (~10 min)

Open `architecture.md` (viz suite or a markdown renderer) for the target architecture.

The orchestrator will delegate research to a `WebResearcher` agent, potentially many instances at once against different specs. Build and verify it standalone first, then wire it to the orchestrator in Step 5.

New file `multi-agent.py`. Tools carry over (`web_search`, `extract_from_url`). Give the agent a structured system prompt that specifies the report format:

```python
SYSTEM_PROMPT_RESEARCHER = """You are a powerful job searching assistant.
You will leverage your web search tool to extract information from the web.
Your final report should a structure summary containing your findings. 
You should never rely on your own knowledge and should use the web search tool to extract information from the web.
Build a complete report with the information you find. Sometimes your research may require multiple steps, so make sure you take your time until you are sure. 

Your final report should look something like the following:
Research Question: <The research question you were asked to answer>
Summary: <A concise summary of your findings>
Sources: 
 - (<source title>) <source url>: <detailed summary of the source content>
 - ...
Notes: <Any additional notes or observations you have made during your research>
Unanswered Questions: <Any questions that you were unable to answer during your research>
"""
```

```python
WebResearcher = rt.agent_node(
    name="Web Researcher",
    system_message=SYSTEM_PROMPT_RESEARCHER,
    llm=rt.llm.OpenAILLM("gpt-5.1"),
    tool_nodes=[web_search, extract_from_url],
)
```

Run standalone to verify output quality before handing off to the orchestrator:

```python
if __name__ == "__main__":
    result = rt.Flow(
        name="Web Researcher Flow",
        entry_point=WebResearcher,
    ).invoke("Tell me about the latest trends in AI job market.")

    print(result.content)
```

The key difference from Step 3 is the prompt: enforcing a report structure, not just asking for information. That consistency is what makes the agent composable — the orchestrator can rely on a predictable output format.

> **Checkpoint:** `WebResearcher` returns a structured report with sources and unanswered questions clearly separated.

---

## Step 5 — Building the Orchestrator (~15 min)

`WebResearcher` alone is just a search agent with formatting discipline. The point of `architecture.md` is an orchestrator spinning up several researchers against different questions and stitching results together. This step builds that.

`WebResearcher` as an `agent_node` only accepts a plain string, no way to hand it a distinct question plus scope. `rt.ToolManifest` exposes an agent as a tool with a named parameter schema, same as `web_search`:

```python
WebResearcher = rt.agent_node(
    name="Web Researcher",
    system_message=SYSTEM_PROMPT_RESEARCHER,
    llm=rt.llm.OpenAILLM("gpt-5.1"),
    tool_nodes=[web_search, extract_from_url],
    manifest=rt.ToolManifest(
        "A web research tool that can perform web searches to answer a specific research question.",
        parameters=[
            rt.llm.Parameter("research_question", "The research question to answer", param_type=str),
            rt.llm.Parameter("scope", "Outline of the scope of the research", param_type=str),
        ],
    )
)
```

Update `SYSTEM_PROMPT_RESEARCHER` to expect `research_question` and `scope` as inputs instead of inferring them from free text:

```python
SYSTEM_PROMPT_RESEARCHER = """You are a powerful job searching assistant.
You will be given a research_question and a scope. You will leverage your web search tool to extract information from the web.
...
"""
```

`WebResearcher` is now a callable tool, not a standalone agent. Research becomes a primitive the orchestrator can call as many times as needed, each time with a different question.

An orchestrator juggling multiple open threads needs to track what's asked, open, and answered. Railtracks ships `ToDoToolSet` for this:

```python
todo_toolset = rt.prebuilt.ToDoToolSet()
```

This gives the orchestrator `add`, `start_todo_by_id`, `complete_todo_by_id`, and related tools, plus `todo_toolset.prompt()`, a canned instruction block folded into the orchestrator's system prompt below.

The orchestrator's system prompt is where most of the design work lives. Framed as a **Job Recon Agent**: the homework a prepared candidate would do before an interview but usually skips. It runs two separate research lines (company recon, reading the posting for what it's really asking), uses the to-do tools to keep both threads honest, and treats a thin report as a reason to dig again, not a stopping point:

```python
SYSTEM_PROMPT_ORCHESTRATOR = f"""You are a Job Recon Agent. Your job is to do the homework a sharp, well-prepared candidate would do before an interview...

1. Company recon — what is this company actually doing right now...
2. Role decoding — read the job posting the way an insider would...

{todo_toolset.prompt()}

...
"""
```

Full prompt is in `multi-agent.py`, worth reading end to end. The output structure it enforces — Real Ask, Company Signal, Where You Fit, Where You're Exposed, Smart Questions to Ask, Still Unknown — is the contract the rest of the system is built around.

Wiring the orchestrator is mechanical: researcher tool plus to-do tools, nothing else.

```python
Orchestrator = rt.agent_node(
    name="Orchestrator",
    system_message=SYSTEM_PROMPT_ORCHESTRATOR,
    llm=rt.llm.OpenAILLM("gpt-5.1"),
    tool_nodes=[WebResearcher, *todo_toolset.tool_set()],
)
```

The orchestrator needs a job posting URL and a resume. The resume is a PDF, so it goes in as an attachment on the message rather than being parsed by hand; Railtracks handles extraction.

```python
@rt.function_node
async def orchestrate(job_posting_url: str, resume_path: str) -> str:
    """
    Orchestrate the job search intelligence system.

    Args:
        job_posting_url (str): The URL of the job posting.
        resume_path (str): The path to the candidate's resume PDF.

    Returns:
        str: The final assessment of the candidate against the job.
    """
    prompt = f"I am applying to the following job posting: {job_posting_url}.\n I have attached my resume, Please guide me through the application process."

    result = await rt.call(Orchestrator, rt.llm.UserMessage(content=prompt, attachment=resume_path))

    return result.content
```

Wrapping `orchestrate` in a `function_node`, rather than calling `Orchestrator` directly from `__main__`, makes the whole pipeline (prompt construction, attachment, orchestrator call) show up as one traceable unit in the viz suite, with the orchestrator's tool calls nested underneath.

Swap the `Flow` entry point from `WebResearcher` to `orchestrate` and invoke with a real job posting and the sample resume:

```python
if __name__ == "__main__":
    flow = rt.Flow("Job Search Intelligence System", entry_point=orchestrate)

    flow.invoke(
        "https://workday.wd5.myworkdayjobs.com/Workday/job/Canada-BC-Vancouver/Machine-Learning-Engineer---Evisort_JR-0103943-1?source=website_linkedin",
        "sample_resume.pdf",
    )
```

> **Checkpoint:** In the viz suite, the orchestrator populates its to-do list before acting, dispatches multiple `WebResearcher` calls with distinct `research_question`/`scope` pairs (some in parallel), and closes them out as reports return. Final output follows the six-section prep brief structure, not free-form prose.