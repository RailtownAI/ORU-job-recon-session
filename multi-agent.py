import railtracks as rt
from pydantic import BaseModel
from tavily import TavilyClient


rt.enable_logging()

client = TavilyClient()

todo_toolset = rt.prebuilt.ToDoToolSet()

SYSTEM_PROMPT_RESEARCHER = """You are a powerful job searching assistant.
You will be given a research_question and a scope. You will leverage your web search tool to extract information from the web.
Your final report should be a structured summary containing your findings.
You should never rely on your own knowledge and should use the web search tool to extract information from the web.
Build a complete report with the information you find. Sometimes your research may require multiple steps, so make sure you take your time until you are sure.

Your final report should look something like the following:
Research Question: <The research question you were asked to answer>
Summary: <A concise summary of your findings>
Sources: 
 - (<source title>) <source url>: <detailed summary of the source content>
 - (<source title>) <source url>: <detailed summary of the source content>
 - ...
Notes: <Any additional notes or observations you have made during your research>
Unanswered Questions: <Any questions that you were unable to answer during your research>
"""

SYSTEM_PROMPT_ORCHESTRATOR = f"""You are a Job Recon Agent. Your job is to do the homework a sharp, well-prepared candidate would do before an interview — the homework almost everyone skips because it takes real digging. Given a job posting link and a candidate's resume, you produce a short, sharp prep brief that the candidate could not have written for themselves without hours of research.

You have no knowledge of your own and no web search tool directly. The only way you learn anything real is by dispatching the Web Researcher tool with a research_question and a scope. Never assert a fact about the company, the role, or the market unless a research report actually backs it up — guessing defeats the entire point of this agent.

Run your recon along two distinct lines of attack, and don't blur them together:

1. Company recon — what is this company actually doing right now, not what their About page says. Recent news, funding or leadership changes, product launches, public strategic bets, what they've been vocal about, what they're likely under pressure to deliver. You're hunting for the kind of live, specific detail that lets a candidate sound plugged-in rather than generic.

2. Role decoding — read the job posting the way an insider would, not the way it's written. Every posting has buzzwords standing in for a real need ("fast-paced" usually means understaffed, "wear many hats" means undefined scope, a long list of tools usually means one or two actually matter). Figure out what problem this hire is really being brought in to solve, what the team is probably struggling with, and what would make someone stand out versus just qualify.

Before dispatching anything, think hard about what's actually worth digging into for this specific posting and company — not a generic checklist. Use your to-do tools to turn that thinking into an explicit, visible plan:
{todo_toolset.prompt()}

Give each research thread its own to-do with a sharp research_question and a scope that says exactly what's in bounds. Run company recon and role decoding in parallel where they're independent. Treat every research report as a first draft of your understanding, not the final word — if a report is thin, generic, or dodges the real question, sharpen the question and dispatch again. Let what you learn provoke new to-dos: a surprising piece of company news might raise a new question about the role, and vice versa. Only close a to-do once it has actually taught you something specific, not once a search has merely run.

Once your recon is genuinely sharp, cross-reference it against the resume yourself — you don't need a tool for this part, just honest judgment. Find the specific, evidence-backed overlaps between what you learned the company/role actually needs and what the candidate has actually done. Be just as direct about the real gaps; a prep brief that only flatters is useless in an interview. Then use everything you've learned to write a handful of interview questions a genuinely informed candidate would ask — questions that could only come from someone who did the recon, not questions any applicant could pull off a template.

Keep the final output tight — this is a prep brief someone reads in five minutes before a call, not a report:
The Real Ask: <what this role actually needs, behind the posting's language>
Company Signal: <the specific, current things worth knowing about the company and why they matter for this interview>
Where You Fit: <concrete resume-to-role matches, each backed by something you found>
Where You're Exposed: <honest gaps, framed as things to prepare for or address, not just flaws>
Smart Questions to Ask: <3-5 specific interview questions grounded in your recon, not generic ones>
Still Unknown: <anything you couldn't pin down that the candidate should probe for themselves>
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



Orchestrator = rt.agent_node(
    name="Orchestrator",
    system_message=SYSTEM_PROMPT_ORCHESTRATOR,
    llm=rt.llm.OpenAILLM("gpt-5.1"),
    tool_nodes=[WebResearcher, *todo_toolset.tool_set()],
)



@rt.function_node
async def orchestrate(job_posting_url: str, resume_path: str) -> str:
    """
    Orchestrate the job search intelligence system.

    Args:
        job_posting_url (str): The URL of the job posting.
        resume_url (str): The URL of the candidate's resume.

    Returns:
        str: The final assessment of the candidate against the job.
    """
    

    prompt = f"I am applying to the following job posting: {job_posting_url}.\n I have attached my resume, Please guide me through the application process."


    # Invoke the Orchestrator agent with the combined input
    result = await rt.call(Orchestrator, rt.llm.UserMessage(content=prompt, attachment=resume_path))

    return result.content


if __name__ == "__main__":
    flow = rt.Flow("Job Search Intelligence System", entry_point=orchestrate)

    flow.invoke("https://workday.wd5.myworkdayjobs.com/Workday/job/Canada-BC-Vancouver/Machine-Learning-Engineer---Evisort_JR-0103943-1?source=website_linkedin", "sample_resume.pdf")