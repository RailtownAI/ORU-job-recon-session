# ORU Job Recon Agent Session

A multi-agent job recon system, built live in this session using [Railtracks](https://github.com/RailtownAI/railtracks). Give it a job posting and a resume, and it does the homework a sharp candidate would do before an interview: digs into the company, decodes what the role is actually asking for, and hands back a structured prep brief instead of a wall of text.

Under the hood, an orchestrator agent runs a to-do list dispatching a `WebResearcher` agent (sometimes several in parallel) against each open question, then stitching the results into a six-section report: Real Ask, Company Signal, Where You Fit, Where You're Exposed, Smart Questions to Ask, and Still Unknown.

## Structure
- `plan.md` -> the written walkthrough of how we built this, step by step
- `agents.py` -> the hello-world agent we started with
- `multi-agent.py` -> the full orchestrator + researcher system
- `multi-agent-architecture.png` -> diagram of the multi-agent architecture

## Thanks for coming

This repo is public, so feel free to come back to it, fork it, or rip it apart to build your own version.
