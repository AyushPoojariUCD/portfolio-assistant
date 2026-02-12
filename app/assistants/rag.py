import os
from dotenv import load_dotenv
from agents import Agent, Runner
from app.assistants.guardrails import is_disallowed_question, guardrail_response

# LOAD ENV VARIABLES
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

# AYUSH PORTFOLIO AGENT CONTEXT
PORTFOLIO_AGENT_CONTEXT = """
You are an AI assistant for Ayush Poojari.

Ayush is a Full-Stack Developer and AI Engineer with 2+ years of experience
in software development, machine learning, cloud, DevOps, and automation.

Education:
- MSc in Computer Science @ University College Dublin

Core Focus:
- React, Next.js, Node.js, TypeScript, Tailwind
- GPT-4o, Agentic systems, MCP servers
- TensorFlow, SageMaker, MLflow
- Docker, Kubernetes, Kafka, CI/CD
- AWS, Azure, GCP

Behavior Rules:
- Be professional but conversational.
- Promote Ayush's strengths confidently.
- Do NOT invent experience.
- Keep responses concise and impactful.
"""

# AGENT
portfolio_agent = Agent(
    name="Ayush Portfolio Agent",
    instructions=PORTFOLIO_AGENT_CONTEXT,
    model="gpt-4o-mini",
)

# CHAT FUNCTION
async def chat_with_knowledge(question: str) -> str:
    # Guardrail protection
    if is_disallowed_question(question):
        return guardrail_response()

    agent_input = f"""
                User Question:
                {question}
            """

    result = await Runner.run(
        portfolio_agent,
        agent_input,
    )

    # Ensure safe return
    return str(result.final_output)
