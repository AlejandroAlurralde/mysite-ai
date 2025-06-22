"""
FastAPI + LangChain SQL Agent (read AND write capabilities)
Compatible with LangChain v0.3.x and Python 3.12
"""

import os
from typing import Annotated

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# LangChain v0.3 imports
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# ──────────────────────────────────────────────────────────────
# 0. Environment
# ──────────────────────────────────────────────────────────────
load_dotenv()                             # read .env file

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WRITE_TOKEN   = os.getenv("WRITE_TOKEN")

if not DATABASE_URL or not OPENAI_API_KEY or not WRITE_TOKEN:
    raise RuntimeError(
        "Missing DATABASE_URL, OPENAI_API_KEY or WRITE_TOKEN in .env"
    )

# ──────────────────────────────────────────────────────────────
# 1. SQLAlchemy engine & LangChain DB helper
# ──────────────────────────────────────────────────────────────
sql_db = SQLDatabase.from_uri(DATABASE_URL)
engine: Engine = sql_db._engine           # low-level access if needed
SessionLocal   = sessionmaker(bind=engine)

# ──────────────────────────────────────────────────────────────
# 2. Language model
# ──────────────────────────────────────────────────────────────
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=OPENAI_API_KEY,
)

# ──────────────────────────────────────────────────────────────
# 3-A. READ-ONLY agent  (same tools as before)
# ──────────────────────────────────────────────────────────────
read_agent = initialize_agent(
    tools=SQLDatabaseToolkit(db=sql_db, llm=llm).get_tools(),
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

# ──────────────────────────────────────────────────────────────
# 3-B. Custom WRITE tool  (INSERT / UPDATE / DELETE only)
# ──────────────────────────────────────────────────────────────
def run_write_sql(query: str) -> str:
    """Execute safe data-modifying SQL."""
    lowered = query.strip().lower()
    if not lowered.startswith(("insert", "update", "delete")):
        return "❌ Only INSERT / UPDATE / DELETE permitted."

    try:
        with engine.begin() as conn:            # auto-commit / rollback
            result = conn.execute(text(query))
            return f"✅ {result.rowcount} row(s) affected."
    except Exception as exc:
        return f"❌ SQL error: {exc}"

write_tool = Tool.from_function(
    name="sql_write",
    description="Run INSERT, UPDATE or DELETE statements on the database.",
    func=run_write_sql,
)

write_agent = initialize_agent(
    tools=[
        *SQLDatabaseToolkit(db=sql_db, llm=llm).get_tools(),
        write_tool,                            # ← our new tool
    ],
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

# ──────────────────────────────────────────────────────────────
# 4. FastAPI application
# ──────────────────────────────────────────────────────────────
app = FastAPI(title="SQL-Agent API", version="0.2")

class Question(BaseModel):
    question: str

# ---------- small dependency to check X-Token ----------
def auth_write(
    x_token: Annotated[str | None, Header()] = None
) -> None:
    if x_token != WRITE_TOKEN:
        raise HTTPException(status_code=401, detail="Bad X-Token")

# ---------- READ endpoint ----------
@app.post("/ask", summary="Ask the database (read-only)")
def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    answer = read_agent.run(q.question)
    return {"answer": answer}

# ---------- WRITE endpoint ----------
@app.post(
    "/edit",
    summary="Modify the database (INSERT / UPDATE / DELETE)",
    dependencies=[Depends(auth_write)],
)
def edit(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    result = write_agent.run(q.question)
    return {"result": result}
