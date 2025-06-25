"""
app.py  –  FastAPI + LangChain SQL agent (read-only & write)
fixed: StaticFiles mount now added *after* the API routes
"""

from __future__ import annotations
import os
from typing import Annotated

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# ─────────────────────────── 0. env ───────────────────────────
load_dotenv()

DATABASE_URL   = os.getenv("DATABASE_URL", "sqlite:///:memory:")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WRITE_TOKEN    = os.getenv("WRITE_TOKEN")

if not OPENAI_API_KEY or not WRITE_TOKEN:
    raise RuntimeError("OPENAI_API_KEY or WRITE_TOKEN missing in .env")

# ─────────────────────────── 1. FastAPI ───────────────────────
app = FastAPI(title="SQL-Agent API", version="0.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ─────────────────────────── 2. globals built once ────────────
sql_db: SQLDatabase | None = None
engine: Engine | None = None
session_maker: sessionmaker | None = None
read_agent = write_agent = None


@app.on_event("startup")
def _build_agents() -> None:
    global sql_db, engine, session_maker, read_agent, write_agent

    sql_db = SQLDatabase.from_uri(
        DATABASE_URL,
        view_support=False,
        sample_rows_in_table_info=2,
    )
    engine = sql_db._engine
    session_maker = sessionmaker(bind=engine)

    llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=OPENAI_API_KEY, temperature=0)

    read_agent = initialize_agent(
        tools=SQLDatabaseToolkit(db=sql_db, llm=llm).get_tools(),
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=False,
    )

    # ── write helper ────────────────────────────────
    def run_write_sql(query: str) -> str:
        lowered = query.strip().lower()
        if not lowered.startswith(("insert", "update", "delete")):
            return "❌ Only INSERT / UPDATE / DELETE permitted."
        try:
            with engine.begin() as conn:
                result = conn.execute(text(query))
                return f"✅ {result.rowcount} row(s) affected."
        except Exception as exc:
            return f"❌ SQL error: {exc}"

    write_tool = Tool.from_function(
        name="sql_write",
        description="Run INSERT / UPDATE / DELETE statements.",
        func=run_write_sql,
    )

    write_agent = initialize_agent(
        tools=[*SQLDatabaseToolkit(db=sql_db, llm=llm).get_tools(), write_tool],
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=False,
    )

    print("🚀  Agents built – API ready")


# ─────────────────────────── 3. schemas & deps ────────────────
class Question(BaseModel):
    question: str


def auth_write(x_token: Annotated[str | None, Header()] = None) -> None:
    if x_token != WRITE_TOKEN:
        raise HTTPException(status_code=401, detail="Bad X-Token")


# ─────────────────────────── 4. endpoints ─────────────────────
@app.post("/ask", summary="Ask the database (read-only)")
def ask(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    return {"answer": read_agent.run(q.question)}


@app.post(
    "/edit",
    summary="Modify the database (INSERT / UPDATE / DELETE)",
    dependencies=[Depends(auth_write)],
)
def edit(q: Question):
    if not q.question.strip():
        raise HTTPException(status_code=400, detail="Empty question")
    return {"result": write_agent.run(q.question)}


# ─────────────────────────── 5. static front-end … **last** ──
# Mount AFTER the API routes so it doesn’t shadow them
if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="static")
