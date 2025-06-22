"""
agent.py  ·  LangChain SQL agent (MySQL) — tested 2025-06

• Compatible package set
    langchain-core      ≥ 0.3.65,<0.4
    langchain-community ≥ 0.3.25,<0.4
    langchain-openai    ≥ 0.2.14,<0.3
    openai              ≥ 1.32,<2
    sqlalchemy          ≥ 2.0
    mysql-connector-python ≥ 8.4
    python-dotenv       ≥ 1.1

• Requires two env vars in .env (same folder):
      DATABASE_URL=mysql+mysqlconnector://user:pwd@127.0.0.1:3306/dbname
      OPENAI_API_KEY=sk-…

Run:  python agent.py
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType


# ── helpers ───────────────────────────────────────────────────────────────


def must_get(var: str) -> str:
    value = os.getenv(var)
    if not value:
        raise RuntimeError(f"{var} not set in environment (.env)")
    return value


def preflight(url: str) -> None:
    """Lightweight DB sanity-check (catches bad URL / perms early)."""
    eng = create_engine(url)
    try:
        with eng.connect() as con:
            con.execute(text("SELECT 1"))
            tables = [r[0] for r in con.execute(text("SHOW TABLES"))]
        print(f"✓ DB OK — {len(tables)} tables: {tables or 'none'}")
    finally:
        eng.dispose()


# ── bootstrap ─────────────────────────────────────────────────────────────

load_dotenv()                       # read .env

DATABASE_URL = must_get("DATABASE_URL")
OPENAI_KEY   = must_get("OPENAI_API_KEY")

preflight(DATABASE_URL)             # will raise if URL is wrong

db  = SQLDatabase.from_uri(DATABASE_URL)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_KEY)

agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,                       # prints reasoning & SQL
    handle_parsing_errors=True,
)

# ── REPL ──────────────────────────────────────────────────────────────────

EXAMPLES = [
    "How many employees are in the database?",
    "What is the average salary by department?",
    "Who works in the Engineering department?",
    "What is the highest salary in the company?",
]

print("\n🎉  SQL agent ready!  Example queries:")
for i, q in enumerate(EXAMPLES, 1):
    print(f"  {i}. {q}")

print("\n💬  Ask a question (blank or 'quit' to exit):")
while True:
    try:
        user = input("\n🗨︎ > ").strip()
        if not user or user.lower() in {"quit", "exit", "q"}:
            break
        answer = agent.run(user)
        print(f"\n🤖 {answer}")
    except (KeyboardInterrupt, EOFError):
        print("\n👋  Goodbye!")
        break
    except Exception as e:
        print(f"⚠️  {e}\nTry re-phrasing or check DB connection.")
