"""
Very quick health-check for the SQL-Agent stack (CI-friendly).

• Imports every critical dependency.
• Loads environment variables (DATABASE_URL, OPENAI_API_KEY).
• Instantiates ChatOpenAI, SQLDatabase, and the agent.
• Asks one trivial question and succeeds **even if the LLM replies
  “I don’t know”** (thanks to handle_parsing_errors=True).

Exit code 0  → success
Exit code 1  → failure
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType

EXIT_SUCCESS, EXIT_FAIL = 0, 1


def main() -> int:
    try:
        # ---------------------------------------------------------------------
        # 1)  Load .env and sanity-check the two variables we need
        # ---------------------------------------------------------------------
        load_dotenv()

        db_url: str | None = os.getenv("DATABASE_URL")
        api_key: str | None = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing")
        if not db_url:
            raise RuntimeError("DATABASE_URL is missing")

        # ---------------------------------------------------------------------
        # 2)  Construct LLM, DB handle, Agent
        #     • We pass handle_parsing_errors=True so any answer is accepted.
        # ---------------------------------------------------------------------
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key, temperature=0)

        db = SQLDatabase.from_uri(db_url)

        agent = create_sql_agent(
            llm=llm,
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,   # <-- tolerate “I don’t know”
            verbose=False,
        )

        # ---------------------------------------------------------------------
        # 3)  Fire one lightweight query
        # ---------------------------------------------------------------------
        result = agent.invoke({"input": "How many tables are in the DB?"})

        print(json.dumps(result, indent=2, ensure_ascii=False))
        print("✅  Smoke-test OK")
        return EXIT_SUCCESS

    except Exception:
        print("❌  Smoke-test FAILED\n")
        traceback.print_exc()
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
