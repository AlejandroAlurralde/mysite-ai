"""
Very quick health-check for the SQL-Agent stack.

▪ Imports every critical dependency.
▪ Loads environment variables.
▪ Instantiates a ChatOpenAI client (no request is made).
▪ Creates a SQLDatabase connection.
▪ Runs one trivial query through the agent.

Return code 0 = success.
"""

import os, sys, json, traceback
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

EXIT_SUCCESS, EXIT_FAIL = 0, 1

def main() -> int:
    try:
        load_dotenv()

        db_url = os.getenv("DATABASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
        if not db_url or not api_key:
            raise RuntimeError("DATABASE_URL or OPENAI_API_KEY missing")

        # just create objects – this fails fast if packages mismatch
        llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
        db  = SQLDatabase.from_uri(db_url)
        agent = create_sql_agent(llm=llm, db=db)

        # light query (zero-shot) – keep it deterministic
        result = agent.invoke({"input": "How many tables are in the DB?"})
        print(json.dumps(result, indent=2))
        print("✅  Smoke-test OK")
        return EXIT_SUCCESS

    except Exception as e:
        print("❌  Smoke-test FAILED\n")
        traceback.print_exc()
        return EXIT_FAIL

if __name__ == "__main__":
    sys.exit(main())
