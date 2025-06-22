"""
LangChain SQL Agent Example
Compatible with LangChain v0.3 and Python 3.12

This example demonstrates how to create a SQL agent that can answer
natural language questions about your MySQL database.
"""

import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# LangChain imports (v0.3 compatible)
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents.agent_types import AgentType

# Load environment variables
load_dotenv()

def test_database_connection(database_url: str) -> bool:
    """
    Test database connection before using with LangChain
    This helps identify connection issues early
    """
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            # Test basic connection
            result = conn.execute(text("SELECT 1"))
            print("✓ Database connection successful!")
            
            # Test table listing (this is where NoneType errors often occur)
            result = conn.execute(text("SHOW TABLES"))
            tables = result.fetchall()
            table_names = [table[0] for table in tables]
            print(f"✓ Found {len(tables)} tables: {table_names}")
            
            if not tables:
                print("⚠️  Warning: No tables found in database")
                return False
                
            return True
            
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check if MySQL server is running")
        print("2. Verify database name exists")
        print("3. Check user permissions")
        print("4. Try alternative URL format (see setup_guide.md)")
        return False

def create_sample_data(database_url: str):
    """
    Create sample data for testing if no tables exist
    """
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            # Create a simple table for testing
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS employees (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    name VARCHAR(100),
                    department VARCHAR(50),
                    salary DECIMAL(10,2),
                    hire_date DATE
                )
            """))
            
            # Insert sample data
            conn.execute(text("""
                INSERT IGNORE INTO employees (id, name, department, salary, hire_date) VALUES
                (1, 'John Doe', 'Engineering', 75000.00, '2023-01-15'),
                (2, 'Jane Smith', 'Marketing', 65000.00, '2023-02-20'),
                (3, 'Bob Johnson', 'Engineering', 80000.00, '2022-11-10'),
                (4, 'Alice Brown', 'HR', 60000.00, '2023-03-05')
            """))
            
            conn.commit()
            print("✓ Sample data created successfully!")
            
    except Exception as e:
        print(f"❌ Failed to create sample data: {e}")

def main():
    """
    Main function to demonstrate SQL agent usage
    """
    # Configuration
    database_url = os.getenv("DATABASE_URL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Validate environment variables
    if not database_url:
        print("❌ DATABASE_URL not found in environment variables")
        print("Make sure your .env file contains:")
        print("DATABASE_URL=mysql+mysqlconnector://user:password@host:port/database")
        return
        
    if not openai_api_key:
        print("❌ OPENAI_API_KEY not found in environment variables")
        return
    
    print("🔍 Testing database connection...")
    if not test_database_connection(database_url):
        print("\n🔧 Attempting to create sample data...")
        create_sample_data(database_url)
        
        # Test again after creating sample data
        if not test_database_connection(database_url):
            return
    
    try:
        print("\n🤖 Initializing SQL Agent...")
        
        # Initialize the language model
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            api_key=openai_api_key
        )
        
        # Create SQLDatabase instance with error handling
        try:
            db = SQLDatabase.from_uri(database_url)
            print("✓ SQLDatabase created successfully!")
            
            # Test database info retrieval
            print(f"✓ Database dialect: {db.dialect}")
            print(f"✓ Available tables: {db.get_usable_table_names()}")
            
        except Exception as e:
            print(f"❌ Failed to create SQLDatabase: {e}")
            print("\nThis is likely the NoneType.replace error.")
            print("Try these solutions:")
            print("1. Use PyMySQL instead: mysql+pymysql://...")
            print("2. Check database URL format")
            print("3. Ensure database name is included in URL")
            return
        
        # Create the SQL agent
        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True
        )
        
        print("✓ SQL Agent created successfully!")
        print("\n" + "="*50)
        print("🎉 Setup complete! You can now ask questions about your database.")
        print("="*50)
        
        # Example queries
        example_queries = [
            "How many employees are in the database?",
            "What is the average salary by department?",
            "Who are the employees in the Engineering department?",
            "What is the highest salary in the company?"
        ]
        
        print("\n📝 Example queries you can try:")
        for i, query in enumerate(example_queries, 1):
            print(f"{i}. {query}")
        
        # Interactive mode
        print("\n💬 Interactive mode (type 'quit' to exit):")
        while True:
            try:
                question = input("\nYour question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not question:
                    continue
                    
                print(f"\n🤔 Processing: {question}")
                result = agent_executor.invoke({"input": question})
                print(f"\n✅ Answer: {result['output']}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error processing query: {e}")
                print("Try rephrasing your question or check the database connection.")
                
    except Exception as e:
        print(f"❌ Failed to initialize SQL Agent: {e}")
        print("\nCommon solutions:")
        print("1. Check OpenAI API key")
        print("2. Verify LangChain package versions")
        print("3. Review database connection")

if __name__ == "__main__":
    # Enable logging for debugging (optional)
    # logging.basicConfig(level=logging.INFO)
    # logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)
    
    main()

