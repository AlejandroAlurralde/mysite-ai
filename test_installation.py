"""
Test script to verify LangChain installation and imports
This script tests that all required packages can be imported without conflicts
"""

def test_imports():
    """Test all critical imports"""
    print("Testing LangChain imports...")
    
    try:
        # Core LangChain imports
        from langchain_openai import ChatOpenAI
        print("✓ langchain_openai.ChatOpenAI")
        
        from langchain_community.utilities import SQLDatabase
        print("✓ langchain_community.utilities.SQLDatabase")
        
        from langchain_community.agent_toolkits import create_sql_agent
        print("✓ langchain_community.agent_toolkits.create_sql_agent")
        
        from langchain.agents.agent_types import AgentType
        print("✓ langchain.agents.agent_types.AgentType")
        
        # Database imports
        from sqlalchemy import create_engine, text
        print("✓ sqlalchemy")
        
        import mysql.connector
        print("✓ mysql.connector")
        
        import pymysql
        print("✓ pymysql")
        
        # Other imports
        from dotenv import load_dotenv
        print("✓ python-dotenv")
        
        import openai
        print("✓ openai")
        
        print("\n🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_versions():
    """Check package versions"""
    print("\nChecking package versions...")
    
    try:
        import langchain
        print(f"langchain: {langchain.__version__}")
        
        import langchain_core
        print(f"langchain-core: {langchain_core.__version__}")
        
        import langchain_community
        print(f"langchain-community: {langchain_community.__version__}")
        
        import langchain_openai
        print(f"langchain-openai: {langchain_openai.__version__}")
        
        import openai
        print(f"openai: {openai.__version__}")
        
        import sqlalchemy
        print(f"sqlalchemy: {sqlalchemy.__version__}")
        
        import pydantic
        print(f"pydantic: {pydantic.__version__}")
        
    except Exception as e:
        print(f"❌ Error checking versions: {e}")

def test_basic_functionality():
    """Test basic LangChain functionality without database"""
    print("\nTesting basic LangChain functionality...")
    
    try:
        from langchain_openai import ChatOpenAI
        
        # Test ChatOpenAI initialization (without API key)
        try:
            llm = ChatOpenAI(model="gpt-3.5-turbo", api_key="test-key")
            print("✓ ChatOpenAI initialization successful")
        except Exception as e:
            if "api_key" in str(e).lower():
                print("✓ ChatOpenAI initialization works (API key validation expected)")
            else:
                print(f"⚠️  ChatOpenAI initialization issue: {e}")
        
        # Test SQLDatabase class (without actual connection)
        from langchain_community.utilities import SQLDatabase
        print("✓ SQLDatabase class available")
        
        # Test agent creation function
        from langchain_community.agent_toolkits import create_sql_agent
        print("✓ create_sql_agent function available")
        
        print("✓ Basic functionality test passed!")
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")

if __name__ == "__main__":
    print("🧪 LangChain Installation Test")
    print("=" * 40)
    
    success = test_imports()
    test_versions()
    test_basic_functionality()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ Installation test PASSED!")
        print("You can now use the sql_agent_example.py script.")
    else:
        print("❌ Installation test FAILED!")
        print("Please check the error messages above and reinstall packages.")

