"""
app.py  –  FastAPI + LangChain SQL agent (IMPROVED VERSION)
────────────────────────────────────────────────────────────────────────────
Endpoints
    POST /ask    : read-only natural-language questions
    POST /edit   : INSERT / UPDATE / DELETE     (needs X-Token)
    POST /code   : write / overwrite files in ./workspace/ (needs X-Token)

IMPROVEMENTS in this version
───────────────────────────
✓  Smart parsing for /code endpoint - handles natural language instructions
✓  Better error messages with examples
✓  More flexible HTML validation
✓  Enhanced file serving with preview capabilities
✓  Comprehensive error handling
"""

from __future__ import annotations

import os
import pathlib
import re
from typing import Annotated, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import text as sa_text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker

# LangChain v0.3 family
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# ─────────────────────────── 0. ENV ────────────────────────────
load_dotenv()

DATABASE_URL   = os.getenv("DATABASE_URL", "sqlite:///:memory:")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WRITE_TOKEN    = os.getenv("WRITE_TOKEN", "letmein")          # demo token
WORKSPACE_DIR  = pathlib.Path("workspace")
WORKSPACE_DIR.mkdir(exist_ok=True)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing in .env")

# ────────────────────────── 1. FastAPI ─────────────────────────
app = FastAPI(title="SQL-Agent API", version="0.6-improved")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ───────────────────── 2. globals (built once) ─────────────────
sql_db: SQLDatabase | None = None
engine: Engine | None      = None
SessionLocal: sessionmaker | None = None
read_agent = write_agent = None

# ─────────────────── 3. IMPROVED PARSING LOGIC ─────────────────

def parse_code_instruction(instruction: str) -> Tuple[str, str]:
    """
    Smart parser for code instructions that handles multiple formats:
    1. Direct format: "workspace/file.ext\\ncontent..."
    2. Natural language: "Please create workspace/file.ext with content..."
    3. Casual instructions: "Make workspace/file.ext that says hello"
    
    Returns: (filename, content)
    """
    instruction = instruction.strip()
    if not instruction:
        raise ValueError("Empty instruction")
    
    # Strategy 1: Try direct format first (original expected format)
    lines = instruction.splitlines()
    first_line = lines[0].strip()
    
    if first_line.startswith("workspace/") and len(lines) > 1:
        # Original format - first line is the path
        filename = pathlib.Path(first_line).name
        content = "\\n".join(lines[1:]).lstrip()
        if content:  # Only return if we have content
            return filename, content
    
    # Strategy 2: Extract workspace/filename pattern from anywhere in text
    workspace_pattern = r'workspace/([a-zA-Z0-9._-]+(?:\\.[a-zA-Z0-9]+)?)'
    matches = re.findall(workspace_pattern, instruction)
    
    if not matches:
        raise ValueError(
            "No workspace/filename found. Examples of valid formats:\\n\\n"
            "Direct format:\\n"
            "workspace/hello.html\\n<h1>Hello World</h1>\\n\\n"
            "Natural language:\\n"
            "Please create workspace/hello.html with <h1>Hello World</h1>\\n\\n"
            "Casual instruction:\\n"
            "Make workspace/style.css with body { margin: 0; }"
        )
    
    filename = matches[0]  # Take first match
    
    # Strategy 3: Extract content after the workspace/filename mention
    workspace_full = f'workspace/{filename}'
    parts = instruction.split(workspace_full, 1)
    
    content = ""
    if len(parts) > 1:
        after_filename = parts[1].strip()
        
        # Remove common connecting words/phrases
        connecting_words = [
            'with the following content:',
            'with content:',
            'with the content:',
            'containing',
            'that contains',
            'that says',
            'with',
            'and',
        ]
        
        for word in connecting_words:
            if after_filename.lower().startswith(word.lower()):
                after_filename = after_filename[len(word):].strip()
                break
        
        # Remove leading colon or other punctuation
        after_filename = re.sub(r'^[:\\s]+', '', after_filename)
        content = after_filename
    
    if not content:
        raise ValueError(f"No content found for {filename}. Please specify what should be in the file.")
    
    return filename, content

def run_code_improved(instruction: str) -> str:
    """
    Improved version of run_code that handles natural language instructions
    """
    try:
        filename, content = parse_code_instruction(instruction)
        
        # Validate filename (enhanced security)
        if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
            raise ValueError(f"Invalid filename: {filename}. Only letters, numbers, dots, underscores, and hyphens allowed.")
        
        if not filename or '.' not in filename:
            raise ValueError(f"Filename must have an extension: {filename}")
        
        # Create file path
        dst = WORKSPACE_DIR / filename
        
        # Enhanced HTML validation
        if dst.suffix.lower() == ".html":
            content_check = content.lstrip().lower()
            # Accept HTML that starts with common HTML tags or add DOCTYPE if needed
            if not (content_check.startswith("<!doctype") or 
                   content_check.startswith("<html") or
                   content_check.startswith("<h") or
                   content_check.startswith("<p") or
                   content_check.startswith("<div")):
                # If it looks like HTML content but missing DOCTYPE, add it
                if any(tag in content_check for tag in ["<h1>", "<h2>", "<p>", "<div>", "<span>"]):
                    content = f"<!DOCTYPE html>\\n<html><body>\\n{content}\\n</body></html>"
        
        # Write file
        dst.write_text(content, encoding="utf-8")
        
        # Enhanced success message with file info
        file_size = dst.stat().st_size
        return f"✅ wrote {filename} ({file_size} bytes) - Preview at /workspace/{filename}"
        
    except Exception as e:
        # Enhanced error messages with examples
        error_msg = str(e)
        if "workspace/filename" in error_msg.lower():
            error_msg += "\\n\\nTip: The agent needs to format instructions properly for the code tool."
        
        raise ValueError(error_msg)

# ─────────────────────── 4. STARTUP LOGIC ──────────────────────

@app.on_event("startup")
def _startup() -> None:
    """Build DB helper + two agents once."""
    global sql_db, engine, SessionLocal, read_agent, write_agent

    # DB helper
    sql_db = SQLDatabase.from_uri(
        DATABASE_URL,
        view_support=False,
        sample_rows_in_table_info=2,
    )
    engine        = sql_db._engine
    SessionLocal  = sessionmaker(bind=engine)

    llm = ChatOpenAI(model="gpt-3.5-turbo",
                     api_key=OPENAI_API_KEY,
                     temperature=0)

    # READ-ONLY agent
    read_agent = initialize_agent(
        tools=SQLDatabaseToolkit(db=sql_db, llm=llm).get_tools(),
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=False,
    )

    # SQL write helper
    def run_write_sql(query: str) -> str:
        if not query.strip().lower().startswith(("insert", "update", "delete")):
            return "❌ Only INSERT / UPDATE / DELETE permitted."
        try:
            with engine.begin() as con:
                result = con.execute(sa_text(query))
            return f"✅ {result.rowcount} row(s) affected."
        except Exception as exc:
            return f"❌ SQL error: {exc}"

    sql_write_tool = Tool.from_function(
        name="sql_write",
        description="Execute INSERT / UPDATE / DELETE statements.",
        func=run_write_sql,
    )

    # IMPROVED code tool with better instructions for the agent
    code_tool = Tool.from_function(
        name="dev_write_file",
        description=(
            "Create or overwrite files inside ./workspace directory. "
            "Format the instruction as: workspace/filename.ext followed by the file content. "
            "For example: workspace/hello.html\\n<h1>Hello World</h1>"
        ),
        func=run_code_improved,
    )

    write_agent = initialize_agent(
        tools=[*SQLDatabaseToolkit(db=sql_db, llm=llm).get_tools(),
               sql_write_tool,
               code_tool],
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=False,
    )

    print("🚀  Agents ready (with improved /code parsing)")

# ─────────────────── 5. Schemas & tiny auth ────────────────────
class Prompt(BaseModel):
    question: str

def require_token(x_token: Annotated[str | None, Header()] = None):
    if x_token != WRITE_TOKEN:
        raise HTTPException(401, "Bad X-Token")

# ───────────────────────── 6. Routes ────────────────────────────

@app.get("/")
def root():
    """Enhanced root endpoint with usage examples"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SQL Agent API - Improved</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }
            .improvement { background: #d4edda; padding: 10px; margin: 10px 0; border-left: 4px solid #28a745; }
            code { background: #e9ecef; padding: 2px 4px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>SQL Agent API - Improved Version</h1>
        
        <div class="improvement">
            <h3>🎉 New in this version:</h3>
            <ul>
                <li>Smart parsing for /code endpoint - handles natural language</li>
                <li>Better error messages with examples</li>
                <li>More flexible HTML validation</li>
                <li>Enhanced file serving capabilities</li>
            </ul>
        </div>
        
        <h2>Endpoints</h2>
        
        <div class="endpoint">
            <strong>POST /ask</strong> - Read-only natural language questions<br>
            <code>{"question": "How many users are in the database?"}</code>
        </div>
        
        <div class="endpoint">
            <strong>POST /edit</strong> - INSERT/UPDATE/DELETE operations (requires X-Token)<br>
            <code>{"question": "Insert a new user with name John"}</code>
        </div>
        
        <div class="endpoint">
            <strong>POST /code</strong> - Create files in workspace (requires X-Token)<br>
            Now supports multiple formats:<br>
            • <code>{"question": "workspace/hello.html\\n&lt;h1&gt;Hello&lt;/h1&gt;"}</code><br>
            • <code>{"question": "Please create workspace/hello.html with &lt;h1&gt;Hello&lt;/h1&gt;"}</code><br>
            • <code>{"question": "Make workspace/style.css with body { margin: 0; }"}</code>
        </div>
        
        <div class="endpoint">
            <strong>GET /workspace/{filename}</strong> - View created files<br>
            <strong>GET /files</strong> - List all workspace files
        </div>
        
        <p><strong>Authentication:</strong> Use header <code>X-Token: {WRITE_TOKEN}</code> for /edit and /code endpoints.</p>
    </body>
    </html>
    """)

@app.post("/ask", summary="Ask (read-only)")
def ask(p: Prompt):
    if not p.question.strip():
        raise HTTPException(400, "Empty question")
    try:
        return {"answer": read_agent.run(p.question)}
    except Exception as e:
        raise HTTPException(500, f"Agent error: {str(e)}")

@app.post("/edit", summary="Change data", dependencies=[Depends(require_token)])
def edit(p: Prompt):
    if not p.question.strip():
        raise HTTPException(400, "Empty question")
    try:
        return {"result": write_agent.run(p.question)}
    except Exception as e:
        raise HTTPException(500, f"Agent error: {str(e)}")

@app.post("/code", summary="Write a file", dependencies=[Depends(require_token)])
def code(p: Prompt):
    if not p.question.strip():
        raise HTTPException(400, "Empty instruction")
    try:
        return {"result": write_agent.run(p.question)}
    except Exception as e:
        raise HTTPException(500, f"Agent error: {str(e)}")

# ─────────────────── 7. ENHANCED FILE SERVING ──────────────────

@app.get("/files")
def list_files():
    """List all files in workspace with metadata"""
    files = []
    for file_path in WORKSPACE_DIR.iterdir():
        if file_path.is_file():
            stat = file_path.stat()
            files.append({
                "filename": file_path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "url": f"/workspace/{file_path.name}"
            })
    return {"files": sorted(files, key=lambda f: f["modified"], reverse=True)}

@app.get("/preview/{filename}")
def preview_file(filename: str):
    """Preview file with basic syntax highlighting"""
    file_path = WORKSPACE_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {filename}")
    
    try:
        content = file_path.read_text(encoding="utf-8")
        
        # Simple HTML preview
        if filename.endswith('.html'):
            return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head><title>Preview: {filename}</title></head>
            <body>
                <h2>Preview: {filename}</h2>
                <iframe srcdoc="{content.replace('"', '&quot;')}" style="width:100%;height:400px;border:1px solid #ccc;"></iframe>
                <hr>
                <h3>Source:</h3>
                <pre style="background:#f8f9fa;padding:15px;border:1px solid #ccc;">{content.replace('<', '&lt;').replace('>', '&gt;')}</pre>
            </body>
            </html>
            """)
        else:
            # Text file preview
            return HTMLResponse(content=f"""
            <!DOCTYPE html>
            <html>
            <head><title>Preview: {filename}</title></head>
            <body>
                <h2>Preview: {filename}</h2>
                <pre style="background:#f8f9fa;padding:15px;border:1px solid #ccc;font-family:monospace;">{content.replace('<', '&lt;').replace('>', '&gt;')}</pre>
            </body>
            </html>
            """)
            
    except UnicodeDecodeError:
        raise HTTPException(400, f"Cannot preview binary file: {filename}")

# ────────────────────── 8. Static mounts ────────────────────────
# IMPORTANT: Mount specific routes BEFORE catch-all routes!

# Mount workspace FIRST (specific route)
app.mount("/workspace", StaticFiles(directory=WORKSPACE_DIR), name="workspace")

# Mount frontend LAST (catch-all route)
if os.path.isdir("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# ─────────────────────── 9. STARTUP MESSAGE ────────────────────
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting improved FastAPI + LangChain SQL Agent")
    print("📁 Workspace directory: ./workspace")
    print("🔧 /code endpoint now supports natural language instructions!")
    uvicorn.run(app, host="0.0.0.0", port=8000)
