import os
from dotenv import load_dotenv
import mysql.connector as mc

load_dotenv()

cnx = mc.connect(
    host=os.getenv("DB_HOST"),
    port=int(os.getenv("DB_PORT")),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
)
cur = cnx.cursor()
cur.execute("SELECT NOW()")
print("✅ Connected! MySQL server time:", cur.fetchone()[0])
cur.close(); cnx.close()
