import sqlite3
import pandas as pd
import os

db_filename = "gen_review.db"

if not os.path.exists(db_filename):
    print(f"❌ Error: Could not find '{db_filename}'.")
    exit(1)

conn = sqlite3.connect(db_filename)

# Added GROUP BY to ensure 100 strictly unique papers
query = """
SELECT 
    s.id AS paper_id, 
    s.title, 
    s.pdf AS pdf_url, 
    s.when_submitted,
    MAX(r.main_review) AS human_review, 
    MAX(r.binocular_score) AS human_binocular_score,
    MAX(g.generated) AS gpt4o_ai_review
FROM submission s
JOIN review r ON s.id = r.paper_id
JOIN genai_review g ON s.id = g.paper_id AND g.type = 'neutral'
WHERE r.main_review IS NOT NULL 
  AND g.generated IS NOT NULL
GROUP BY s.id
ORDER BY s.when_submitted DESC
LIMIT 100;
"""

df = pd.read_sql_query(query, conn)
df.to_json("reviews_metadata_100.json", orient="records", indent=2)
print(f"🎉 Success! Extracted {len(df)} UNIQUE papers.")
conn.close()