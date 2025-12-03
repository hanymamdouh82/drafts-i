import sqlite3
import sqlite_vec
import numpy as np
import json

DB = "notes.db"

def main():
    conn = sqlite3.connect(DB)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Prepare dummy vectors
    vec1 = np.ones(384, dtype=np.float32).tolist()
    vec2 = np.zeros(384, dtype=np.float32).tolist()

    # Convert to JSON arrays (sqlite-vec will parse automatically)
    j1 = json.dumps(vec1)
    j2 = json.dumps(vec2)

    # Insert dummy vectors
    conn.execute(
        "INSERT INTO chunks (file_path, text, embedding) VALUES (?,?,?)",
        ("demo1.md", "Test chunk one", j1),
    )
    conn.execute(
        "INSERT INTO chunks (file_path, text, embedding) VALUES (?,?,?)",
        ("demo2.md", "Test chunk two", j2),
    )
    conn.commit()

    # Insert into index
    conn.execute("""
        INSERT INTO chunks_index (rowid, embedding)
        SELECT id, embedding FROM chunks;
    """)
    conn.commit()

    # Query nearest neighbor to vec1
    q_json = json.dumps(vec1)

    cursor = conn.execute("""
        SELECT rowid, distance
        FROM chunks_index
        WHERE embedding MATCH ?
        ORDER BY distance
        LIMIT 2;
    """, (q_json,))

    print("--- Nearest neighbors ---")
    for row in cursor.fetchall():
        print(row)

    conn.close()

if __name__ == "__main__":
    main()
