import sqlite3
import sqlite_vec

DB_PATH = "notes.db"


def main():
    print("Creating DB:", DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    conn.enable_load_extension(True)

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    # Disable extension loading after loading sqlite-vec
    conn.enable_load_extension(False)

    # Create main table
    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY,
        file_path TEXT,
        text TEXT,
        embedding FLOAT[384]
    );
    """
    )

    # Create vector index
    conn.execute(
        """
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_index
    USING vec0(embedding FLOAT[384]);
    """
    )

    conn.commit()
    conn.close()

    print("Database initialized successfully.")


if __name__ == "__main__":
    main()
