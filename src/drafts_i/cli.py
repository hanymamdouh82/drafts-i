import sys
from drafts_i.config import NOTES_DIR, DB_PATH


def main():
    if len(sys.argv) < 2:
        from drafts_i.query import query

        query()
        return

    cmd = sys.argv[1]

    if cmd == "ingest":
        from drafts_i.ingest import ingest_all

        ingest_all(NOTES_DIR, DB_PATH)

    elif cmd == "ask":
        from drafts_i.query import query

        query()

    else:
        print(f"Unknown command: {cmd}")
