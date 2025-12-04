import sys
from drafts_i.config import NOTES_DIR, DB_PATH


def main():
    # no command no argument
    if len(sys.argv) < 2:
        from drafts_i.query import query

        query()
        return
    # command + argument
    else:
        cmd = sys.argv[1]
        if cmd == "ask":
            from drafts_i.query import query

            query()

        elif cmd == "ingest":
            from drafts_i.ingest import ingest_all

            path_to_ingest = NOTES_DIR
            if len(sys.argv) > 2:
                path_to_ingest = sys.argv[2]

            ingest_all(path_to_ingest, DB_PATH)

        else:
            print(f"Unknown command: {cmd}")
