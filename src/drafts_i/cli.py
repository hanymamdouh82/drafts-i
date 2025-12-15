import sys
import argparse
import subprocess
from contextlib import contextmanager
from drafts_i.config import DOCKER_COMPOSE

@contextmanager
def qdrant_service():
    subprocess.run(
        ["docker", "compose", "-f", DOCKER_COMPOSE, "up", "-d"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )
    try:
        yield
    finally:
        subprocess.run(
            ["docker", "compose", "-f", DOCKER_COMPOSE, "stop"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )

def fAsk(args):
    from drafts_i.query import ask
    ask(args.question)


def fExplain(args):
    from drafts_i.query import explain
    explain(args.question)

def fInteractive(args):
    from drafts_i.query import interactive 
    interactive()

def fIngest(args):
    from drafts_i.ingest import ingest_all
    ingest_all()

def main():
    with qdrant_service():
        parser = argparse.ArgumentParser(
            prog="drafts-i",
            description="A RAG-Based knowledge base for my notes and Markdown files",
            epilog="Some text here"
        )

        subparser = parser.add_subparsers(
            title="commands",
            dest="command",
            required=True
        )

        # Ask
        cask = subparser.add_parser("ask", help="Ask a question and get short answer")
        cask.add_argument("question", help="Question to be answered")
        cask.set_defaults(func=fAsk)

        # Explain
        cexplain = subparser.add_parser("explain", help="ASk a question and get detailed answer")
        cexplain.add_argument("question", help="Question to be explained")
        cexplain.set_defaults(func=fExplain)

        # Interacvie
        interactive = subparser.add_parser("chat", help="Terminal-Based interactive session")
        interactive.set_defaults(func=fInteractive)

        # ingest 
        ingest = subparser.add_parser("ingest", help="Ingest dirs defined in config file")
        ingest.set_defaults(func=fIngest)

        args = parser.parse_args()

        args.func(args)
