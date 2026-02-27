import os
from dotenv import load_dotenv

load_dotenv()

from src.langgraphagenticai.main import run_with_streaming


def main():
    print("Benney AI - University Chatbot (with history)")
    print("Ask a question (or 'quit' to exit)\n")
    messages = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        print("Benney: ", end="", flush=True)
        messages = run_with_streaming(user_input, messages)
        print()


if __name__ == "__main__":
    main()
