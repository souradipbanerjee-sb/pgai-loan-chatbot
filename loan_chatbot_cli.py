"""
Text-prompt chatbot in the terminal.

Usage:  python loan_chatbot_cli.py [--offline] [--debug]
  --offline  do not call Groq (rule-based parsing + template questions)
  --debug    show model score, latency and elapsed time after the decision
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dotenv import load_dotenv
load_dotenv(os.path.join(HERE, ".env"))
from chatbot_engine import LoanChatbot

sys.stdout.reconfigure(encoding="utf-8")


def main():
    bot = LoanChatbot(channel="cli", use_llm="--offline" not in sys.argv)
    print(f"[LLM: {'Groq ' + bot.llm.model if bot.llm else 'offline'}]  (type 'quit' to exit)\n")
    print("Bot:", bot.start(), "\n")
    while not bot.done:
        try:
            text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            text = "quit"
        if text.strip().lower() in {"quit", "exit"}:
            bot._finish("abandoned")
            print("Bot: Goodbye!")
            break
        print("\nBot:", bot.handle(text), "\n")
    if "--debug" in sys.argv and bot.decisions:
        for d in bot.decisions:
            print(f"[debug] ref {d['reference']} | P(decline) {d['probability_decline']} | threshold {d['threshold']} "
                  f"| decision {d['decision']} | model {d['latency_ms']} ms | decision {d['decision_latency_ms']} ms")
        print(f"[debug] completion time {bot.elapsed_s:.1f} s, {bot.n_user_turns} user turns")


if __name__ == "__main__":
    main()
