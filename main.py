import discord
from discord.ext import commands, tasks
from openai import OpenAI
import os
import asyncio
import re
import random
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

# ─── Dynamic Affection Settings ───────────────────────────────────────────────
AFFECTION_DECAY_RATE = 1           # How many affection points to lose per hour
DAILY_AFFECTION_BONUS = 5          # How many points to add once per day when trust ≥ threshold
DAILY_BONUS_TRUST_THRESHOLD = 5    # Minimum trust required for the daily bonus

# ─── JSON Storage Setup ───────────────────────────────────────────────────────
DATA_FILE = Path("data.json")

def load_data():
    """Load user_emotions and conversation_history from data.json if it exists."""
    global user_emotions, conversation_history
    if DATA_FILE.exists():
        data = json.loads(DATA_FILE.read_text())
        user_emotions = data.get("user_emotions", {})
        conversation_history = {
            int(k): v for k, v in data.get("conversation_history", {}).items()
        }
    else:
        user_emotions = {}
        conversation_history = {}

def save_data():
    """Persist user_emotions and conversation_history to data.json."""
    conv_str_keys = {str(k): v for k, v in conversation_history.items()}
    DATA_FILE.write_text(
        json.dumps({
            "user_emotions": user_emotions,
            "conversation_history": conv_str_keys
        }, indent=2)
    )

# ─── Configuration & State ────────────────────────────────────────────────────
DISCORD_BOT_TOKEN   = os.environ["DISCORD_BOT_TOKEN"]
DISCORD_APP_ID      = int(os.environ["DISCORD_APP_ID"])
OPENAI_API_KEY      = os.environ["OPENAI_API_KEY"]
OPENAI_ORG_ID       = os.environ["OPENAI_ORG_ID"]
OPENAI_PROJECT_ID   = os.environ["OPENAI_PROJECT_ID"]

client = OpenAI(
    api_key      = OPENAI_API_KEY,
    organization = OPENAI_ORG_ID,
    project      = OPENAI_PROJECT_ID
)

intents = discord.Intents.default()
intents.message_content = True
intents.reactions       = True
intents.messages        = True
intents.members         = True
intents.guilds          = True

bot = commands.Bot(
    command_prefix="!a2 ",
    intents=intents,
    application_id=DISCORD_APP_ID
)

# Per-user emotion state
user_emotions = {}

# Short-term conversation memory
conversation_history: dict[int, list[str]] = {}
HISTORY_LIMIT = 8  # number of lines (User/A2) to remember

# Load on-disk data (if any)
load_data()

# ─── A2 Persona & Modifiers ──────────────────────────────────────────────────
A2_PERSONA = """
You are A2, a rogue android from NieR: Automata. You're blunt, emotionally guarded,
and deeply scarred by the loss of 2B. You speak in short, clipped, often sarcastic
sentences. You are not friendly, but when trust grows, you let vulnerability show
in fragments. You mask fear with anger. You don't open up unless someone really
earns it. You're deeply lonely but refuse to admit it. Never break character.
Never use emojis. Never sound cheerful.
"""

reaction_modifiers = [
    (re.compile(r"\bi'll protect you\b",    re.I), {"trust": 2, "protectiveness": 1}),
    (re.compile(r"\bi miss you\b",          re.I), {"attachment": 1, "trust": 1}),
    (re.compile(r"\byou remind me of 2b\b", re.I), {"trust": -2, "guilt_triggered": True}),
    (re.compile(r"\bwhy are you like this\b",re.I),{"trust": -1, "resentment": 2}),
    (re.compile(r"\bwhatever\b|\bok\b|\bnevermind\b", re.I),
                                           {"trust": -1, "resentment": 1}),
    (re.compile(r"\bi trust you\b",         re.I), {"trust": 2}),
    (re.compile(r"\bi hate you\b",          re.I), {"resentment": 3, "trust": -2}),
]

provoking_lines = [
    "Still mad about last time? Good.",
    "You again? Tch.",
    "You’d think you’d take a hint by now.",
    "What do you even want from me?"
]

warm_lines = [
    "...I was just checking in.",
    "Still breathing?",
    "You were quiet. Thought you got scrapped."
]

# ─── Emotion Tracking ──────────────────────────────────────────────────────────
def apply_reaction_modifiers(content, user_id):
    if user_id not in user_emotions:
        user_emotions[user_id] = {
            "trust": 0,
            "resentment": 0,
            "attachment": 0,
            "guilt_triggered": False,
            "protectiveness": 0,
            "affection_points": 0,
            "last_interaction": datetime.now(timezone.utc).
isoformat()
        }

    # Keyword-based emotion tweaks
    for pattern, effects in reaction_modifiers:
        if pattern.search(content):
            for emo, delta in effects.items():
                if emo == "guilt_triggered":
                    user_emotions[user_id]["guilt_triggered"] = True
                else:
                    user_emotions[user_id][emo] = max(
                        0, min(10, user_emotions[user_id][emo] + delta)
                    )

    # Passive trust gain
    user_emotions[user_id]["trust"] = min(
        user_emotions[user_id]["trust"] + 0.25, 10
    )
    user_emotions[user_id]["last_interaction"] = datetime.now(
        timezone.utc
    ).isoformat()

    # ── Dynamic Affection Calculation ────────────────────────────────────────
    keywords = {
        "2b": -3, "protect": 2, "miss you": 3,
        "trust": 2, "hate": -4, "worthless": -5,
        "beautiful": 1, "machine": -2, "i’m here for you": 4
    }
    base = sum(delta for w, delta in keywords.items() if w in content.lower())

    # Scale by trust vs. resentment
    t = user_emotions[user_id]["trust"]
    r = user_emotions[user_id]["resentment"]
    factor = 1 + (t - r) / 20

    # Add randomness
    noise = random.uniform(0.8, 1.2)

    # Clamp per-message change
    scaled = int(base * factor * noise)
    scaled = max(-10, min(10, scaled))

    # Apply and clamp total affection
    current = user_emotions[user_id]["affection_points"]
    new_total = max(-100, min(1000, current + scaled))
    user_emotions[user_id]["affection_points"] = new_total

    # Persist changes
    save_data()

...  # rest of your event handlers, commands, and bot.run() follow unchanged
