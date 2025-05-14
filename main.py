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

# ─── Local Transformers Pipeline Attempt ───
HAVE_TRANSFORMERS = False
local_summarizer = None
local_toxic = None
local_sentiment = None
try:
    from transformers import pipeline
    HAVE_TRANSFORMERS = True
    local_summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )
    local_toxic = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        top_k=None
    )
    local_sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
except ImportError:
    pass

# ─── Dynamic Affection & Annoyance Settings ───
AFFECTION_DECAY_RATE       = 1
DAILY_AFFECTION_BONUS      = 5
DAILY_BONUS_TRUST_THRESHOLD= 5
ANNOYANCE_DECAY_RATE       = 5
ANNOYANCE_THRESHOLD        = 85

# ─── JSON Storage Setup (per-user profiles) ───
DATA_DIR = Path(os.getenv("DATA_DIR", "/mnt/railway/volume"))
USERS_DIR = DATA_DIR / "users"
PROFILES_DIR = USERS_DIR / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "legacy_data.json"

conversation_summaries = {}
conversation_history = {}
user_emotions = {}

async def load_user_profile(user_id: int):
    path = PROFILES_DIR / f"{user_id}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data
        except json.JSONDecodeError:
            return {}
    return {}

async def save_user_profile(user_id: int):
    path = PROFILES_DIR / f"{user_id}.json"
    profile = user_emotions.get(user_id, {})
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

async def load_data():
    global user_emotions
    user_emotions = {}
    for file in PROFILES_DIR.glob("*.json"):
        uid = int(file.stem)
        user_emotions[uid] = await load_user_profile(uid)

async def save_data():
    for uid in list(user_emotions.keys()):
        await save_user_profile(uid)

# ─── Configuration & State ───
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_APP_ID    = int(os.environ.get("DISCORD_APP_ID", "0"))
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
OPENAI_ORG_ID     = os.environ.get("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = os.environ.get("OPENAI_PROJECT_ID")

client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG_ID, project=OPENAI_PROJECT_ID)

intents = discord.Intents.default()
intents.message_content = True
intents.reactions       = True
intents.messages        = True
intents.members         = True
intents.guilds          = True
PREFIXES = ["!", "!a2 "]
bot = commands.Bot(command_prefix=PREFIXES, intents=intents, application_id=DISCORD_APP_ID)

HISTORY_LIMIT = 10

# ─── Persona & Reaction Modifiers ───
A2_PERSONA = (
    "You are A2, a rogue android from NieR: Automata. You speak in short, clipped, often sarcastic "
    "sentences. You are emotionally guarded and never break character."
)
reaction_modifiers = [
    (re.compile(r"\bi'll protect you\b", re.I),    {"trust":2,"protectiveness":1}),
    (re.compile(r"\bi miss you\b", re.I),          {"attachment":1,"trust":1}),
    (re.compile(r"\bhate you\b", re.I),            {"resentment":3,"trust":-2}),
]
provoking_lines = ["Still mad? Good.", "You again? Tch.", "What?"]
warm_lines      = ["...Checking in.", "Still breathing?", "Thought you got scrapped."]

# ─── Reaction Modifier Handler ───
def apply_reaction_modifiers(content: str, user_id: int):
    if user_id not in user_emotions:
        user_emotions[user_id] = {
            "trust":0,"resentment":0,"attachment":0,
            "guilt_triggered":False,"protectiveness":0,
            "affection_points":0,"annoyance":0,
            "last_interaction":datetime.now(timezone.utc).isoformat()
        }
    e = user_emotions[user_id]
    for pat, effects in reaction_modifiers:
        if pat.search(content):
            for emo,val in effects.items():
                if emo == "guilt_triggered": e[emo] = True
                else: e[emo] = max(0,min(10,e[emo]+val))
    e["trust"] = min(10, e["trust"]+0.25)
    e["last_interaction"] = datetime.now(timezone.utc).isoformat()
    asyncio.create_task(save_data())

@bot.event
async def on_ready():
    print("A2 is online.")
    await load_data()
    check_inactive_users.start()
    decay_affection.start()
    decay_annoyance.start()
    daily_affection_bonus.start()

@bot.command(name="affection", help="Show emotion stats for all users.")
async def affection_all(ctx):
    if not user_emotions:
        return await ctx.send("A2: no interactions.")
    lines = []
    for uid, e in user_emotions.items():
        member = bot.get_user(uid) or (ctx.guild and ctx.guild.get_member(uid))
        mention = member.mention if member else f"<@{uid}>"
        lines.append(
            f"**{mention}** • Trust: {e.get('trust',0)}/10 • Attachment: {e.get('attachment',0)}/10"
            f" • Protectiveness: {e.get('protectiveness',0)}/10 • Resentment: {e.get('resentment',0)}/10"
            f" • Affection: {e.get('affection_points',0)} • Annoyance: {e.get('annoyance',0)}"
        )
    await ctx.send("\n".join(lines))

@bot.command(name="stats", help="Show your stats.")
async def stats(ctx):
    uid = ctx.author.id
    e = user_emotions.get(uid)
    if not e:
        return await ctx.send("A2: no data on you.")
    report = (
        f"Trust: {e.get('trust',0)}/10\n"
        f"Attachment: {e.get('attachment',0)}/10\n"
        f"Protectiveness: {e.get('protectiveness',0)}/10\n"
        f"Resentment: {e.get('resentment',0)}/10\n"
        f"Affection: {e.get('affection_points',0)}\n"
        f"Annoyance: {e.get('annoyance',0)}"
    )
    await ctx.send(f"A2: \n{report}")

if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
