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

# ─── Reaction Modifier Handler ───
def apply_reaction_modifiers(content: str, user_id: int):
    if user_id not in user_emotions:
        user_emotions[user_id] = {
            "trust": 0, "resentment": 0, "attachment": 0,
            "guilt_triggered": False, "protectiveness": 0,
            "affection_points": 0, "annoyance": 0,
            "last_interaction": datetime.now(timezone.utc).isoformat()
        }
    e = user_emotions[user_id]
    for pat, effects in reaction_modifiers:
        if pat.search(content):
            for emo, val in effects.items():
                if emo == "guilt_triggered":
                    e[emo] = True
                else:
                    e[emo] = max(0, min(10, e[emo] + val))
    e["trust"] = min(10, e["trust"] + 0.25)
    e["last_interaction"] = datetime.now(timezone.utc).isoformat()
    asyncio.create_task(save_data())
# ─── Tasks ───
@tasks.loop(minutes=10)
async def check_inactive_users():
    now = datetime.now(timezone.utc)
    for guild in bot.guilds:
        for member in guild.members:
            if member.bot or member.id not in user_emotions:
                continue
            last = datetime.fromisoformat(user_emotions[member.id]["last_interaction"])
            if now - last > timedelta(hours=6):
                dm = await member.create_dm()
                if user_emotions[member.id]["trust"] >= 7:
                    await dm.send(random.choice(warm_lines))
                else:
                    await dm.send(random.choice(provoking_lines))
    await save_data()

@tasks.loop(hours=1)
async def decay_affection():
    for e in user_emotions.values():
        e["affection_points"] = max(-100, e["affection_points"] - AFFECTION_DECAY_RATE)
    await save_data()

@tasks.loop(hours=1)
async def decay_annoyance():
    for e in user_emotions.values():
        e["annoyance"] = max(0, e["annoyance"] - ANNOYANCE_DECAY_RATE)
    await save_data()

@tasks.loop(hours=24)
async def daily_affection_bonus():
    for e in user_emotions.values():
        if e["trust"] >= DAILY_BONUS_TRUST_THRESHOLD:
            e["affection_points"] = min(1000, e["affection_points"] + DAILY_AFFECTION_BONUS)
    await save_data()

@bot.event
async def on_ready():
    print("A2 is online.")
    await load_data()
    check_inactive_users.start()
    decay_affection.start()
    decay_annoyance.start()
    daily_affection_bonus.start()

@bot.event
async def on_message(message):
    if message.author.bot:
        return
    uid, content = message.author.id, message.content.strip()
    if uid not in user_emotions:
        user_emotions[uid] = {
            "trust": 0, "resentment": 0, "attachment": 0,
            "guilt_triggered": False, "protectiveness": 0,
            "affection_points": 0, "annoyance": 0,
            "last_interaction": datetime.now(timezone.utc).isoformat()
        }
    affection = user_emotions[uid]["affection_points"]
    is_command = any(content.startswith(p) for p in PREFIXES)
    is_mention = bot.user in message.mentions
    if affection < 100 and not (is_command or is_mention):
        return
    hist = conversation_history.setdefault(uid, [])
    hist.append(f"User: {content}")
    if len(hist) > HISTORY_LIMIT * 2:
        hist.pop(0)
    apply_reaction_modifiers(content, uid)
    trust = user_emotions[uid]["trust"]
    await bot.process_commands(message)
    if is_command:
        return
    resp = await generate_a2_response(content, trust, uid)
    await message.channel.send(f"A2: {resp}")
    hist.append(f"A2: {resp}")
    if len(hist) > HISTORY_LIMIT * 2:
        hist.pop(0)
    await save_data()

@bot.event
async def on_reaction_add(reaction, user):
    if user.bot:
        return
    uid = user.id
    if uid not in user_emotions:
        user_emotions[uid] = {
            "trust": 0, "resentment": 0, "attachment": 0,
            "guilt_triggered": False, "protectiveness": 0,
            "affection_points": 0, "annoyance": 0,
            "last_interaction": datetime.now(timezone.utc).isoformat()
        }
    emo = str(reaction.emoji)
    if emo in ["❤️", "💖", "💕"]:
        user_emotions[uid]["attachment"] += 1
        user_emotions[uid]["trust"] = min(10, user_emotions[uid]["trust"] + 1)
    elif emo in ["😠", "👿"]:
        user_emotions[uid]["resentment"] += 1
    if reaction.message.author == bot.user:
        await reaction.message.channel.send(f"A2: I saw that. Interesting choice, {user.name}.")
    await save_data()

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

@bot.command(name="set_stat", help="Dev: set a stat.")
async def set_stat(ctx, member: discord.Member, stat: str, value: float):
    uid = member.id
    e = user_emotions.setdefault(uid, {
        'trust':0, 'resentment':0, 'attachment':0,
        'protectiveness':0, 'affection_points':0,'annoyance':0,
        'guilt_triggered':False, 'last_interaction':datetime.now(timezone.utc).isoformat()
    })
    if stat not in e:
        return await ctx.send(f"A2: Unknown stat '{stat}'.")
    limits = {
        'trust': (0,10), 'resentment': (0,10), 'attachment': (0,10),
        'protectiveness': (0,10), 'annoyance': (0,100), 'affection_points': (-100,1000)
    }
    lo, hi = limits.get(stat, (0,10))
    new = max(lo, min(hi, value))
    e[stat] = new
    await save_data()
    await ctx.send(f"A2: Set {stat} to {new} for {member.mention}.")

@bot.command(name="ping", help="Ping the bot.")
async def ping(ctx):
    await ctx.send("Pong!")

@bot.command(name="test_decay", help="Run affection and annoyance decay immediately.")
async def test_decay(ctx):
    decay_affection.restart()
    decay_annoyance.restart()
    await ctx.send("A2: Decay tasks triggered.")

@bot.command(name="test_daily", help="Run daily affection bonus immediately.")
async def test_daily(ctx):
    daily_affection_bonus.restart()
    await ctx.send("A2: Daily affection bonus triggered.")

@bot.command(name="view_emotions", help="View raw emotion data for a user.")
async def view_emotions(ctx, member: discord.Member = None):
    target = member or ctx.author
    uid = target.id
    if uid not in user_emotions:
        return await ctx.send(f"A2: No data for {target.mention}.")
    e = user_emotions[uid]
    await ctx.send(f"Emotion data for {target.mention}: {json.dumps(e, indent=2)}")

if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
