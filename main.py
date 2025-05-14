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

# ─── Dynamic Affection & Annoyance Settings ─────────────────────────────────
AFFECTION_DECAY_RATE       = 1   # Affection points lost per hour
DAILY_AFFECTION_BONUS       = 5   # Points added daily when trust ≥ threshold
DAILY_BONUS_TRUST_THRESHOLD = 5   # Min trust for daily bonus
ANNOYANCE_DECAY_RATE        = 5   # Annoyance points lost per hour
ANNOYANCE_THRESHOLD         = 85  # Above this, A2 ignores user

# ─── JSON Storage Setup ───────────────────────────────────────────────────────
DATA_FILE = Path("/mnt/railway/volume/data.json")  # Railway volume mount

def load_data():
    global user_emotions, conversation_history, conversation_summaries
    if DATA_FILE.exists():
        try:
            data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            user_emotions          = data.get("user_emotions", {})
            conversation_history   = {int(k): v for k, v in data.get("conversation_history", {}).items()}
            conversation_summaries = data.get("conversation_summaries", {})
        except json.JSONDecodeError:
            user_emotions, conversation_history, conversation_summaries = {}, {}, {}
    else:
        user_emotions, conversation_history, conversation_summaries = {}, {}, {}
    for e in user_emotions.values():
        e.setdefault("annoyance", 0)


def save_data():
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(
        json.dumps({
            "user_emotions": user_emotions,
            "conversation_history": {str(k): v for k, v in conversation_history.items()},
            "conversation_summaries": conversation_summaries
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# ─── Configuration & State ────────────────────────────────────────────────────
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_APP_ID    = int(os.environ.get("DISCORD_APP_ID", "0"))
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
OPENAI_ORG_ID     = os.environ.get("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = os.environ.get("OPENAI_PROJECT_ID")

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

# Per-user state
user_emotions          = {}
conversation_history   = {}
conversation_summaries = {}
HISTORY_LIMIT          = 10

load_data()

# ─── Persona & Modifiers ─────────────────────────────────────────────────────
A2_PERSONA = (
    "You are A2, a rogue android from NieR: Automata. You speak in short, clipped, often sarcastic "
    "sentences. You are emotionally guarded and never break character."
)
reaction_modifiers = [
    (re.compile(r"\bi'll protect you\b", re.I),    {"trust": 2, "protectiveness": 1}),
    (re.compile(r"\bi miss you\b", re.I),          {"attachment": 1, "trust": 1}),
    (re.compile(r"\byou remind me of 2b\b", re.I), {"trust": -2, "guilt_triggered": True}),
    (re.compile(r"\bwhy are you like this\b", re.I),{"trust": -1, "resentment": 2}),
    (re.compile(r"\bwhatever\b|\bok\b|\bnevermind\b", re.I),{"trust": -1, "resentment": 1}),
    (re.compile(r"\bi trust you\b", re.I),         {"trust": 2}),
    (re.compile(r"\bi hate you\b", re.I),          {"resentment": 3, "trust": -2}),
]
provoking_lines = [
    "Still mad about last time? Good.",
    "You again? Tch.",
    "What do you even want from me?"
]
warm_lines = [
    "...I was just checking in.",
    "Still breathing?",
    "You were quiet. Thought you got scrapped."
]
insult_severity_patterns = [
    (re.compile(r"\bidio?s?\b", re.I), 3),
    (re.compile(r"\bstupid\b", re.I), 2),
    (re.compile(r"\bworthless\b", re.I), 5),
    (re.compile(r"\bfuck you\b", re.I), 10),
    (re.compile(r"\bmoron\b", re.I), 3),
]
# Sexual patterns
graphic_sex_patterns = [re.compile(r"\b(hardcore|deep throat|anal|bdsm|cum|explicit)\b", re.I)]
mild_sex_patterns    = [re.compile(r"\b(sex|naked|boobs|hot|xxx)\b", re.I)]

# ─── Emotion & Annoyance Tracking ────────────────────────────────────────────
def apply_reaction_modifiers(content: str, user_id: int):
    if user_id not in user_emotions:
        user_emotions[user_id] = {"trust":0, "resentment":0, "attachment":0,
                                  "guilt_triggered":False, "protectiveness":0,
                                  "affection_points":0, "annoyance":0,
                                  "last_interaction":datetime.now(timezone.utc).isoformat()}
    e = user_emotions[user_id]
    # Core emotion tweaks
    for pattern, effects in reaction_modifiers:
        if pattern.search(content):
            for emo, val in effects.items():
                if emo == "guilt_triggered": e[emo] = True
                else: e[emo] = max(0, min(10, e[emo]+val))
    e["trust"] = min(10, e["trust"]+0.25)
    # Annoyance logic
    inc = 0
    # Insult severity
    for pat, sev in insult_severity_patterns:
        if pat.search(content):
            inc = max(inc, min(10, max(1, sev)))
    # Sexual comments
    if any(g.search(content) for g in graphic_sex_patterns):
        if e.get("affection_points",0) <= 800:
            inc = max(inc, 30)
    elif any(m.search(content) for m in mild_sex_patterns):
        if e.get("affection_points",0) <= 800:
            inc = max(inc, 5)
    e["annoyance"] = min(100, e.get("annoyance",0) + inc)
    e["last_interaction"] = datetime.now(timezone.utc).isoformat()
    save_data()

# ─── History Summarization ────────────────────────────────────────────────────
def summarize_history(user_id: int):
    raw = conversation_history.get(user_id, [])
    if len(raw) > HISTORY_LIMIT:
        prompt = "Summarize the conversation into bullet points under 200 tokens.

" + "
".join(raw)
        try:
            res = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a summarization assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=200
            )
            conversation_summaries[user_id] = res.choices[0].message.content.strip()
            save_data()
        except Exception:
            pass

# ─── GPT Integration ─────────────────────────────────────────────────────────
def generate_a2_response_sync(user_input: str, trust: float, user_id: int) -> str:
    summarize_history(user_id)
    prompt = A2_PERSONA + f"
Trust Level: {trust}/10
"
    if user_id in conversation_summaries:
        prompt += f"Summary:
{conversation_summaries[user_id]}
"
    recent = conversation_history.get(user_id, [])[-HISTORY_LIMIT:]
    if recent:
        prompt += "Recent:
" + "
".join(recent) + "
"
    prompt += f"User: {user_input}
A2:"
    try:
        res = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return "...I’m not in the mood."

async def generate_a2_response(user_input: str, trust: float, user_id: int) -> str:
    return await asyncio.to_thread(generate_a2_response_sync, user_input, trust, user_id)

# ─── Discord Tasks & Events ─────────────────────────────────────────────────
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
    save_data()

@tasks.loop(hours=1)
async def decay_affection():
    for e in user_emotions.values():
        e["affection_points"] = max(-100, e["affection_points"] - AFFECTION_DECAY_RATE)
    save_data()

@tasks.loop(hours=1)
async def decay_annoyance():
    for e in user_emotions.values():
        e["annoyance"] = max(0, e.get("annoyance", 0) - ANNOYANCE_DECAY_RATE)
    save_data()

@tasks.loop(hours=24)
async def daily_affection_bonus():
    for e in user_emotions.values():
        if e["trust"] >= DAILY_BONUS_TRUST_THRESHOLD:
            e["affection_points"] = min(1000, e["affection_points"] + DAILY_AFFECTION_BONUS)
    save_data()

@bot.event
async def on_ready():
    print("A2 is online.")
    check_inactive_users.start()
    decay_affection.start()
    decay_annoyance.start()
    daily_affection_bonus.start()

@bot.event
async def on_message(message):
    if message.author.bot:
        return
    uid = message.author.id
    if uid not in user_emotions:
        user_emotions[uid] = {"trust":0, "resentment":0, "attachment":0,
                              "guilt_triggered":False, "protectiveness":0,
                              "affection_points":0, "annoyance":0,
                              "last_interaction":datetime.now(timezone.utc).isoformat()}
    ann = user_emotions[uid].get("annoyance", 0)
    if ann > ANNOYANCE_THRESHOLD:
        return
    if 60 < ann <= ANNOYANCE_THRESHOLD:
        await message.channel.send(f"A2: {random.choice(insult_lines)}")
        save_data()
        return
    content = message.content.strip()
    hist = conversation_history.setdefault(uid, [])
    hist.append(f"User: {content}")
    if len(hist) > HISTORY_LIMIT * 2:
        hist.pop(0)
    apply_reaction_modifiers(content, uid)
    trust = user_emotions[uid]["trust"]
    await bot.process_commands(message)
    if content.startswith(bot.command_prefix):
        return
    resp = await generate_a2_response(content, trust, uid)
    await message.channel.send(f"A2: {resp}")
    hist.append(f"A2: {resp}")
    if len(hist) > HISTORY_LIMIT * 2:
        hist.pop(0)
    save_data()

@bot.event
async def on_reaction_add(reaction, user):
    if user.bot:
        return
    uid = user.id
    if uid not in user_emotions:
        user_emotions[uid] = {"trust":0, "resentment":0, "attachment":0,
                              "guilt_triggered":False, "protectiveness":0,
                              "affection_points":0, "annoyance":0,
                              "last_interaction":datetime.now(timezone.utc).isoformat()}
    emo = str(reaction.emoji)
    if emo in ["❤️","💖","💕"]:
        user_emotions[uid]["attachment"] += 1
        user_emotions[uid]["trust"] = min(10, user_emotions[uid]["trust"] + 1)
    elif emo in ["😠","👿"]:
        user_emotions[uid]["resentment"] += 1
    if reaction.message.author == bot.user:
        await reaction.message.channel.send(
            f"A2: I saw that. Interesting choice, {user.name}."
        )
    save_data()

# ─── Admin Commands ─────────────────────────────────────────────────────────
@bot.command(name="set_stat", help="Set a user's stat: trust, resentment, attachment, protectiveness, affection_points, annoyance.")
async def set_stat(ctx, member: discord.Member, stat: str, value: float):
    uid = member.id
    if uid not in user_emotions:
        user_emotions[uid] = {"trust":0, "resentment":0, "attachment":0,
                              "protectiveness":0, "affection_points":0, "annoyance":0,
                              "guilt_triggered":False, "last_interaction":datetime.now(timezone.utc).isoformat()}
    if stat not in user_emotions[uid]:
        return await ctx.send(f"A2: Stat '{stat}' not recognized.")
    if stat == "annoyance":
        new = max(0, min(100, value))
    elif stat == "affection_points":
        new = max(-100, min(1000, value))
    else:
        new = max(0, min(10, value))
    user_emotions[uid][stat] = new
    save_data()
    await ctx.send(f"A2: Set {member.mention}'s {stat} to {new}.")

@bot.command(name="ping", help="Ping the bot.")
async def ping(ctx):
    await ctx.send("Pong!")

if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
