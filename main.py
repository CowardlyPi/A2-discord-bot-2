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

# ─── Local Transformers Pipeline Attempt ──────────────────────────────────────
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

# ─── Dynamic Affection & Annoyance Settings ─────────────────────────────────
AFFECTION_DECAY_RATE       = 1   # points lost/hour
DAILY_AFFECTION_BONUS      = 5   # points/day if trust ≥ threshold
DAILY_BONUS_TRUST_THRESHOLD= 5   # min trust for bonus
ANNOYANCE_DECAY_RATE       = 5   # points lost/hour
ANNOYANCE_THRESHOLD        = 85  # ignore if above

# ─── JSON Storage Setup (per-user profiles) ─────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "/mnt/railway/volume"))
USERS_DIR = DATA_DIR / "users"
PROFILES_DIR = USERS_DIR / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

conversation_summaries = {}
conversation_history = {}
user_emotions = {}

async def load_user_profile(user_id: int):
    path = PROFILES_DIR / f"{user_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
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

# ─── Configuration & State ───────────────────────────────────────────────────
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_APP_ID    = int(os.environ.get("DISCORD_APP_ID", "0") or 0)
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
OPENAI_ORG_ID     = os.environ.get("OPENAI_ORG_ID", "")
OPENAI_PROJECT_ID = os.environ.get("OPENAI_PROJECT_ID", "")

client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG_ID, project=OPENAI_PROJECT_ID)

intents = discord.Intents.default()
intents.message_content = True
intents.reactions       = True
intents.messages        = True
intents.members         = True
intents.guilds          = True
PREFIXES = ["!", "!a2 "]
command_prefix = commands.when_mentioned_or(*PREFIXES)
bot = commands.Bot(command_prefix=command_prefix, intents=intents, application_id=DISCORD_APP_ID)

# ─── Per-user State & Utilities ─────────────────────────────────────────────
HISTORY_LIMIT          = 10
asyncio.get_event_loop().run_until_complete(load_data())

# ─── Persona & Modifiers ─────────────────────────────────────────────────────
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

# ─── Helper: Should Respond Logic ───────────────────────────────────────────
def should_respond_to(content: str, uid: int, is_cmd: bool, is_mention: bool) -> bool:
    affection = user_emotions.get(uid, {}).get('affection_points', 0)
    if is_cmd or is_mention:
        return True
    if affection >= 800:
        return True
    if affection >= 500:
        return random.random() < 0.2
    return False

# ─── Emotion & Annoyance Tracking ───────────────────────────────────────────
def apply_reaction_modifiers(content: str, user_id: int):
    if user_id not in user_emotions:
        user_emotions[user_id] = {
            "trust":0, "resentment":0, "attachment":0,
            "guilt_triggered":False, "protectiveness":0,
            "affection_points":0, "annoyance":0,
            "last_interaction":datetime.now(timezone.utc).isoformat()
        }
    e = user_emotions[user_id]
    for pat, effects in reaction_modifiers:
        if pat.search(content):
            for emo, val in effects.items():
                if emo == "guilt_triggered":
                    e[emo] = True
                else:
                    e[emo] = max(0, min(10, e.get(emo, 0) + val))
    e["trust"] = min(10, e.get("trust", 0) + 0.25)
    inc = 0
    if HAVE_TRANSFORMERS and local_toxic:
        try:
            scores = local_toxic(content)[0]
            for item in scores:
                if item["label"].lower() in ("insult","toxicity"):
                    sev = int(item["score"] * 10)
                    inc = max(inc, min(10, max(1, sev)))
        except Exception:
            pass
    else:
        for pat, effects in reaction_modifiers:
            if pat.search(content):
                inc = max(inc, 1)
    e["annoyance"] = min(100, e.get("annoyance", 0) + inc)
    if HAVE_TRANSFORMERS and local_sentiment:
        try:
            s = local_sentiment(content)[0]
            delta = int((s["score"] * (1 if s["label"] == "POSITIVE" else -1)) * 5)
        except Exception:
            delta = 0
    else:
        delta = sum(1 for w in ["miss you","support","love"] if w in content.lower())
    factor = 1 + (e.get("trust", 0) - e.get("resentment", 0)) / 20
    e["affection_points"] = max(-100, min(1000, e.get("affection_points", 0) + int(delta * factor)))
    e["last_interaction"] = datetime.now(timezone.utc).isoformat()
    asyncio.create_task(save_data())

# ─── Summarization ───────────────────────────────────────────────────────────
def summarize_history(user_id: int):
    raw = conversation_history.get(user_id, [])
    if len(raw) > HISTORY_LIMIT:
        if HAVE_TRANSFORMERS and local_summarizer:
            try:
                text = " ".join(raw)
                summary = local_summarizer(text, max_length=150, min_length=40)[0]["summary_text"]
                conversation_summaries[user_id] = summary
                asyncio.create_task(save_data())
                return
            except Exception:
                pass
        prompt = "Summarize into bullet points under 200 tokens:\n" + "\n".join(raw)
        try:
            res = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":prompt}],
                temperature=0.5,
                max_tokens=200
            )
            conversation_summaries[user_id] = res.choices[0].message.content.strip()
            asyncio.create_task(save_data())
        except Exception:
            pass

# ─── A2 Response ─────────────────────────────────────────────────────────────
def generate_a2_response_sync(user_input: str, trust: float, user_id: int) -> str:
    summarize_history(user_id)
    model_name = "gpt-3.5-turbo" if trust < 5 else "gpt-4"
    prompt = A2_PERSONA + f"\nTrust: {trust}/10\n"
    if user_id in conversation_summaries:
        prompt += f"Summary:\n{conversation_summaries[user_id]}\n"
    recent = conversation_history.get(user_id, [])[-HISTORY_LIMIT:]
    if recent:
        prompt += "Recent:\n" + "\n".join(recent) + "\n"
    prompt += f"User: {user_input}\nA2:"
    try:
        res = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"system","content":prompt}],
            temperature=0.7,
            max_tokens=100
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return "...I’m not in the mood."

async def generate_a2_response(user_input: str, trust: float, user_id: int) -> str:
    return await asyncio.to_thread(generate_a2_response_sync, user_input, trust, user_id)

# ─── Tasks ───────────────────────────────────────────────────────────────────
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
    asyncio.create_task(save_data())

@tasks.loop(hours=1)
async def decay_affection():
    for e in user_emotions.values():
        e["affection_points"] = max(-100, e.get("affection_points", 0) - AFFECTION_DECAY_RATE)
    asyncio.create_task(save_data())

@tasks.loop(hours=1)
async def decay_annoyance():
    for e in user_emotions.values():
        e["annoyance"] = max(0, e.get("annoyance", 0) - ANNOYANCE_DECAY_RATE)
    asyncio.create_task(save_data())

@tasks.loop(hours=24)
async def daily_affection_bonus():
    for e in user_emotions.values():
        if e.get("trust", 0) >= DAILY_BONUS_TRUST_THRESHOLD:
            e["affection_points"] = min(1000, e.get("affection_points", 0) + DAILY_AFFECTION_BONUS)
    asyncio.create_task(save_data())

# ─── Events ─────────────────────────────────────────────────────────────────
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
    content = message.content.strip()
    if uid not in user_emotions:
        user_emotions[uid] = {
            "trust":0, "resentment":0, "attachment":0,
            "guilt_triggered":False, "protectiveness":0,
            "affection_points":0, "annoyance":0,
            "last_interaction":datetime.now(timezone.utc).isoformat()
        }
    is_cmd = any(content.startswith(p) for p in PREFIXES)
    is_mention = bot.user in message.mentions
    if not should_respond_to(content, uid, is_cmd, is_mention):
        return
    hist = conversation_history.setdefault(uid, [])
    hist.append(f"User: {content}")
    if len(hist) > HISTORY_LIMIT * 2:
        hist.pop(0)
    apply_reaction_modifiers(content, uid)
    await bot.process_commands(message)
    if is_cmd:
        return
    trust = user_emotions[uid]["trust"]
    resp = await generate_a2_response(content, trust, uid)
    await message.channel.send(f"A2: {resp}")
    hist.append(f"A2: {resp}")
    if len(hist) > HISTORY_LIMIT * 2:
        hist.pop(0)
    asyncio.create_task(save_data())

@bot.event
async def on_reaction_add(reaction, user):
    if user.bot:
        return
    uid = user.id
    if uid not in user_emotions:
        user_emotions[uid] = {
            "trust":0, "resentment":0, "attachment":0,
            "guilt_triggered":False, "protectiveness":0,
            "affection_points":0, "annoyance":0,
            "last_interaction":datetime.now(timezone.utc).isoformat()
        }
    emo = str(reaction.emoji)
    if emo in ["❤️","💖","💕"]:
        user_emotions[uid]["attachment"] += 1
        user_emotions[uid]["trust"] = min(10, user_emotions[uid]["trust"] + 1)
    elif emo in ["😠","👿"]:
        user_emotions[uid]["resentment"] += 1
    if reaction.message.author == bot.user:
        await reaction.message.channel.send(f"A2: I saw that. Interesting choice, {user.name}.")
    asyncio.create_task(save_data())

# ─── Commands ───────────────────────────────────────────────────────────────
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
    embed = discord.Embed(
        title="Your Emotion Stats",
        color=discord.Color.blue(),
        timestamp=datetime.now(timezone.utc)
    )
    embed.add_field(name="Trust", value=f"{e.get('trust',0)}/10", inline=True)
    embed.add_field(name="Attachment", value=f"{e.get('attachment',0)}/10", inline=True)
    embed.add_field(name="Protectiveness", value=f"{e.get('protectiveness',0)}/10", inline=True)
    embed.add_field(name="Resentment", value=f"{e.get('resentment',0)}/10", inline=True)
    embed.add_field(name="Affection", value=str(e.get('affection_points',0)), inline=True)
    embed.add_field(name="Annoyance", value=str(e.get('annoyance',0)), inline=True)
    embed.set_footer(text="A2 Bot")
    await ctx.send(embed=embed)

@bot.command(name="set_stat", aliases=["setstat"], help="Dev: set a stat for a user or yourself.")
async def set_stat(ctx, stat: str, value: float, member: discord.Member = None):
    target = member or ctx.author
    uid = target.id
    e = user_emotions.setdefault(uid, {
        "trust":0, "resentment":0, "attachment":0,
        "protectiveness":0, "affection_points":0, "annoyance":0,
        "guilt_triggered":False, "last_interaction":datetime.now(timezone.utc).isoformat()
    })
    limits = {
        'trust': (0,10), 'resentment': (0,10), 'attachment': (0,10),
        'protectiveness': (0,10), 'annoyance': (0,100), 'affection_points': (-100,1000)
    }
    key = stat.lower()
    if key == 'affection': key = 'affection_points'
    if key not in limits:
        return await ctx.send(f"A2: Unknown stat '{stat}'. Valid stats: {', '.join(limits.keys())}.")
    lo, hi = limits[key]
    e[key] = max(lo, min(hi, value))
    asyncio.create_task(save_data())
    await ctx.send(f"A2: Set {key} to {e[key]} for {target.mention}.")

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
    await ctx.send(f"Emotion data for {target.mention}: {json.dumps(user_emotions[uid], indent=2)}")

if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
