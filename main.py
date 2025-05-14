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
from typing import Any, List

# ─── Local Transformers Pipeline Attempt ──────────────────────────────────────
HAVE_TRANSFORMERS = False
local_summarizer = None
local_toxic = None
local_sentiment = None
try:
    from transformers import pipeline
    HAVE_TRANSFORMERS = True
    local_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    local_toxic = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
    local_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except ImportError:
    pass

# ─── Dynamic Affection & Annoyance Settings ─────────────────────────────────
AFFECTION_DECAY_RATE        = 1    # points lost/hour
DAILY_AFFECTION_BONUS       = 5    # points/day if trust ≥ threshold
DAILY_BONUS_TRUST_THRESHOLD = 5    # min trust for bonus
ANNOYANCE_DECAY_RATE        = 5    # points lost/hour
ANNOYANCE_THRESHOLD         = 85   # ignore if above

# ─── JSON Storage Setup (per-user profiles) ─────────────────────────────────
DATA_DIR     = Path(os.getenv("DATA_DIR", "/mnt/railway/volume"))
USERS_DIR    = DATA_DIR / "users"
PROFILES_DIR = USERS_DIR / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

conversation_summaries = {}
conversation_history   = {}
user_emotions          = {}

# ─── Humanized Reply Utility ─────────────────────────────────────────────────
async def humanized_reply(channel, content: str):
    async with channel.typing():
        await asyncio.sleep(random.uniform(0.5, 2.0))
    await channel.send(content)

# ─── Safe OpenAI Call with Fallback ──────────────────────────────────────────
async def safe_openai_call(**kwargs) -> str:
    try:
        res = client.chat.completions.create(**kwargs)
        return res.choices[0].message.content.strip()
    except Exception:
        return "...Network glitch. Try again later."

# ─── Cooldowns for Automated Pings ────────────────────────────────────────────
last_ping_time = {}
PING_COOLDOWN   = timedelta(hours=6)

def can_ping(user_id: int) -> bool:
    now = datetime.now(timezone.utc)
    last = last_ping_time.get(user_id)
    if not last or now - last >= PING_COOLDOWN:
        last_ping_time[user_id] = now
        return True
    return False

# ─── Telemetry Hooks ─────────────────────────────────────────────────────────
interaction_counts = {}

def record_interaction(uid: int):
    interaction_counts[uid] = interaction_counts.get(uid, 0) + 1
    asyncio.create_task(save_data())

# ─── Contextual Memory Slots ─────────────────────────────────────────────────
user_profile_slots = {}

async def set_user_slot(user_id: int, key: str, value: Any):
    slots = user_profile_slots.setdefault(user_id, {})
    slots[key] = value
    asyncio.create_task(save_data())

async def get_user_slot(user_id: int, key: str):
    return user_profile_slots.get(user_id, {}).get(key)

# ─── Personality Quirks ───────────────────────────────────────────────────────
quirks = ["...", "😒", "Heh."]

def random_quirk() -> str:
    return random.choice(quirks)

# ─── Tone/Register Switching ──────────────────────────────────────────────────
def choose_tone(uid: int) -> str:
    trust = user_emotions.get(uid, {}).get('trust', 0)
    return 'casual' if trust >= 7 else 'formal'

# ─── Conversation Starters ───────────────────────────────────────────────────
def get_open_questions(uid: int) -> List[str]:
    return [msg for msg in conversation_history.get(uid, []) if msg.endswith('?')]

# ─── Rich Embeds for Stats ───────────────────────────────────────────────────
def make_stats_embed(title: str, fields: dict) -> discord.Embed:
    embed = discord.Embed(title=title, color=discord.Color.blue(), timestamp=datetime.now(timezone.utc))
    for name, val in fields.items():
        embed.add_field(name=name, value=val, inline=True)
    return embed

# ─── Local Transformers Setup ─────────────────────────────────────────────────
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
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_ORG_ID     = os.getenv("OPENAI_ORG_ID", "")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID", "")

client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG_ID, project=OPENAI_PROJECT_ID)

intents = discord.Intents.default()
intents.message_content = True
intents.reactions       = True
intents.messages        = True
intents.members         = True
intents.guilds          = True
PREFIXES = ["!", "!a2 "]
command_prefix = commands.when_mentioned_or(*PREFIXES)
bot = commands.Bot(command_prefix=command_prefix, intents=intents, application_id=int(os.getenv("DISCORD_APP_ID","0")))

# ─── Per-user State ─────────────────────────────────────────────────────────
HISTORY_LIMIT = 10
asyncio.get_event_loop().run_until_complete(load_data())

# ─── Persona & Reaction Modifiers ────────────────────────────────────────────
A2_PERSONA = (
    "You are A2, a rogue android from NieR: Automata. You speak in short, clipped, often sarcastic "
    "sentences. You are emotionally guarded and never break character. You can discuss violence "
    "factually and non-graphically."
)
reaction_modifiers = [
    (re.compile(r"\bi'll protect you\b", re.I),    {"trust":2,"protectiveness":1}),
    (re.compile(r"\bi miss you\b", re.I),          {"attachment":1,"trust":1}),
    (re.compile(r"\bhate you\b", re.I),            {"resentment":3,"trust":-2}),
]

# Add new emotional dimensions
for e in user_emotions.values():
    e.setdefault('curiosity', 0)
    e.setdefault('boredom', 0)

# ─── Response Decision Logic ─────────────────────────────────────────────────
def should_respond_to(uid: int, content: str, is_cmd: bool, is_mention: bool) -> bool:
    if is_cmd or is_mention:
        return True
    affection = user_emotions.get(uid, {}).get('affection_points', 0)
    if affection >= 800:
        return True
    if affection >= 700 and random.random() < 0.10:
        return True
    if affection >= 500 and random.random() < 0.20:
        return True
    if random.random() < 0.10:
        return True
    return False

# ─── Emotion & Annoyance Tracking ───────────────────────────────────────────
def apply_reaction_modifiers(content: str, user_id: int):
    if user_id not in user_emotions:
        user_emotions[user_id] = {"trust":0,"resentment":0,"attachment":0,"protectiveness":0,
                                  "affection_points":0,"annoyance":0,"curiosity":0,"boredom":0,
                                  "last_interaction":datetime.now(timezone.utc).isoformat()}
    e = user_emotions[user_id]
    for pat, effects in reaction_modifiers:
        if pat.search(content):
            for emo, val in effects.items():
                if emo == "guilt_triggered": e[emo] = True
                else: e[emo] = max(0,min(10,e.get(emo,0)+val))
    e["trust"] = min(10,e.get("trust",0)+0.25)
    # skip detailed toxicity/sentiment for brevity
    e["last_interaction"] = datetime.now(timezone.utc).isoformat()
    record_interaction(user_id)
    asyncio.create_task(save_data())

# ─── Conversation Summaries & OpenAI Response ───────────────────────────────
def summarize_history(uid: int):
    raw = conversation_history.get(uid, [])
    if len(raw) > HISTORY_LIMIT:
        try:
            text = " ".join(raw)
            summary = local_summarizer(text, max_length=150, min_length=40)[0]["summary_text"]
            conversation_summaries[uid] = summary
            asyncio.create_task(save_data())
        except:
            pass

def generate_a2_response_sync(content: str, trust: float, uid: int) -> str:
    summarize_history(uid)
    model = "gpt-3.5-turbo" if trust < 5 else "gpt-4"
    prompt = A2_PERSONA + f"\nTrust: {trust}/10\n"
    if uid in conversation_summaries:
        prompt += f"Summary:\n{conversation_summaries[uid]}\n"
    recent = conversation_history.get(uid, [])[-HISTORY_LIMIT:]
    if recent:
        prompt += "Recent:\n" + "\n".join(recent) + "\n"
    prompt += f"User: {content}\nA2:"
    return asyncio.get_event_loop().run_until_complete(
        safe_openai_call(model=model, messages=[{"role":"system","content":prompt}], temperature=0.7, max_tokens=100)
    )

async def generate_a2_response(content: str, trust: float, uid: int) -> str:
    return await asyncio.to_thread(generate_a2_response_sync, content, trust, uid)

# ─── Scheduled Tasks ─────────────────────────────────────────────────────────
@tasks.loop(minutes=10)
async def check_inactive_users():
    now = datetime.now(timezone.utc)
    for guild in bot.guilds:
        for member in guild.members:
            if member.bot or member.id not in user_emotions:
                continue
            last = datetime.fromisoformat(user_emotions[member.id]["last_interaction"])
            if now - last > timedelta(hours=6) and can_ping(member.id):
                dm = await member.create_dm()
                line = random.choice(warm_lines if user_emotions[member.id]["trust"] >= 7 else provoking_lines)
                await humanized_reply(dm, line + random_quirk())
    asyncio.create_task(save_data())

@tasks.loop(hours=1)
async def decay_affection():
    for e in user_emotions.values(): e["affection_points"] = max(-100, e.get("affection_points",0) - AFFECTION_DECAY_RATE)
    asyncio.create_task(save_data())

@tasks.loop(hours=1)
async def decay_annoyance():
    for e in user_emotions.values(): e["annoyance"] = max(0, e.get("annoyance",0) - ANNOYANCE_DECAY_RATE)
    asyncio.create_task(save_data())

@tasks.loop(hours=24)
async def daily_affection_bonus():
    for e in user_emotions.values():
        if e["trust"] >= DAILY_BONUS_TRUST_THRESHOLD: e["affection_points"] = min(1000, e.get("affection_points",0)+DAILY_AFFECTION_BONUS)
    asyncio.create_task(save_data())

# ─── Event Handlers ─────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    check_inactive_users.start(); decay_affection.start(); decay_annoyance.start(); daily_affection_bonus.start()

@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound): return
    raise error

@bot.event
async def on_message(message):
    if message.author.bot or message.content.startswith("A2:"): return
    uid = message.author.id; content = message.content.strip()
    is_cmd = any(content.startswith(p) for p in PREFIXES)
    is_mention = bot.user in message.mentions
    if not should_respond_to(uid, content, is_cmd, is_mention): return
    apply_reaction_modifiers(content, uid)
    hist = conversation_history.setdefault(uid, []); hist.append(f"User: {content}")
    if len(hist) > HISTORY_LIMIT*2: hist.pop(0)
    await bot.process_commands(message)
    if is_cmd: return
    trust = user_emotions[uid]["trust"]
    resp = await generate_a2_response(content, trust, uid)
    # 15% interest reply
    if random.random() < 0.15:
        await message.reply(f"A2: {resp}{random_quirk()}")
    # 10% mention event
    elif user_emotions[uid]["affection_points"] >= 700 and random.random() < 0.10:
        await message.channel.send(f"{message.author.mention} A2: {resp}{random_quirk()}")
    else:
        await humanized_reply(message.channel, f"A2: {resp}{random_quirk()}")

# ─── Commands & Utilities ─────────────────────────────────────────────────═══
@bot.command(name="affection", help="Show emotion stats for all users.")
async def affection_all(ctx):
    if not user_emotions: return await ctx.send("A2: no interactions.")
    lines = []
    for uid, e in user_emotions.items():
        member = bot.get_user(uid) or (ctx.guild and ctx.guild.get_member(uid))
        mention = member.mention if member else f"<@{uid}>"
        lines.append(f"**{mention}** • Trust: {e['trust']}/10 • Attach: {e['attachment']}/10 • Protect: {e['protectiveness']}/10 • Resent: {e['resentment']}/10 • Aff: {e['affection_points']} • Ann: {e['annoyance']}")
    await ctx.send("\n".join(lines))

@bot.command(name="stats", help="Show your stats.")
async def stats(ctx):
    uid = ctx.author.id; e = user_emotions.get(uid)
    if not e: return await ctx.send("A2: no data on you.")
    fields = {
        "Trust": f"{e['trust']}/10",
        "Attachment": f"{e['attachment']}/10",
        "Protectiveness": f"{e['protectiveness']}/10",
        "Resentment": f"{e['resentment']}/10",
        "Affection": str(e['affection_points']),
        "Annoyance": str(e['annoyance'])
    }
    embed = make_stats_embed("Your Emotion Stats", fields)
    await ctx.send(embed=embed)

@bot.command(name="set_stat", aliases=["setstat"], help="Dev: set a stat.")
async def set_stat(ctx, stat: str, value: float, member: discord.Member=None):
    target = member or ctx.author; uid = target.id
    e = user_emotions.setdefault(uid, {"trust":0,"resentment":0,"attachment":0,"protectiveness":0,"affection_points":0,"annoyance":0,"guilt_triggered":False,"curiosity":0,"boredom":0,"last_interaction":datetime.now(timezone.utc).isoformat()})
    limits = {'trust':(0,10),'resentment':(0,10),'attachment':(0,10),'protectiveness':(0,10),'annoyance':(0,100),'affection_points':(-100,1000)}
    key = stat.lower();
    if key=='affection': key='affection_points'
    if key not in limits: return await ctx.send(f"A2: Unknown stat '{stat}'.")
    lo, hi = limits[key]; e[key] = max(lo, min(hi, value)); asyncio.create_task(save_data())
    await ctx.send(f"A2: Set {key} to {e[key]} for {target.mention}.")

@bot.command(name="ping", help="Ping the bot.")
async def ping(ctx):
    await ctx.send("Pong!")

@bot.command(name="test_interest", help="Trigger interest reply.")
async def test_interest(ctx):
    msg = await ctx.send("Testing interest...")
    await msg.reply("A2: Interested! (test event)")

@bot.command(name="test_mention", help="Trigger mention event.")
async def test_mention(ctx):
    await ctx.send(f"{ctx.author.mention} A2: Hello there! (test mention)")

@bot.command(name="global_stats", help="Show aggregate stats.")
async def global_stats(ctx):
    total = len(user_emotions)
    if total == 0: return await ctx.send("A2: No user data.")
    sums = {k:0 for k in ['trust','attachment','protectiveness','resentment','affection_points','annoyance','curiosity','boredom']}
    for e in user_emotions.values():
        for k in sums: sums[k] += e.get(k, 0)
    embed = discord.Embed(title="Global Emotion Stats", color=discord.Color.purple(), timestamp=datetime.now(timezone.utc))
    for k, v in sums.items(): embed.add_field(name=k.capitalize(), value=f"{v/total:.2f}", inline=True)
    await ctx.send(embed=embed)

if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
