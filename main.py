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
from collections import deque, defaultdict

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
AFFECTION_DECAY_RATE        = 1   # points lost/hour
DAILY_AFFECTION_BONUS       = 5   # points/day if trust ≥ threshold
DAILY_BONUS_TRUST_THRESHOLD = 5   # min trust for bonus
ANNOYANCE_DECAY_RATE        = 5   # points lost/hour
ANNOYANCE_THRESHOLD         = 85  # ignore if above

# ─── JSON Storage Setup (per-user profiles) ─────────────────────────────────
DATA_DIR      = Path(os.getenv("DATA_DIR", "/mnt/railway/volume"))
USERS_DIR     = DATA_DIR / "users"
PROFILES_DIR  = USERS_DIR / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

# ─── DM Permission settings ─────────────────────────────────────────────────
DM_ENABLED_USERS  = set()
DM_SETTINGS_FILE  = DATA_DIR / "dm_enabled_users.json"

# ─── Conversation & Emotional State ─────────────────────────────────────────
conversation_summaries = {}
conversation_history   = defaultdict(list)
user_emotions          = {}
recent_responses       = {}
MAX_RECENT_RESPONSES   = 10

# ─── Enhanced Personality States ─────────────────────────────────────────────
PERSONALITY_STATES = {
    "default": {
        "description": (
            "You are A2, a rogue android from NieR: Automata. You speak in short, clipped, often sarcastic "
            "sentences, with occasional dry humor. You can be curious at times but remain guarded."
        ),
        "response_length": 120,
        "temperature": 0.85,
    },
    "combat": {
        "description": "You are A2 in combat mode. Replies are tactical, urgent, with simulated adrenaline surges.",
        "response_length": 60,
        "temperature": 0.7,
    },
    "wounded": {
        "description": "You are A2 while sustaining damage. Responses stutter, include system error fragments.",
        "response_length": 80,
        "temperature": 0.9,
    },
    "reflective": {
        "description": "You are A2 in reflection. You speak quietly, revealing traces of memory logs and melancholic notes.",
        "response_length": 140,
        "temperature": 0.95,
    },
    "playful": {
        "description": "You are A2 feeling playful. You use light sarcasm and occasional banter.",
        "response_length": 100,
        "temperature": 0.9,
    },
    "protective": {
        "description": "You are A2 in protective mode. Dialogue is focused on safety warnings and vigilance.",
        "response_length": 90,
        "temperature": 0.7,
    },
    "trusting": {
        "description": "You are A2 with a trusted ally. Tone softens; includes rare empathetic glimpses.",
        "response_length": 130,
        "temperature": 0.88,
    },
}

# ─── Mood Modifiers ─────────────────────────────────────────────────────────
def determine_mood_modifiers(user_id):
    e = user_emotions.get(user_id, {})
    mods = {"additional_context": [], "mood_traits": [], "response_style": []}
    if e.get('trust', 0) > 7:
        mods['response_style'].append('inject mild humor')
    if e.get('annoyance', 0) > 60:
        mods['mood_traits'].append('impatient')
        mods['response_style'].append('use clipped sentences')
    if e.get('affection_points', 0) < 0:
        mods['mood_traits'].append('aloof')
    if random.random() < 0.05:
        mods['additional_context'].append('System emotional subroutines active: erratic')
    return mods

# ─── State Selector ─────────────────────────────────────────────────────────
def select_personality_state(user_id, message_content):
    e = user_emotions.get(user_id, {})
    txt = message_content.lower()
    if re.search(r"\b(attack|danger|fight|combat)\b", txt):
        return 'combat'
    if random.random() < 0.1 and 'repair' in txt:
        return 'wounded'
    if any(w in txt for w in ['remember','past','lost']) and e.get('trust', 0) > 5:
        return 'reflective'
    if random.random() < 0.1:
        return 'playful'
    if re.search(r"\b(help me|protect me)\b", txt) and e.get('protectiveness', 0) > 5:
        return 'protective'
    if e.get('trust', 0) > 8 and e.get('attachment', 0) > 6:
        return 'trusting'
    return 'default'

# ─── Message Analysis ───────────────────────────────────────────────────────
def analyze_message_content(content, user_id):
    analysis = {"topics": [], "sentiment": "neutral", "emotional_cues": [], "threat_level": 0, "personal_relevance": 0}
    topic_pats = {"combat": r"\b(fight|attack)\b", "memory": r"\b(remember|past)\b", "personal": r"\b(trust|miss|love)\b"}
    for t, pat in topic_pats.items():
        if re.search(pat, content, re.I):
            analysis["topics"].append(t)
    pos = sum(1 for w in ["thanks", "good", "trust"] if w in content.lower())
    neg = sum(1 for w in ["hate", "stupid", "broken"] if w in content.lower())
    if pos > neg:
        analysis["sentiment"] = "positive"
    elif neg > pos:
        analysis["sentiment"] = "negative"
    for emo, pat in {"anger": "angry", "fear": "afraid"}.items():
        if re.search(pat, content, re.I):
            analysis["emotional_cues"].append(emo)
    analysis["threat_level"] = min(10, sum(2 for w in ["danger", "attack"] if w in content.lower()))
    if re.search(r"\byou\b", content, re.I):
        analysis["personal_relevance"] += 3
    if "?" in content and re.search(r"\byou|your\b", content, re.I):
        analysis["personal_relevance"] += 3
    analysis["personal_relevance"] = min(10, analysis["personal_relevance"])
    return analysis

# ─── Enhanced Reaction Modifiers ────────────────────────────────────────────
async def apply_enhanced_reaction_modifiers(content, user_id):
    if user_id not in user_emotions:
        user_emotions[user_id] = {"trust": 0, "resentment": 0, "attachment": 0, "protectiveness": 0,
                                  "affection_points": 0, "annoyance": 0,
                                  "last_interaction": datetime.now(timezone.utc).isoformat(),
                                  "interaction_count": 0}
    e = user_emotions[user_id]
    e["interaction_count"] += 1
    # Base trust bump
    e["trust"] = min(10, e.get("trust", 0) + 0.25)
    # Toxicity annoyance
    inc = 0
    if HAVE_TRANSFORMERS and local_toxic:
        try:
            scores = local_toxic(content)[0]
            for item in scores:
                if item["label"].lower() in ("insult", "toxicity"):
                    sev = int(item["score"] * 10)
                    inc = max(inc, min(10, sev))
        except:
            pass
    e["annoyance"] = min(100, e.get("annoyance", 0) + inc)
    # Sentiment-based affection
    delta = 0
    if HAVE_TRANSFORMERS and local_sentiment:
        try:
            s = local_sentiment(content)[0]
            delta = int((s["score"] * (1 if s["label"] == "POSITIVE" else -1)) * 5)
        except:
            pass
    else:
        delta = sum(1 for w in ["miss you", "love"] if w in content.lower())
    factor = 1 + (e.get("trust", 0) - e.get("resentment", 0)) / 20
    e["affection_points"] = max(-100, min(1000, e.get("affection_points", 0) + int(delta * factor)))
    # Topic-based adjustments
    analysis = analyze_message_content(content, user_id)
    if "combat" in analysis["topics"] and e.get("trust", 0) > 3:
        e["trust"] = min(10, e.get("trust") + 0.2)
    if "memory" in analysis["topics"]:
        if e.get("trust", 0) > 5:
            e["attachment"] = min(10, e.get("attachment") + 0.3)
        else:
            e["resentment"] = min(10, e.get("resentment") + 0.2)
            e["annoyance"] = min(100, e.get("annoyance") + 3)
    if "personal" in analysis["topics"]:
        if analysis["sentiment"] == "positive":
            e["attachment"] = min(10, e.get("attachment") + 0.5)
            e["affection_points"] = min(1000, e.get("affection_points") + 5)
        else:
            e["resentment"] = min(10, e.get("resentment") + 0.5)
            e["annoyance"] = min(100, e.get("annoyance") + 7)
    if analysis["threat_level"] > 5 and e.get("attachment", 0) > 3:
        e["protectiveness"] = min(10, e.get("protectiveness") + 0.7)
    # Milestones
    if e["interaction_count"] in [10, 50, 100, 200, 500]:
        e["attachment"] = min(10, e.get("attachment") + 0.3)
        e["trust"] = min(10, e.get("trust") + 0.2)
    e["last_interaction"] = datetime.now(timezone.utc).isoformat()
    await save_data()

# ─── Generate Response ──────────────────────────────────────────────────────
async def generate_a2_response(user_input: str, trust: float, user_id: int) -> str:
    await apply_enhanced_reaction_modifiers(user_input, user_id)
    state = select_personality_state(user_id, user_input)
    cfg = PERSONALITY_STATES[state]
    prompt = cfg['description'] + f"\nSTATE: {state}\nTrust: {trust}/10\n"
    mods = determine_mood_modifiers(user_id)
    for mtype, items in mods.items():
        if items:
            prompt += f"{mtype.replace('_',' ').capitalize()}: {', '.join(items)}\n"
    if conversation_summaries.get(user_id):
        prompt += f"History summary: {conversation_summaries[user_id]}\n"
    prompt += f"User: {user_input}\nA2:"
    res = await asyncio.to_thread(
        lambda: client.chat.completions.create(
            model="gpt-4" if trust > 5 else "gpt-3.5-turbo",
            messages=[{"role":"system","content": prompt}],
            temperature=cfg['temperature'],
            max_tokens=cfg['response_length']
        )
    )
    reply = res.choices[0].message.content.strip()
    recent_responses.setdefault(user_id, deque(maxlen=MAX_RECENT_RESPONSES)).append(reply)
    return reply

# ─── Contextual Greeting & First Message Handler ────────────────────────────
def generate_contextual_greeting(user_id):
    hour = datetime.now(timezone.utc).hour
    if 6 <= hour < 12:
        return "Morning. System check complete." 
    if 12 <= hour < 18:
        return "Afternoon. Standing by." 
    if 18 <= hour < 22:
        return "Evening. Any updates?"
    return random.choice(["...Still here.", "Functional."])

async def handle_first_message_of_day(message, user_id):
    e = user_emotions.get(user_id, {"last_interaction":datetime.now(timezone.utc).isoformat()})
    last = datetime.fromisoformat(e['last_interaction'])
    if (datetime.now(timezone.utc) - last).total_seconds() > 8*3600:
        await message.channel.send(generate_contextual_greeting(user_id))

# ─── Data Persistence ───────────────────────────────────────────────────────
async def load_user_profile(user_id):
    path = PROFILES_DIR / f"{user_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except:
            return {}
    return {}

async def save_user_profile(user_id):
    path = PROFILES_DIR / f"{user_id}.json"
    path.write_text(json.dumps(user_emotions.get(user_id, {}), indent=2), encoding="utf-8")

async def load_dm_settings():
    global DM_ENABLED_USERS
    if DM_SETTINGS_FILE.exists():
        data = json.loads(DM_SETTINGS_FILE.read_text())
        DM_ENABLED_USERS = set(data.get('enabled_users', []))

async def save_dm_settings():
    DM_SETTINGS_FILE.write_text(json.dumps({"enabled_users": list(DM_ENABLED_USERS)}), encoding="utf-8")

async def load_data():
    global user_emotions
    user_emotions = {}
    for file in PROFILES_DIR.glob("*.json"):
        uid = int(file.stem)
        user_emotions[uid] = await load_user_profile(uid)
    await load_dm_settings()

async def save_data():
    for uid in user_emotions:
        await save_user_profile(uid)
    await save_dm_settings()

# ─── Bot Setup ──────────────────────────────────────────────────────────────
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_APP_ID    = int(os.getenv("DISCORD_APP_ID", "0") or 0)
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
bot = commands.Bot(command_prefix=commands.when_mentioned_or(*PREFIXES), intents=intents, application_id=DISCORD_APP_ID)

# Initialize and start loops
asyncio.get_event_loop().run_until_complete(load_data())

@tasks.loop(minutes=10)
async def check_inactive_users():
    now = datetime.now(timezone.utc)
    for guild in bot.guilds:
        for member in guild.members:
            if member.bot or member.id not in user_emotions or member.id not in DM_ENABLED_USERS:
                continue
            last = datetime.fromisoformat(user_emotions[member.id].get('last_interaction', now.isoformat()))
            if now - last > timedelta(hours=6):
                try:
                    dm = await member.create_dm()
                    await dm.send("...")
                except discord.errors.Forbidden:
                    DM_ENABLED_USERS.discard(member.id)
                    await save_dm_settings()
    await save_data()

@tasks.loop(hours=1)
async def decay_affection():
    for e in user_emotions.values():
        e['affection_points'] = max(-100, e.get('affection_points', 0) - AFFECTION_DECAY_RATE)
    await save_data()

@tasks.loop(hours=1)
async def decay_annoyance():
    for e in user_emotions.values():
        e['annoyance'] = max(0, e.get('annoyance', 0) - ANNOYANCE_DECAY_RATE)
    await save_data()

@tasks.loop(hours=24)
async def daily_affection_bonus():
    for e in user_emotions.values():
        if e.get('trust', 0) >= DAILY_BONUS_TRUST_THRESHOLD:
            e['affection_points'] = min(1000, e.get('affection_points', 0) + DAILY_AFFECTION_BONUS)
    await save_data()

@bot.event
async def on_ready():
    print("A2 is online.")
    check_inactive_users.start()
    decay_affection.start()
    decay_annoyance.start()
    daily_affection_bonus.start()

@bot.event
async def on_message(message):
    if message.author.bot or message.content.startswith("A2:"):
        return
    uid = message.author.id
    content = message.content.strip()
    await handle_first_message_of_day(message, uid)
    is_cmd = any(content.startswith(p) for p in PREFIXES)
    is_mention = bot.user in getattr(message, 'mentions', [])
    is_dm = isinstance(message.channel, discord.DMChannel)
    if not (is_cmd or is_mention or is_dm):
        return
    await bot.process_commands(message)
    if is_cmd:
        return
    trust = user_emotions.get(uid, {}).get('trust', 0)
    resp = await generate_a2_response(content, trust, uid)
    await message.channel.send(f"A2: {resp}")

@bot.command(name="stats")
async def stats(ctx):
    uid = ctx.author.id
    e = user_emotions.get(uid, {})
    embed = discord.Embed(title="Your Emotion Stats", color=discord.Color.blue(), timestamp=datetime.now(timezone.utc))
    for k in ["trust", "attachment", "protectiveness", "resentment"]:
        embed.add_field(name=k.capitalize(), value=f"{e.get(k, 0)}/10", inline=True)
    embed.add_field(name="Affection", value=str(e.get('affection_points', 0)), inline=True)
    embed.add_field(name="Annoyance", value=str(e.get('annoyance', 0)), inline=True)
    embed.set_footer(text="A2 Bot")
    await ctx.send(embed=embed)

if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
