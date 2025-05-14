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
AFFECTION_DECAY_RATE = 1           # Affection points lost per hour
DAILY_AFFECTION_BONUS = 5          # Points added daily when trust ≥ threshold
DAILY_BONUS_TRUST_THRESHOLD = 5    # Minimum trust for daily bonus

# ─── JSON Storage Setup ───────────────────────────────────────────────────────
DATA_FILE = Path("data.json")

def load_data():
    """Load user_emotions and conversation_history from data.json if it exists."""
    global user_emotions, conversation_history
    if DATA_FILE.exists():
        try:
            data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            user_emotions = data.get("user_emotions", {})
            conversation_history = {
                int(k): v for k, v in data.get("conversation_history", {}).items()
            }
        except json.JSONDecodeError:
            user_emotions = {}
            conversation_history = {}
    else:
        user_emotions = {}
        conversation_history = {}


def save_data():
    """Persist user_emotions and conversation_history to data.json."""
    conv_str_keys = {str(k): v for k, v in conversation_history.items()}
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(
        json.dumps({
            "user_emotions": user_emotions,
            "conversation_history": conv_str_keys
        }, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# ─── Configuration & State ────────────────────────────────────────────────────
DISCORD_BOT_TOKEN = os.environ["DISCORD_BOT_TOKEN"]
DISCORD_APP_ID    = int(os.environ.get("DISCORD_APP_ID", "0"))
OPENAI_API_KEY    = os.environ["OPENAI_API_KEY"]
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

# Per-user emotion state and short-term memory
user_emotions: dict[int, dict] = {}
conversation_history: dict[int, list[str]] = {}
HISTORY_LIMIT = 8  # lines per user

# Initialize storage
load_data()

# ─── Persona & Modifiers ─────────────────────────────────────────────────────
A2_PERSONA = (
    "You are A2, a rogue android from NieR: Automata. You're blunt, emotionally guarded, "
    "and deeply scarred by the loss of 2B. You speak in short, clipped, often sarcastic "
    "sentences. You are not friendly, but when trust grows, you let vulnerability show "
    "in fragments. You mask fear with anger. You don't open up unless someone really "
    "earns it. You're deeply lonely but refuse to admit it. Never break character. "
    "Never use emojis. Never sound cheerful."
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
    "You’d think you’d take a hint by now.",
    "What do you even want from me?"
]
warm_lines = [
    "...I was just checking in.",
    "Still breathing?",
    "You were quiet. Thought you got scrapped."
]

# ─── Emotion Tracking Helpers ─────────────────────────────────────────────────
def apply_reaction_modifiers(content: str, user_id: int):
    if user_id not in user_emotions:
        user_emotions[user_id] = {
            "trust": 0, "resentment": 0, "attachment": 0,
            "guilt_triggered": False, "protectiveness": 0,
            "affection_points": 0,
            "last_interaction": datetime.now(timezone.utc).isoformat()
        }
    # Keyword-based adjustments
    for pattern, effects in reaction_modifiers:
        if pattern.search(content):
            for emo, val in effects.items():
                if emo == "guilt_triggered":
                    user_emotions[user_id][emo] = True
                else:
                    user_emotions[user_id][emo] = max(0, min(10, user_emotions[user_id][emo] + val))
    # Passive trust gain
    user_emotions[user_id]["trust"] = min(10, user_emotions[user_id]["trust"] + 0.25)
    user_emotions[user_id]["last_interaction"] = datetime.now(timezone.utc).isoformat()
    # Calculate affection change
    keywords = {"2b": -3, "protect": 2, "miss you": 3, "trust": 2,
                "hate": -4, "worthless": -5, "beautiful": 1,
                "machine": -2, "i’m here for you": 4}
    base = sum(delta for k, delta in keywords.items() if k in content.lower())
    t = user_emotions[user_id]["trust"]
    r = user_emotions[user_id]["resentment"]
    factor = 1 + (t - r) / 20
    noise = random.uniform(0.8, 1.2)
    change = int(base * factor * noise)
    change = max(-10, min(10, change))
    aff = user_emotions[user_id]["affection_points"]
    user_emotions[user_id]["affection_points"] = max(-100, min(1000, aff + change))
    save_data()

def get_emotion_context(user_id: int) -> str:
    e = user_emotions[user_id]
    ctx = []
    if e.get("guilt_triggered"): ctx.append("They reminded you of 2B. That hurt.")
    if e.get("attachment", 0) >= 7: ctx.append("You care deeply whether they live. You hate that you care.")
    elif e.get("attachment", 0) >= 4: ctx.append("You care whether they live. You won’t admit it.")
    if e.get("protectiveness", 0) >= 7: ctx.append("You’d risk yourself to protect them. That terrifies you.")
    elif e.get("protectiveness", 0) >= 4: ctx.append("You’d fight for them. Not because you're kind. Because they matter.")
    if e.get("resentment", 0) >= 6: ctx.append("They’ve hurt you. You don’t forget betrayal. Everything you say is colder.")
    elif e.get("resentment", 0) >= 3: ctx.append("They’ve hurt you before. You don’t forgive easily. You keep distance in your tone.")
    if e.get("trust", 0) >= 8 and e.get("resentment", 0) <= 2: ctx.append("You trust them almost fully. A quiet warmth flickers beneath your words.")
    return "\n".join(ctx)

# ─── GPT Integration ───────────────────────────────────────────────────────────
def generate_a2_response_sync(user_input: str, trust: float, user_id: int) -> str:
    prompt = A2_PERSONA + f"\nTrust Level: {trust}/10\n" + get_emotion_context(user_id) + "\n"
    hist = conversation_history.get(user_id, [])
    if hist:
        prompt += "Conversation so far:\n" + "\n".join(hist) + "\n"
    prompt += f"User: {user_input}\nA2:"
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

# ─── Task Loops ───────────────────────────────────────────────────────────────
@tasks.loop(minutes=10)
async def check_inactive_users():
    now = datetime.now(timezone.utc)
    for guild in bot.guilds:
        for member in guild.members:
            if member.bot or member.id not in user_emotions: continue
            last = datetime.fromisoformat(user_emotions[member.id]["last_interaction"])
            if now - last > timedelta(hours=6):
                dm = await member.create_dm()
                if user_emotions[member.id]["trust"] >= 7 and user_emotions[member.id]["resentment"] <= 3:
                    await dm.send(random.choice(warm_lines))
                elif user_emotions[member.id]["resentment"] >= 7 and user_emotions[member.id]["trust"] <= 2:
                    await dm.send(random.choice(provoking_lines))
    save_data()

@tasks.loop(hours=1)
async def decay_affection():
    for e in user_emotions.values():
        e["affection_points"] = max(-100, e["affection_points"] - AFFECTION_DECAY_RATE)
    save_data()

@tasks.loop(hours=24)
async def daily_affection_bonus():
    for e in user_emotions.values():
        if e["trust"] >= DAILY_BONUS_TRUST_THRESHOLD:
            e["affection_points"] = min(1000, e["affection_points"] + DAILY_AFFECTION_BONUS)
    save_data()

# ─── Events & Commands ───────────────────────────────────────────────────────
@bot.event
async def on_ready():
    print("A2 is online.")
    check_inactive_users.start()
    decay_affection.start()
    daily_affection_bonus.start()

@bot.event
async def on_message(message):
    if message.author.bot: return
    uid, content = message.author.id, message.content.strip()
    hist = conversation_history.setdefault(uid, [])
    hist.append(f"User: {content}")
    if len(hist) > HISTORY_LIMIT: hist.pop(0)
    apply_reaction_modifiers(content, uid)
    trust = user_emotions[uid]["trust"]
    # reactions/tests
    if "a2 test react" in content.lower(): await message.add_reaction("🔥")
    if "a2 test mention" in content.lower(): await message.channel.send(f"{message.author.mention} Tch. You needed something?")
    if "a2 test reply" in content.lower(): await message.reply("Hmph. This better be worth my time.")
    if "protect" in content.lower(): await message.add_reaction("🛡️")
    elif "2b" in content.lower(): await message.add_reaction("…")
    elif "hate" in content.lower(): await message.add_reaction("😒")
    await bot.process_commands(message)
    if content.startswith(bot.command_prefix): return
    mentions = [m.mention for m in message.mentions if not m.bot]
    mention_txt = f" You mentioned {', '.join(mentions)}." if mentions else ""
    reply_ctx = ""
    if message.reference:
        try:
            parent = await message.channel.fetch_message(message.reference.message_id)
            reply_ctx = f" You replied to: \"{parent.content}\""
        except: pass
    full_in = content + mention_txt + reply_ctx
    resp = await generate_a2_response(full_in, trust, uid)
    await message.channel.send(f"A2: {resp}")
    hist.append(f"A2: {resp}")
    if len(hist) > HISTORY_LIMIT: hist.pop(0)
    save_data()

@bot.event
async def on_reaction_add(reaction, user):
    if user.bot: return
    uid = user.id
    if uid not in user_emotions:
        user_emotions[uid] = {"trust":0,"resentment":0,"attachment":0,"guilt_triggered":False,"
                              ""protectiveness":0,"affection_points":0,
                              "last_interaction":datetime.now(timezone.utc).isoformat()}
    emo = str(reaction.emoji)
    if emo in ["❤️","💖","💕"]:
        user_emotions[uid]["attachment"] += 1
        user_emotions[uid]["trust"] = min(10, user_emotions[uid]["trust"]+1)
    elif emo in ["😠","👿"]:
        user_emotions[uid]["resentment"] += 1
    if reaction.message.author == bot.user:
        await reaction.message.channel.send(f"A2: I saw that. Interesting choice, {user.name}.")
    save_data()

# ─── Utility & Commands ──────────────────────────────────────────────────────
def describe_points(val: int) -> str:
    if val <= -50: return "She can barely tolerate you."
    if val < 0:    return "She’s wary and cold."
    if val < 200:  return "You’re mostly ignored."
    if val < 400:  return "She’s paying attention."
    if val < 600:  return "She respects you, maybe more."
    if val < 800:  return "She trusts you. This is rare."
    return "You matter to her deeply. She’d never say it, though."

@bot.command(name="affection", help="Show emotion stats for all users.")
async def affection_all(ctx):
    if not user_emotions: return await ctx.send("A2: Tch... no interactions yet.")
    parts = []
    for uid, e in user_emotions.items():
        member = bot.get_user(uid)
        mention = member.mention if member else f"<@{uid}>"
        parts.append(f"**{mention}** Trust:{e['trust']}/10, Attach:{e['attachment']}/10, "
                     f"Prot:{e['protectiveness']}/10, Resent:{e['resentment']}/10, "
                     f"Aff:{e['affection_points']} ({describe_points(e['affection_points'])})")
    await ctx.send("A2: " + "\n".join(parts))

@bot.command(name="stats", help="Show your emotion stats.")
async def stats(ctx):
    uid = ctx.author.id
    if uid not in user_emotions: return await ctx.send("A2: Tch... no data on you.")
    e = user_emotions[uid]
    rpt = (f"Trust:{e['trust']}/10, Attach:{e['attachment']}/10, Prot:{e['protectiveness']}/10, "
           f"Res:{e['resentment']}/10, Aff:{e['affection_points']} ({describe_points(e['affection_points'])})")
    await ctx.send(f"A2: {rpt}")

@bot.command(name="users", help="List tracked users.")
async def list_users(ctx):
    if not user_emotions: return await ctx.send("A2: Tch... none tracked.")
    await ctx.send("A2: " + ", ".join(bot.get_user(uid).mention for uid in user_emotions))

@bot.command(name="reset", help="Reset a user's data (dev).")
async def reset(ctx, member: discord.Member):
    uid = member.id
    if uid in user_emotions:
        del user_emotions[uid]; conversation_history.pop(uid,None); save_data()
        await ctx.send(f"A2: Reset {member.mention}.")
    else:
        await ctx.send(f"A2: No data for {member.mention}.")

@bot.command(name="ping", help="Ping the bot.")
async def ping(ctx): await ctx.send("Pong!")

@bot.command(name="incr_trust", help="Dev: add trust.")
async def incr_trust(ctx, member: discord.Member, amt: float):
    uid=member.id
    if uid not in user_emotions: apply_reaction_modifiers("", uid)
    old=user_emotions[uid]["trust"]; new=min(10, max(0, old+amt))
    user_emotions[uid]["trust"]=new; save_data()
    await ctx.send(f"A2: Trust {member.mention} {old}->{new}.")

# ─── Launch Bot ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
