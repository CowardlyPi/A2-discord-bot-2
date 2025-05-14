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
from dotenv import load_dotenv

# ─── Dynamic Affection Settings ───────────────────────────────────────────────
# How many affection points to lose per hour
AFFECTION_DECAY_RATE = 1
# How many points to add once per day when trust ≥ threshold
DAILY_AFFECTION_BONUS = 5
# Minimum trust required for the daily bonus
DAILY_BONUS_TRUST_THRESHOLD = 5

# ─── JSON Storage Setup ───────────────────────────────────────────────────────
DATA_FILE = Path("data.json")

def load_data():
    """Load user_emotions and conversation_history from data.json if it exists."""
    global user_emotions, conversation_history
    if DATA_FILE.exists():
        data = json.loads(DATA_FILE.read_text())
        user_emotions = data.get("user_emotions", {})
        # Convert conversation_history keys back to int
        conversation_history = {
            int(k): v for k, v in data.get("conversation_history", {}).items()
        }
    else:
        user_emotions = {}
        conversation_history = {}

def save_data():
    """Persist user_emotions and conversation_history to data.json."""
    # Convert conversation_history keys to str for JSON
    conv_str_keys = {str(k): v for k, v in conversation_history.items()}
    DATA_FILE.write_text(
        json.dumps({
            "user_emotions": user_emotions,
            "conversation_history": conv_str_keys
        }, indent=2)
    )

# ─── Configuration & State ────────────────────────────────────────────────────
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID     = os.getenv("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")

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

bot = commands.Bot(command_prefix="!a2 ", intents=intents)

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
            "last_interaction": datetime.now(timezone.utc).isoformat()
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

def get_emotion_context(user_id):
    e = user_emotions[user_id]
    ctx = ""
    if e.get("guilt_triggered"):
        ctx += "They reminded you of 2B. That hurt.\n"
    if e.get("attachment", 0) >= 7:
        ctx += "You care deeply whether they live. You hate that you care.\n"
    elif e.get("attachment", 0) >= 4:
        ctx += "You care whether they live. You won’t admit it.\n"
    if e.get("protectiveness", 0) >= 7:
        ctx += "You’d risk yourself to protect them. That terrifies you.\n"
    elif e.get("protectiveness", 0) >= 4:
        ctx += "You’d fight for them. Not because you're kind. Because they matter.\n"
    if e.get("resentment", 0) >= 6:
        ctx += "They’ve hurt you. You don’t forget betrayal. Everything you say is colder.\n"
    elif e.get("resentment", 0) >= 3:
        ctx += "They’ve hurt you before. You don’t forgive easily. You keep distance in your tone.\n"
    if e.get("trust", 0) >= 8 and e.get("resentment", 0) <= 2:
        ctx += "You trust them almost fully. A quiet warmth flickers beneath your words.\n"
    return ctx

# ─── GPT Integration with Memory ──────────────────────────────────────────────
def generate_a2_response_sync(user_input, trust_level, user_id):
    prompt = A2_PERSONA + f"\nTrust Level: {trust_level}/10\n" + get_emotion_context(user_id) + "\n"
    history = conversation_history.get(user_id, [])
    if history:
        prompt += "Conversation so far:\n" + "\n".join(history) + "\n"
    prompt += f"User: {user_input}\nA2:"
    try:
        res = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI Error:", e)
        return "...I’m not in the mood."

async def generate_a2_response(user_input, trust_level, user_id):
    return await asyncio.to_thread(generate_a2_response_sync, user_input, trust_level, user_id)

# ─── Discord Event Handlers ─────────────────────────────────────────────────
@bot.event
async def on_ready():
    print("A2 is online.")
    check_inactive_users.start()
    decay_affection.start()
    daily_affection_bonus.start()

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
                if user_emotions[member.id]["trust"] >= 7 and user_emotions[member.id]["resentment"] <= 3:
                    await dm.send(random.choice(warm_lines))
                elif user_emotions[member.id]["resentment"] >= 7 and user_emotions[member.id]["trust"] <= 2:
                    await dm.send(random.choice(provoking_lines))
    save_data()

@tasks.loop(hours=1)
async def decay_affection():
    """Every hour, drift affection points downward."""
    for e in user_emotions.values():
        e["affection_points"] = max(-100, e["affection_points"] - AFFECTION_DECAY_RATE)
    save_data()

@tasks.loop(hours=24)
async def daily_affection_bonus():
    """Daily bonus for trusted users."""
    for e in user_emotions.values():
        if e["trust"] >= DAILY_BONUS_TRUST_THRESHOLD:
            e["affection_points"] = min(1000, e["affection_points"] + DAILY_AFFECTION_BONUS)
    save_data()

@bot.event
async def on_message(message):
    if message.author.bot:
        return

    uid     = message.author.id
    content = message.content.strip()

    # 1) Record user turn
    hist = conversation_history.setdefault(uid, [])
    hist.append(f"User: {content}")
    if len(hist) > HISTORY_LIMIT:
        hist.pop(0)

    apply_reaction_modifiers(content, uid)
    trust = user_emotions[uid]["trust"]

    # Test phrases & reactions
    if "a2 test react" in content.lower():
        await message.add_reaction("🔥")
    if "a2 test mention" in content.lower():
        await message.channel.send(f"{message.author.mention} Tch. You needed something?")
    if "a2 test reply" in content.lower():
        await message.reply("Hmph. This better be worth my time.")
    if "protect" in content.lower():
        await message.add_reaction("🛡️")
    elif "2b" in content.lower():
        await message.add_reaction("…")
    elif "hate" in content.lower():
        await message.add_reaction("😒")

    # Process commands
    await bot.process_commands(message)
    if content.startswith(bot.command_prefix):
        return

    # 2) Generate A2 reply
    mentions = [m.mention for m in message.mentions if not m.bot]
    mention_txt = f" You mentioned {', '.join(mentions)}." if mentions else ""
    reply_ctx = ""
    if message.reference:
        try:
            parent = await message.channel.fetch_message(message.reference.message_id)
            reply_ctx = f" You replied to: \"{parent.content}\""
        except:
            pass

    full_input = content + mention_txt + reply_ctx
    response = await generate_a2_response(full_input, trust, uid)
    await message.channel.send(f"A2: {response}")

    # 3) Record A2 turn
    hist.append(f"A2: {response}")
    if len(hist) > HISTORY_LIMIT:
        hist.pop(0)

    save_data()

@bot.event
async def on_reaction_add(reaction, user):
    if user.bot:
        return

    uid = user.id
    if uid not in user_emotions:
        user_emotions[uid] = {
            "trust": 0, "resentment": 0, "attachment": 0,
            "guilt_triggered": False, "protectiveness": 0,
            "affection_points": 0,
            "last_interaction": datetime.now(timezone.utc).isoformat()
        }

    if str(reaction.emoji) in ["❤️", "💖", "💕"]:
        user_emotions[uid]["attachment"] += 1
        user_emotions[uid]["trust"]      = min(user_emotions[uid]["trust"] + 1, 10)
    elif str(reaction.emoji) in ["😠", "👿"]:
        user_emotions[uid]["resentment"] += 1

    if reaction.message.author == bot.user:
        await reaction.message.channel.send(
            f"A2: I saw that. Interesting choice, {user.name}."
        )

    save_data()

# ─── Helpers for Commands ──────────────────────────────────────────────────────
def describe_points(value: int) -> str:
    if value <= -50:
        return "She can barely tolerate you."
    elif value < 0:
        return "She’s wary and cold."
    elif value < 200:
        return "You’re mostly ignored."
    elif value < 400:
        return "She’s paying attention."
    elif value < 600:
        return "She respects you, maybe more."
    elif value < 800:
        return "She trusts you. This is rare."
    else:
        return "You matter to her deeply. She’d never say it, though."

# ─── Global & Per-User Commands + Testing Helpers ────────────────────────────
@bot.command(name="affection", help="Show emotion stats for every user.")
async def affection_all(ctx: commands.Context):
    if not user_emotions:
        return await ctx.send("A2: Tch... she hasn't interacted with anyone yet.")
    lines = []
    for uid, e in user_emotions.items():
        member = bot.get_user(uid) or ctx.guild.get_member(uid)
        mention = member.mention if member else f"<@{uid}>"
        lines.append(
            f"**{mention}**\n"
            f"• Trust: {e['trust']}/10\n"
            f"• Attachment: {e['attachment']}/10\n"
            f"• Protectiveness: {e['protectiveness']}/10\n"
            f"• Resentment: {e['resentment']}/10\n"
            f"• Affection Points: {e['affection_points']} ({describe_points(e['affection_points'])})\n"
            f"• Guilt Triggered: {'Yes' if e['guilt_triggered'] else 'No'}\n"
            "――"
        )
    await ctx.send("A2: Current affection with all users:\n" + "\n".join(lines))

@bot.command(name="stats", help="Show your current emotion stats.")
async def stats(ctx):
    uid = ctx.author.id
    if uid not in user_emotions:
        return await ctx.send("A2: Tch... I haven't interacted with you yet.")
    e = user_emotions[uid]
    report = (
        f"Tch... fine.\n"
        f"Trust: {e['trust']}/10\n"
        f"Attachment: {e['attachment']}/10\n"
        f"Protectiveness: {e['protectiveness']}/10\n"
        f"Resentment: {e['resentment']}/10\n"
        f"Affection Points: {e['affection_points']} ({describe_points(e['affection_points'])})\n"
        f"Guilt Triggered: {'Yes' if e['guilt_triggered'] else 'No'}"
    )
    await ctx.send(f"A2: {report}")

@bot.command(name="stats_user", help="Show a specified user's emotion stats.")
async def stats_user(ctx, member: discord.Member):
    uid = member.id
    if uid not in user_emotions:
        return await ctx.send(f"A2: {member.mention} hasn't interacted with me yet.")
    e = user_emotions[uid]
    report = (
        f"Tch... fine.\n"
        f"Trust: {e['trust']}/10\n"
        f"Attachment: {e['attachment']}/10\n"
        f"Protectiveness: {e['protectiveness']}/10\n"
        f"Resentment: {e['resentment']}/10\n"
        f"Affection Points: {e['affection_points']} ({describe_points(e['affection_points'])})\n"
        f"Guilt Triggered: {'Yes' if e['guilt_triggered'] else 'No'}"
    )
    await ctx.send(f"A2: {report}")

@bot.command(name="users", help="List all users with tracked emotion data.")
async def list_users(ctx):
    if not user_emotions:
        return await ctx.send("A2: Tch... no users are being tracked yet.")
    mentions = [
        (bot.get_user(uid).mention if bot.get_user(uid) else f"<@{uid}>")
        for uid in user_emotions
    ]
    await ctx.send("A2: Tracked users:\n" + ", ".join(mentions))

@bot.command(name="reset", help="Reset a user's emotion state (dev only).")
async def reset(ctx, member: discord.Member):
    uid = member.id
    if uid in user_emotions:
        del user_emotions[uid]
        conversation_history.pop(uid, None)
        save_data()
        await ctx.send(f"A2: Reset emotions for {member.mention}.")
    else:
        await ctx.send(f"A2: No data to reset for {member.mention}.")

@bot.command(name="ping", help="Check bot responsiveness.")
async def ping(ctx):
    await ctx.send("Pong!")

@bot.command(name="incr_trust", help="Dev: increment a user's trust level by <amount>.")
async def incr_trust(ctx, member: discord.Member, amount: float):
    uid = member.id
    if uid not in user_emotions:
        user_emotions[uid] = {
            "trust": 0, "resentment": 0, "attachment": 0,
            "guilt_triggered": False, "protectiveness": 0,
            "affection_points": 0,
            "last_interaction": datetime.now(timezone.utc).isoformat()
        }
    old = user_emotions[uid]["trust"]
    new = max(0, min(10, old + amount))
    user_emotions[uid]["trust"] = new
    save_data()
    await ctx.send(f"A2: Trust for {member.mention} changed from {old} to {new}/10.")

# ─── Launch ───────────────────────────────────────────────────────────────────
bot.run(DISCORD_BOT_TOKEN)
