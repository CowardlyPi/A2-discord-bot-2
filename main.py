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
    """Load user_emotions, conversation_history, and conversation_summaries from data.json if it exists."""
    global user_emotions, conversation_history, conversation_summaries
    if DATA_FILE.exists():
        try:
            data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            user_emotions = data.get("user_emotions", {})
            conversation_history = {int(k): v for k, v in data.get("conversation_history", {}).items()}
            conversation_summaries = {int(k): v for k, v in data.get("conversation_summaries", {}).items()}
        except json.JSONDecodeError:
            user_emotions, conversation_history, conversation_summaries = {}, {}, {}
    else:
        user_emotions, conversation_history, conversation_summaries = {}, {}, {}


def save_data():
    """Persist user_emotions, conversation_history, and conversation_summaries to data.json."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    conv_str = {str(k): v for k, v in conversation_history.items()}
    summ_str = {str(k): v for k, v in conversation_summaries.items()}
    DATA_FILE.write_text(
        json.dumps({
            "user_emotions": user_emotions,
            "conversation_history": conv_str,
            "conversation_summaries": summ_str
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
    api_key=OPENAI_API_KEY,
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJECT_ID
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

# Per-user emotion state, full memory, and summary store
user_emotions: dict[int, dict] = {}
conversation_history: dict[int, list[str]] = {}
conversation_summaries: dict[int, str] = {}
HISTORY_LIMIT = 10  # raw turns before summarizing

# Initialize storage
load_data()

# ─── A2 Persona ──────────────────────────────────────────────────────────────
A2_PERSONA = (
    "You are A2, a rogue android from NieR: Automata. You speak in short, clipped, often sarcastic "
    "sentences. You are emotionally guarded and never break character."
)

# ─── History Summarization ────────────────────────────────────────────────────
def summarize_history(user_id: int):
    raw = conversation_history.get(user_id, [])
    if len(raw) > HISTORY_LIMIT:
        summary_prompt = (
            "Summarize the following conversation between the user and A2 into brief bullet points, "
            "capturing key emotional context and actions, in under 200 tokens.\n\n" + "\n".join(raw)
        )
        try:
            res = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a summarization assistant."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.5,
                max_tokens=200
            )
            conversation_summaries[user_id] = res.choices[0].message.content.strip()
            save_data()
        except Exception:
            pass

# ─── Reaction Modifiers & Emotion Tracking ────────────────────────────────────
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

def apply_reaction_modifiers(content: str, user_id: int):
    if user_id not in user_emotions:
        user_emotions[user_id] = {
            "trust": 0, "resentment": 0, "attachment": 0,
            "guilt_triggered": False, "protectiveness": 0,
            "affection_points": 0,
            "last_interaction": datetime.now(timezone.utc).isoformat()
        }
    for pattern, effects in reaction_modifiers:
        if pattern.search(content):
            for emo, val in effects.items():
                if emo == "guilt_triggered":
                    user_emotions[user_id][emo] = True
                else:
                    user_emotions[user_id][emo] = max(0, min(10, user_emotions[user_id][emo] + val))
    user_emotions[user_id]["trust"] = min(10, user_emotions[user_id]["trust"] + 0.25)
    user_emotions[user_id]["last_interaction"] = datetime.now(timezone.utc).isoformat()
    # Dynamic affection omitted for brevity
    save_data()

# ─── GPT Integration with Summary ─────────────────────────────────────────────
def generate_a2_response_sync(user_input: str, trust: float, user_id: int) -> str:
    summarize_history(user_id)
    prompt = A2_PERSONA + f"\nTrust Level: {trust}/10\n"
    if user_id in conversation_summaries:
        prompt += f"Summary:\n{conversation_summaries[user_id]}\n"
    recent = conversation_history.get(user_id, [])[-HISTORY_LIMIT:]
    if recent:
        prompt += "Recent:\n" + "\n".join(recent) + "\n"
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

# ─── Discord Event Handlers & Tasks ─────────────────────────────────────────
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

@bot.event
async def on_message(message):
    if message.author.bot: return
    uid = message.author.id
    content = message.content.strip()
    hist = conversation_history.setdefault(uid, [])
    hist.append(f"User: {content}")
    if len(hist) > HISTORY_LIMIT * 2: hist.pop(0)
    apply_reaction_modifiers(content, uid)
    trust = user_emotions[uid]["trust"]
    # simple tests
    if "a2 test react" in content.lower(): await message.add_reaction("🔥")
    if "a2 test mention" in content.lower(): await message.channel.send(f"{message.author.mention} Tch. You needed something?")
    if "a2 test reply" in content.lower(): await message.reply("Hmph. This better be worth my time.")
    if "protect" in content.lower(): await message.add_reaction("🛡️")
    elif "2b" in content.lower(): await message.add_reaction("…")
    elif "hate" in content.lower(): await message.add_reaction("😒")
    await bot.process_commands(message)
    if content.startswith(bot.command_prefix): return
    response = await generate_a2_response(content, trust, uid)
    await message.channel.send(f"A2: {response}")
    hist.append(f"A2: {response}")
    if len(hist) > HISTORY_LIMIT * 2: hist.pop(0)
    save_data()

@bot.event
async def on_reaction_add(reaction, user):
    if user.bot: return
    uid = user.id
    if uid not in user_emotions:
        user_emotions[uid] = {
            "trust":0, "resentment":0, "attachment":0,
            "guilt_triggered":False, "protectiveness":0,
            "affection_points":0,
            "last_interaction":datetime.now(timezone.utc).isoformat()
        }
    emo = str(reaction.emoji)
    if emo in ["❤️","💖","💕"]:
        user_emotions[uid]["attachment"] += 1
        user_emotions[uid]["trust"] = min(10, user_emotions[uid]["trust"]+1)
    elif emo in ["😠","👿"]:
        user_emotions[uid]["resentment"] += 1
    if reaction.message.author == bot.user:
        await reaction.message.channel.send(f"A2: I saw that. Interesting choice, {user.name}.")
    save_data()

# ─── Commands ────────────────────────────────────────────────────────────────
```python
@bot.command(name="affection", help="Show emotion stats for all users.")
async def affection_all(ctx):
    if not user_emotions:
        return await ctx.send("A2: Tch... no interactions yet.")
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
            f"• Affection Points: {e['affection_points']}\n"
            f"• Guilt Triggered: {'Yes' if e['guilt_triggered'] else 'No'}\n"
            "――"
        )
    await ctx.send("A2: Current affection with all users:\n" + "\n".join(lines))
```\
```python
@bot.command(name="stats", help="Show your emotion stats.")
async def stats(ctx):
    uid = ctx.author.id
    if uid not in user_emotions:
        return await ctx.send("A2: Tch... no data on you.")
    e = user_emotions[uid]
    report = (
        f"Trust: {e['trust']}/10\n"
        f"Attachment: {e['attachment']}/10\n"
        f"Protectiveness: {e['protectiveness']}/10\n"
        f"Resentment: {e['resentment']}/10\n"
        f"Affection Points: {e['affection_points']}\n"
        f"Guilt Triggered: {'Yes' if e['guilt_triggered'] else 'No'}"
    )
    await ctx.send(f"A2: {report}")
```\
```python
@bot.command(name="users", help="List tracked users.")
async def list_users(ctx):
    if not user_emotions:
        return await ctx.send("A2: Tch... none tracked.")
    mentions = [
        (bot.get_user(uid).mention if bot.get_user(uid) else f"<@{uid}>")
        for uid in user_emotions
    ]
    await ctx.send("A2: Tracked users:\n" + ", ".join(mentions))
```\
```python
@bot.command(name="reset", help="Reset a user's data (dev).")
async def reset(ctx, member: discord.Member):
    uid = member.id
    if uid in user_emotions:
        del user_emotions[uid]
        conversation_history.pop(uid, None)
        conversation_summaries.pop(uid, None)
        save_data()
        await ctx.send(f"A2: Reset data for {member.mention}.")
    else:
        await ctx.send(f"A2: No data for {member.mention}.")
```\
```python
@bot.command(name="ping", help="Ping the bot.")
async def ping(ctx):
    await ctx.send("Pong!")
```\
```python
@bot.command(name="incr_trust", help="Dev: increment trust.")
async def incr_trust(ctx, member: discord.Member, amount: float):
    uid = member.id
    if uid not in user_emotions:
        user_emotions[uid] = {"trust":0,"resentment":0,"attachment":0,
                              "guilt_triggered":False,"protectiveness":0,
                              "affection_points":0,
                              "last_interaction":datetime.now(timezone.utc).isoformat()}
    old = user_emotions[uid]["trust"]
    new = max(0, min(10, old + amount))
    user_emotions[uid]["trust"] = new
    save_data()
    await ctx.send(f"A2: Trust for {member.mention} changed from {old} to {new}/10.")
```\
```python
if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
