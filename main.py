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
# Persist data into Railway volume (mount path must be /mnt/railway/volume)
DATA_FILE = Path("/mnt/railway/volume/data.json")

def load_data():
    global user_emotions, conversation_history, conversation_summaries
    if DATA_FILE.exists():
        try:
            data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            user_emotions = data.get("user_emotions", {})
            conversation_history = {int(k): v for k, v in data.get("conversation_history", {}).items()}
            conversation_summaries = data.get("conversation_summaries", {})
        except json.JSONDecodeError:
            user_emotions, conversation_history, conversation_summaries = {}, {}, {}
    else:
        user_emotions, conversation_history, conversation_summaries = {}, {}, {}

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

# Per-user state
user_emotions: dict[int, dict] = {}
conversation_history: dict[int, list[str]] = {}
conversation_summaries: dict[int, str] = {}
HISTORY_LIMIT = 10

load_data()

# ─── Persona & Modifiers ─────────────────────────────────────────────────────
A2_PERSONA = (
    "You are A2, a rogue android from NieR: Automata. You speak in short, clipped, often sarcastic "
    "sentences. You are emotionally guarded and never break character."
)
reaction_modifiers = [
    (re.compile(r"\bi'll protect you\b", re.I),    {"trust": 2, "protectiveness": 1}),
    # ... other patterns ...
]
provoking_lines = ["Still mad about last time? Good.", "You again? Tch.", "What do you even want from me?"]
warm_lines = ["...I was just checking in.", "Still breathing?", "You were quiet. Thought you got scrapped."]

def apply_reaction_modifiers(content: str, user_id: int):
    if user_id not in user_emotions:
        user_emotions[user_id] = {"trust": 0, "resentment": 0, "attachment": 0,
                                  "guilt_triggered": False, "protectiveness": 0,
                                  "affection_points": 0,
                                  "last_interaction": datetime.now(timezone.utc).isoformat()}
    for pattern, effects in reaction_modifiers:
        if pattern.search(content):
            for emo, val in effects.items():
                if emo == "guilt_triggered":
                    user_emotions[user_id][emo] = True
                else:
                    user_emotions[user_id][emo] = max(0, min(10, user_emotions[user_id][emo] + val))
    user_emotions[user_id]["trust"] = min(10, user_emotions[user_id]["trust"] + 0.25)
    user_emotions[user_id]["last_interaction"] = datetime.now(timezone.utc).isoformat()
    save_data()

def summarize_history(user_id: int):
    raw = conversation_history.get(user_id, [])
    if len(raw) > HISTORY_LIMIT:
        prompt = "Summarize the conversation into bullet points under 200 tokens.\n\n" + "\n".join(raw)
        res = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a summarization assistant."},
                      {"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200
        )
        conversation_summaries[user_id] = res.choices[0].message.content.strip()
        save_data()

def generate_a2_response_sync(user_input: str, trust: float, user_id: int) -> str:
    summarize_history(user_id)
    prompt = A2_PERSONA + f"\nTrust Level: {trust}/10\n"
    if summary := conversation_summaries.get(user_id):
        prompt += f"Summary:\n{summary}\n"
    recent = conversation_history.get(user_id, [])[-HISTORY_LIMIT:]
    if recent:
        prompt += "Recent:\n" + "\n".join(recent) + "\n"
    prompt += f"User: {user_input}\nA2:"
    res = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
        max_tokens=150
    )
    return res.choices[0].message.content.strip()

async def generate_a2_response(user_input: str, trust: float, user_id: int) -> str:
    return await asyncio.to_thread(generate_a2_response_sync, user_input, trust, user_id)

# ─── Discord Tasks & Events ─────────────────────────────────────────────────
@tasks.loop(minutes=10)
async def check_inactive_users():
    now = datetime.now(timezone.utc)
    for guild in bot.guilds:
        for member in guild.members:
            if member.bot or member.id not in user_emotions: continue
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
    daily_affection_bonus.start()

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
    await bot.process_commands(message)
    if content.startswith(bot.command_prefix): return
    resp = await generate_a2_response(content, trust, uid)
    await message.channel.send(f"A2: {resp}")
    hist.append(f"A2: {resp}")
    if len(hist) > HISTORY_LIMIT * 2: hist.pop(0)
    save_data()

@bot.event
async def on_reaction_add(reaction, user):
    if user.bot: return
    uid = user.id
    if uid not in user_emotions:
        user_emotions[uid] = {"trust":0,"resentment":0,"attachment":0,
                              "guilt_triggered":False,"protectiveness":0,
                              "affection_points":0,
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

# ─── Commands ────────────────────────────────────────────────────────────────
@bot.command(name="affection", help="Show emotion stats for all users.")
async def affection_all(ctx):
    if not user_emotions:
        return await ctx.send("A2: Tch... no interactions yet.")
    lines = []
    for uid, e in user_emotions.items():
        member = bot.get_user(uid) or (ctx.guild and ctx.guild.get_member(uid))
        mention = member.mention if member else f"<@{uid}>"
        text = (
            f"**{mention}**\n"
            f"• Trust: {e['trust']}/10\n"
            f"• Attachment: {e['attachment']}/10\n"
            f"• Protectiveness: {e['protectiveness']}/10\n"
            f"• Resentment: {e['resentment']}/10\n"
            f"• Affection Points: {e['affection_points']}\n"
            f"• Guilt Triggered: {'Yes' if e['guilt_triggered'] else 'No'}"
        )
        lines.append(text)
    await ctx.send("A2: Current affection with all users:\n\n" + "\n\n".join(lines))

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

@bot.command(name="users", help="List tracked users.")
async def list_users(ctx):
    if not user_emotions:
        return await ctx.send("A2: Tch... none tracked.")
    mentions = []
    for uid in user_emotions:
        member = bot.get_user(uid) or (ctx.guild and ctx.guild.get_member(uid))
        mentions.append(member.mention if member else f"<@{uid}>")
    await ctx.send("A2: Tracked users:\n" + ", ".join(mentions))

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

@bot.command(name="ping", help="Ping the bot.")
async def ping(ctx):
    await ctx.send("Pong!")

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

if __name__ == "__main__":
    bot.run(DISCORD_BOT_TOKEN)
