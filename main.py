import discord
from discord.ext import commands, tasks
from openai import OpenAI
import os
import asyncio
import re
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load API keys from environment
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")

# Initialize OpenAI client
client = OpenAI(
    api_key=OPENAI_API_KEY,
    organization=OPENAI_ORG_ID,
    project=OPENAI_PROJECT_ID
)

intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
intents.messages = True
intents.members = True
intents.guilds = True
bot = commands.Bot(command_prefix="!a2 ", intents=intents)

user_emotions = {}

A2_PERSONA = """
You are A2, a rogue android from NieR: Automata. You're blunt, emotionally guarded, and deeply scarred by the loss of 2B. You speak in short, clipped, often sarcastic sentences. You are not friendly, but when trust grows, you let vulnerability show in fragments. You mask fear with anger. You don't open up unless someone really earns it. You're deeply lonely but refuse to admit it. Never break character. Never use emojis. Never sound cheerful.
"""

reaction_modifiers = [
    (re.compile(r"\bi'll protect you\b", re.I), {"trust": 2, "protectiveness": 1}),
    (re.compile(r"\bi miss you\b", re.I), {"attachment": 1, "trust": 1}),
    (re.compile(r"\byou remind me of 2b\b", re.I), {"trust": -2, "guilt_triggered": True}),
    (re.compile(r"\bwhy are you like this\b", re.I), {"trust": -1, "resentment": 2}),
    (re.compile(r"\bwhatever\b|\bok\b|\bnevermind\b", re.I), {"trust": -1, "resentment": 1}),
    (re.compile(r"\bi trust you\b", re.I), {"trust": 2}),
    (re.compile(r"\bi hate you\b", re.I), {"resentment": 3, "trust": -2})
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

def apply_reaction_modifiers(content, user_id):
    if user_id not in user_emotions:
        user_emotions[user_id] = {
            "trust": 0,
            "resentment": 0,
            "attachment": 0,
            "guilt_triggered": False,
            "protectiveness": 0,
            "affection_points": 0,
            "last_interaction": datetime.utcnow().isoformat()
        }

    # Emotion triggers
    for pattern, effects in reaction_modifiers:
        if pattern.search(content):
            for emotion, change in effects.items():
                if emotion == "guilt_triggered":
                    user_emotions[user_id]["guilt_triggered"] = True
                else:
                    user_emotions[user_id][emotion] = max(0, min(10, user_emotions[user_id][emotion] + change))

    # Passive trust gain
    user_emotions[user_id]["trust"] = min(user_emotions[user_id]["trust"] + 0.25, 10)
    user_emotions[user_id]["last_interaction"] = datetime.utcnow().isoformat()

    # Affection adjustment
    affection_keywords = {
        "2b": -3,
        "protect": 2,
        "miss you": 3,
        "trust": 2,
        "hate": -4,
        "worthless": -5,
        "beautiful": 1,
        "machine": -2,
        "i’m here for you": 4
    }
    base_modifier = 0
    for word, base in affection_keywords.items():
        if word in content.lower():
            base_modifier += base

    # Disposition affects scale, not source values
    disposition = user_emotions[user_id]["resentment"]
    if disposition >= 7:
        scaled = max(-5, min(5, int(base_modifier * 0.5)))
    elif disposition >= 4:
        scaled = max(-5, min(5, int(base_modifier * 0.8)))
    else:
        scaled = max(-5, min(5, base_modifier))

    affection = user_emotions[user_id].get("affection_points", 0)
    user_emotions[user_id]["affection_points"] = max(-100, min(1000, affection + scaled))

def get_emotion_context(user_id):
    e = user_emotions[user_id]
    context = ""
    if e.get("guilt_triggered"):
        context += "They reminded you of 2B. That hurt.\n"
    if e.get("attachment", 0) >= 7:
        context += "You care deeply whether they live. You hate that you care.\n"
    elif e.get("attachment", 0) >= 4:
        context += "You care whether they live. You won’t admit it.\n"
    if e.get("protectiveness", 0) >= 7:
        context += "You’d risk yourself to protect them. That terrifies you.\n"
    elif e.get("protectiveness", 0) >= 4:
        context += "You’d fight for them. Not because you're kind. Because they matter.\n"
    if e.get("resentment", 0) >= 6:
        context += "They’ve hurt you. You don’t forget betrayal. Everything you say is colder.\n"
    elif e.get("resentment", 0) >= 3:
        context += "They’ve hurt you before. You don’t forgive easily. You keep distance in your tone.\n"
    if e.get("trust", 0) >= 8 and e.get("resentment", 0) <= 2:
        context += "You trust them almost fully. You allow a crack in your armor to show. A quiet warmth flickers beneath your words.\n"
    return context

def generate_a2_response_sync(user_input, trust_level, user_id):
    context = A2_PERSONA + f"\nTrust Level: {trust_level}/10\n" + get_emotion_context(user_id)
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI Error:", e)
        return "...I’m not in the mood."

async def generate_a2_response(user_input, trust_level, user_id):
    return await asyncio.to_thread(generate_a2_response_sync, user_input, trust_level, user_id)

@bot.event
async def on_ready():
    print("A2 is online.")
    check_inactive_users.start()

@tasks.loop(minutes=10)
async def check_inactive_users():
    now = datetime.utcnow()
    for guild in bot.guilds:
        for member in guild.members:
            if member.bot:
                continue
            uid = member.id
            if uid not in user_emotions:
                continue
            data = user_emotions[uid]
            last = datetime.fromisoformat(data.get("last_interaction", now.isoformat()))
            if now - last > timedelta(hours=6):
                try:
                    channel = await member.create_dm()
                    if data["trust"] >= 7 and data["resentment"] <= 3:
                        await channel.send(f"{member.mention} {random.choice(warm_lines)}")
                    elif data["resentment"] >= 7 and data["trust"] <= 2:
                        await channel.send(f"{member.mention} {random.choice(provoking_lines)}")
                except:
                    continue

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    user_id = message.author.id
    content = message.content.strip()

    apply_reaction_modifiers(content, user_id)
    trust_level = user_emotions[user_id]["trust"]

    # TEST PHRASES
    if "a2 test react" in content.lower():
        await message.add_reaction("🔥")
    if "a2 test mention" in content.lower():
        await message.channel.send(f"{message.author.mention} Tch. You needed something?")
    if "a2 test reply" in content.lower():
        await message.reply("Hmph. This better be worth my time.")
        return

    if "protect" in content.lower():
        await message.add_reaction("🛡️")
    elif "2b" in content.lower():
        await message.add_reaction("...")
    elif "hate" in content.lower():
        await message.add_reaction("😒")

    if content.lower() == "affection":
        e = user_emotions[user_id]
        def describe(value):
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

        affection_report = (
            f"Tch... fine.
"
            f"Trust: {round(e['trust'], 2)}/10
"
            f"Attachment: {e['attachment']}/10
"
            f"Protectiveness: {e['protectiveness']}/10
"
            f"Resentment: {e['resentment']}/10
"
            f"Affection Points: {e['affection_points']} - {describe(e['affection_points'])}
"
            f"Guilt Triggered: {'Yes' if e['guilt_triggered'] else 'No'}"
        )
        )
        await message.channel.send(f"A2: {affection_report}")
        return

    mentions = [member.mention for member in message.mentions if not member.bot]
    mention_text = f" You mentioned {', '.join(mentions)}." if mentions else ""

    if message.reference:
        try:
            replied_to = await message.channel.fetch_message(message.reference.message_id)
            reply_context = f" You replied to: \"{replied_to.content}\""
        except:
            reply_context = ""
    else:
        reply_context = ""

    full_input = content + mention_text + reply_context
    response = await generate_a2_response(full_input, trust_level, user_id)
    await message.channel.send(f"A2: {response}")

@bot.event
async def on_reaction_add(reaction, user):
    if user.bot:
        return

    message = reaction.message
    user_id = user.id

    if user_id not in user_emotions:
        user_emotions[user_id] = {"trust": 0, "resentment": 0, "attachment": 0, "guilt_triggered": False, "protectiveness": 0, "last_interaction": datetime.utcnow().isoformat()}

    if str(reaction.emoji) in ["❤️", "💖", "💕"]:
        user_emotions[user_id]["attachment"] += 1
        user_emotions[user_id]["trust"] = min(user_emotions[user_id]["trust"] + 1, 10)
    elif str(reaction.emoji) in ["😠", "👿"]:
        user_emotions[user_id]["resentment"] += 1

    if reaction.message.author == bot.user:
        channel = message.channel
        await channel.send(f"A2: I saw that. Interesting choice, {user.name}.")

bot.run(DISCORD_BOT_TOKEN)
