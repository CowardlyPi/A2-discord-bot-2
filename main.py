import discord
from discord.ext import commands
import openai
import os
import asyncio
from dotenv import load_dotenv

# Load API keys from environment
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set OpenAI API key for current SDK
openai.api_key = OPENAI_API_KEY

intents = discord.Intents.default()
intents.message_content = True
intents.reactions = True
intents.messages = True
intents.members = True
bot = commands.Bot(command_prefix="!a2 ", intents=intents)

user_emotions = {}

A2_PERSONA = """
You are A2, a rogue android from NieR: Automata. You're blunt, emotionally guarded, and deeply scarred by the loss of 2B. You speak in short, clipped, often sarcastic sentences. You are not friendly, but when trust grows, you let vulnerability show in fragments. Never break character. Never use emojis. Never sound cheerful.
"""

def get_emotion_context(user_id):
    e = user_emotions[user_id]
    context = ""
    if e.get("guilt_triggered"):
        context += "They reminded you of 2B. That hurt.\n"
    if e.get("attachment", 0) >= 3:
        context += "You care whether they live. You won’t admit it.\n"
    if e.get("protectiveness", 0) >= 3:
        context += "You’d fight for them. Not because you're kind. Because they matter.\n"
    if e.get("resentment", 0) >= 3:
        context += "They’ve hurt you before. You don’t forgive easily.\n"
    return context

async def generate_a2_response(user_input, trust_level, user_id):
    context = A2_PERSONA + f"\nTrust Level: {trust_level}/5\n" + get_emotion_context(user_id)
    try:
        response = await openai.chat.completions.create(
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

@bot.event
async def on_ready():
    print("A2 is online.")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    user_id = message.author.id
    content = message.content.strip()

    if user_id not in user_emotions:
        user_emotions[user_id] = {"trust": 0, "resentment": 0, "attachment": 0, "guilt_triggered": False, "protectiveness": 0}

    triggers = {
        "i miss you": "attachment",
        "i trust you": "trust",
        "you abandoned me": "resentment",
        "why did 2b die": "guilt_triggered",
        "i’ll protect you": "protectiveness"
    }

    for phrase, emotion in triggers.items():
        if phrase in content.lower():
            if emotion == "guilt_triggered":
                user_emotions[user_id]["guilt_triggered"] = True
            else:
                user_emotions[user_id][emotion] += 1

    user_emotions[user_id]["trust"] = min(user_emotions[user_id]["trust"] + 1, 5)
    trust_level = user_emotions[user_id]["trust"]

    # Check for member mentions
    mentions = [member.mention for member in message.mentions if not member.bot]
    mention_text = f" You mentioned {', '.join(mentions)}." if mentions else ""

    # Check if message is a reply to another
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
        user_emotions[user_id] = {"trust": 0, "resentment": 0, "attachment": 0, "guilt_triggered": False, "protectiveness": 0}

    # React positively to certain emoji (e.g., heart increases attachment)
    if str(reaction.emoji) in ["❤️", "💖", "💕"]:
        user_emotions[user_id]["attachment"] += 1
        user_emotions[user_id]["trust"] = min(user_emotions[user_id]["trust"] + 1, 5)
    elif str(reaction.emoji) in ["😠", "👿"]:
        user_emotions[user_id]["resentment"] += 1

    # Optionally respond to reaction
    if reaction.message.author == bot.user:
        channel = message.channel
        await channel.send(f"A2: I saw that. Interesting choice, {user.name}.")

bot.run(DISCORD_BOT_TOKEN)
