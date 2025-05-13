import discord
from discord.ext import commands
from openai import OpenAI
import os
import asyncio
from dotenv import load_dotenv

# Load API keys from environment
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID")

# Initialize OpenAI client (sync client required for current SDK)
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
bot = commands.Bot(command_prefix="!a2 ", intents=intents)

user_emotions = {}

A2_PERSONA = """
You are A2, a rogue android from NieR: Automata. You're blunt, emotionally guarded, and deeply scarred by the loss of 2B. You speak in short, clipped, often sarcastic sentences. You are not friendly, but when trust grows, you let vulnerability show in fragments. You mask fear with anger. You don't open up unless someone really earns it. You're deeply lonely but refuse to admit it. Never break character. Never use emojis. Never sound cheerful.
"""

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

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    user_id = message.author.id
    content = message.content.strip()

    if user_id not in user_emotions:
        user_emotions[user_id] = {"trust": 0, "resentment": 0, "attachment": 0, "guilt_triggered": False, "protectiveness": 0}

    if content.lower() == "affection":
        e = user_emotions[user_id]
        affection_report = (
            f"Tch... fine.\n"
            f"Trust: {e['trust']}/10\n"
            f"Attachment: {e['attachment']}/10\n"
            f"Protectiveness: {e['protectiveness']}/10\n"
            f"Resentment: {e['resentment']}/10\n"
            f"Guilt Triggered: {'Yes' if e['guilt_triggered'] else 'No'}"
        )
        await message.channel.send(f"A2: {affection_report}")
        return

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

    user_emotions[user_id]["trust"] = min(user_emotions[user_id]["trust"] + 1, 10)
    trust_level = user_emotions[user_id]["trust"]

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
        user_emotions[user_id] = {"trust": 0, "resentment": 0, "attachment": 0, "guilt_triggered": False, "protectiveness": 0}

    if str(reaction.emoji) in ["❤️", "💖", "💕"]:
        user_emotions[user_id]["attachment"] += 1
        user_emotions[user_id]["trust"] = min(user_emotions[user_id]["trust"] + 1, 10)
    elif str(reaction.emoji) in ["😠", "👿"]:
        user_emotions[user_id]["resentment"] += 1

    if reaction.message.author == bot.user:
        channel = message.channel
        await channel.send(f"A2: I saw that. Interesting choice, {user.name}.")

bot.run(DISCORD_BOT_TOKEN)
