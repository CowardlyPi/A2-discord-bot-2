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

# Optional local pipelines
try:
    from transformers import pipeline
    HAVE_TRANSFORMERS = True
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    toxic_model = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
    sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except ImportError:
    HAVE_TRANSFORMERS = False
    summarizer = toxic_model = sentiment_model = None

# Affection & Annoyance settings
AFFECTION_DECAY_RATE    = 1
DAILY_AFFECTION_BONUS    = 5
DAILY_BONUS_TRUST_THRESHOLD = 5
ANNOYANCE_DECAY_RATE     = 5
ANNOYANCE_THRESHOLD      = 85

# JSON storage
DATA_FILE = Path(os.getenv('DATA_DIR', '/mnt/railway/volume')) / 'data.json'
def load_data():
    global user_emotions, conversation_history, conversation_summaries
    if DATA_FILE.exists():
        data = json.loads(DATA_FILE.read_text(encoding='utf-8'))
        user_emotions = data.get('user_emotions', {})
        conversation_history = {int(k): v for k, v in data.get('conversation_history', {}).items()}
        conversation_summaries = data.get('conversation_summaries', {})
    else:
        user_emotions, conversation_history, conversation_summaries = {}, {}, {}
    for e in user_emotions.values(): e.setdefault('annoyance', 0)

def save_data():
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(
        json.dumps({
            'user_emotions': user_emotions,
            'conversation_history': {str(k): v for k, v in conversation_history.items()},
            'conversation_summaries': conversation_summaries
        }, indent=2),
        encoding='utf-8'
    )

# Configuration
DISCORD_BOT_TOKEN = os.environ['DISCORD_BOT_TOKEN']
DISCORD_APP_ID    = int(os.environ.get('DISCORD_APP_ID', '0'))
OPENAI_API_KEY    = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=OPENAI_API_KEY)

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents, application_id=DISCORD_APP_ID)
# Load data
user_emotions = {}
conversation_history = {}
conversation_summaries = {}
load_data()

# Commands
@bot.command(name='ping')
async def ping(ctx):
    await ctx.send('Pong!')

@bot.command(name='stats')
async def stats(ctx):
    uid = ctx.author.id
    e = user_emotions.get(uid)
    if not e:
        return await ctx.send('No data.')
    await ctx.send(f"Trust: {e['trust']}/10, Affection: {e['affection_points']}, Annoyance: {e['annoyance']}")

# Tasks
@tasks.loop(hours=1)
async def decay():
    for e in user_emotions.values():
        e['affection_points'] = max(-100, e['affection_points'] - AFFECTION_DECAY_RATE)
        e['annoyance'] = max(0, e['annoyance'] - ANNOYANCE_DECAY_RATE)
    save_data()

@tasks.loop(hours=24)
async def daily_bonus():
    for e in user_emotions.values():
        if e['trust'] >= DAILY_BONUS_TRUST_THRESHOLD:
            e['affection_points'] = min(1000, e['affection_points'] + DAILY_AFFECTION_BONUS)
    save_data()

# Events
@bot.event
async def on_ready():
    print('A2 is online')
    decay.start()
    daily_bonus.start()

@bot.event
async def on_message(message):
    if message.author.bot: return
    uid, content = message.author.id, message.content
    # init
    e = user_emotions.setdefault(uid, {'trust':0,'resentment':0,'attachment':0,'affection_points':0,'annoyance':0,'last_interaction':datetime.now(timezone.utc).isoformat()})
    # process commands
    await bot.process_commands(message)

@bot.event
async def on_reaction_add(reaction, user):
    if user.bot: return
    # simple reaction logger
    print(f'Reaction {reaction.emoji} by {user.name}')

if __name__ == '__main__':
    bot.run(DISCORD_BOT_TOKEN)
