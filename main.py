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

# ─── Local Transformers Pipeline Attempt ──────────────────────────────────────
HAVE_TRANSFORMERS = False
local_summarizer = None
local_toxic = None
local_sentiment = None
try:
    from transformers import pipeline
    HAVE_TRANSFORMERS = True
    # Initialize local pipelines
    local_summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )
    local_toxic = pipeline(
        "text-classification",
        model="unitary/toxic-bert",
        return_all_scores=True
    )
    local_sentiment = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
except ImportError:
    # Transformers not installed; fall back to OpenAI
    HAVE_TRANSFORMERS = False
finally:
    # Ensure variables exist
    locals_set = True

# ─── Dynamic Affection & Annoyance Settings ───────────────────────────────── ───────────────────────────────────────────────────────
DATA_FILE = Path("/mnt/railway/volume/data.json")  # persistent mount

def load_data():
    global user_emotions, conversation_history, conversation_summaries
    if DATA_FILE.exists():
        try:
            data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
            user_emotions          = data.get("user_emotions", {})
            conversation_history   = {int(k): v for k,v in data.get("conversation_history", {}).items()}
            conversation_summaries = data.get("conversation_summaries", {})
        except json.JSONDecodeError:
            user_emotions, conversation_history, conversation_summaries = {},{},{}
    else:
        user_emotions, conversation_history, conversation_summaries = {},{},{}
    for e in user_emotions.values(): e.setdefault("annoyance",0)

def save_data():
    DATA_FILE.parent.mkdir(parents=True,exist_ok=True)
    DATA_FILE.write_text(
        json.dumps({
            "user_emotions": user_emotions,
            "conversation_history": {str(k):v for k,v in conversation_history.items()},
            "conversation_summaries": conversation_summaries
        },indent=2,ensure_ascii=False),
        encoding="utf-8"
    )

# ─── Configuration & Client Setup ────────────────────────────────────────────
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN","")
DISCORD_APP_ID    = int(os.environ.get("DISCORD_APP_ID","0"))
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY","")
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
bot = commands.Bot(command_prefix="!", intents=intents, application_id=DISCORD_APP_ID)

# ─── State Containers ───────────────────────────────────────────────────────
user_emotions          = {}
conversation_history   = {}
conversation_summaries = {}
HISTORY_LIMIT          = 10
load_data()

# ─── Persona & Modifier Patterns ─────────────────────────────────────────────
A2_PERSONA = (
    "You are A2, a rogue android. Speak short, sarcastic, never break character."
)
reaction_modifiers = [
    (re.compile(r"\bi'll protect you\b", re.I),    {"trust":2,"protectiveness":1}),
    (re.compile(r"\bi miss you\b", re.I),          {"attachment":1,"trust":1}),
    (re.compile(r"\byou remind me of 2b\b",re.I),  {"trust":-2,"guilt_triggered":True}),
    (re.compile(r"\bwhy are you like this\b",re.I),{"trust":-1,"resentment":2}),
    (re.compile(r"\bwhatever\b|\bok\b|\bnevermind\b",re.I),{"trust":-1,"resentment":1}),
    (re.compile(r"\bi trust you\b",re.I),         {"trust":2}),
    (re.compile(r"\bi hate you\b",re.I),          {"resentment":3,"trust":-2}),
]
provoking_lines = [
    "Still mad? Good.",
    "You again? Tch.",
    "What?"
]
warm_lines      = ["...Checking in.","Still breathing?","Thought you got scrapped."]
insult_severity_patterns = reaction_modifiers  # repurpose for severity
# Sexual regex unchanged

# ─── Reaction & Logic-based Affection/Annoyance ─────────────────────────────
def apply_reaction_modifiers(content:str,user_id:int):
    if user_id not in user_emotions:
        user_emotions[user_id]={"trust":0,"resentment":0,"attachment":0,
                                 "guilt_triggered":False,"protectiveness":0,
                                 "affection_points":0,"annoyance":0,
                                 "last_interaction":datetime.now(timezone.utc).isoformat()}
    e=user_emotions[user_id]
    # trust/res modifiers
    for pat,eff in reaction_modifiers:
        if pat.search(content):
            for emo,val in eff.items():
                if emo=="guilt_triggered": e[emo]=True
                else: e[emo]=max(0,min(10,e[emo]+val))
    e["trust"]=min(10,e["trust"]+0.25)
    # annoyance: local toxic + existing sex logic
    inc=0
    try:
        scores=local_toxic(content)[0]
        for item in scores:
            if item["label"].lower() in ("insult","toxicity"):
                sev=int(item["score"]*10)
                inc=max(inc,min(10,max(1,sev)))
    except: pass
    # sexual bump logic unchanged...
    e["annoyance"]=min(100,e.get("annoyance",0)+inc)
    # affection: sentiment-based
    try:
        s=local_sentiment(content)[0]
        delta=int((s["score"]*(1 if s["label"]=="POSITIVE" else -1))*5)
    except:
        delta=0
    # scale by trust vs resentment
    factor=1+(e["trust"]-e.get("resentment",0))/20
    aff_delta=int(delta*factor)
    e["affection_points"]=max(-100,min(1000,e.get("affection_points",0)+aff_delta))
    e["last_interaction"]=datetime.now(timezone.utc).isoformat()
    save_data()

# ... rest of code unchanged ...

if __name__=="__main__":
    bot.run(DISCORD_BOT_TOKEN)
