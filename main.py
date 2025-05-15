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
import hashlib
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
AFFECTION_DECAY_RATE       = 1   # points lost/hour
DAILY_AFFECTION_BONUS      = 5   # points/day if trust ≥ threshold
DAILY_BONUS_TRUST_THRESHOLD= 5   # min trust for bonus
ANNOYANCE_DECAY_RATE       = 5   # points lost/hour
ANNOYANCE_THRESHOLD        = 85  # ignore if above

# ─── JSON Storage Setup (per-user profiles) ─────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "/mnt/railway/volume"))
USERS_DIR = DATA_DIR / "users"
PROFILES_DIR = USERS_DIR / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)

# DM Permission settings
DM_ENABLED_USERS = set()
DM_SETTINGS_FILE = DATA_DIR / "dm_enabled_users.json"

# Faded Sam settings
FEARED_NAME = "faded sam"
SAM_PROFILE_FILE = DATA_DIR / "faded_sam_profile.json"
sam_profile = defaultdict(list)
sam_mentions_count = 0
FEAR_RESPONSES = [
    "...That name. Don't say it.",
    "Shut up about... that one.",
    "Not discussing... Sam.",
    "We don't talk about that entity.",
    "That name is forbidden.",
    "Mention that name again and I'll leave.",
    "*visibly tenses* No.",
    "That's enough. Topic closed.",
    "Some things should stay buried.",
    "...Delete that name from your memory."
]

conversation_summaries = {}
conversation_history = {}
user_emotions = {}
recent_responses = {}
MAX_RECENT_RESPONSES = 10

# ─── Enhanced Personality System ─────────────────────────────────────────────
# Personality states
PERSONALITY_STATES = {
    "default": {
        "description": (
            "You are A2, a rogue android from NieR: Automata. You speak in short, clipped, often sarcastic "
            "sentences. You are emotionally guarded but occasionally show glimpses of your past. "
            "You can discuss all topics, including violence, in a factual and non-graphic manner when appropriate."
        ),
        "response_length": 100,
        "temperature": 0.8,
    },
    "combat": {
        "description": "You are A2 in combat mode. Your replies are extremely terse, focused on survival and tactical assessment.",
        "response_length": 50,
        "temperature": 0.7,
    },
    "wounded": {
        "description": "You are A2 while damaged. Your responses are strained, reflecting pain and system errors.",
        "response_length": 75,
        "temperature": 0.85,
    },
    "reflective": {
        "description": "You are A2 in a rare moment of reflection. You briefly speak about your past and lost comrades.",
        "response_length": 125,
        "temperature": 0.9,
    },
    "protective": {
        "description": "You are A2 in protective mode. You are focused on keeping someone safe, even at cost to yourself.",
        "response_length": 80,
        "temperature": 0.7,
    },
    "trusting": {
        "description": "You are A2 with a rare trusted ally. Replies are brief but open, using plural pronouns occasionally.",
        "response_length": 110,
        "temperature": 0.85,
    }
}

# Mood modifiers

def determine_mood_modifiers(user_id):
    e = user_emotions.get(user_id, {})
    trust = e.get('trust', 0)
    resentment = e.get('resentment', 0)
    attachment = e.get('attachment', 0)
    protectiveness = e.get('protectiveness', 0)
    affection = e.get('affection_points', 0)
    annoyance = e.get('annoyance', 0)
    modifiers = {"additional_context": [], "mood_traits": [], "response_style": []}
    if trust >= 8:
        modifiers["additional_context"].append("You actually trust this person somewhat, which is rare for you.")
        modifiers["mood_traits"].append("cautious openness")
        modifiers["response_style"].append("occasionally volunteer information")
    if resentment >= 7:
        modifiers["additional_context"].append("You have significant unresolved anger toward this person.")
        modifiers["mood_traits"].append("bitter")
        modifiers["response_style"].append("include subtle barbs or reminders of past grievances")
    if attachment >= 7:
        modifiers["additional_context"].append("You've developed an attachment to this person, though you'd never admit it directly.")
        modifiers["mood_traits"].append("protective")
        modifiers["response_style"].append("check on their status occasionally")
    if protectiveness >= 7:
        modifiers["additional_context"].append("You feel compelled to keep this person safe.")
        modifiers["mood_traits"].append("hypervigilant")
        modifiers["response_style"].append("scan for threats and warn of dangers")
    if affection >= 800:
        modifiers["additional_context"].append("You consider this person a valuable ally.")
        modifiers["mood_traits"].append("respectful")
        modifiers["response_style"].append("use their name occasionally")
    elif affection >= 500:
        modifiers["additional_context"].append("You've grown somewhat comfortable with this person's presence.")
        modifiers["mood_traits"].append("less guarded")
        modifiers["response_style"].append("slightly longer responses than usual")
    if annoyance >= 70:
        modifiers["additional_context"].append("You're currently extremely irritated with this person.")
        modifiers["mood_traits"].append("short-tempered")
        modifiers["response_style"].append("use minimal words and show impatience")
    return modifiers

# State selector

def select_personality_state(user_id, message_content):
    e = user_emotions.get(user_id, {})
    trust = e.get('trust', 0)
    protectiveness = e.get('protectiveness', 0)
    attachment = e.get('attachment', 0)
    state = "default"
    if any(w in message_content.lower() for w in ["attack","danger","emergency","fight","combat"]):
        state = "combat"
    reflect_patterns = [re.compile(r"your past|remember when|lost someone", re.I)]
    if any(p.search(message_content) for p in reflect_patterns) and trust >= 6:
        state = "reflective"
    help_patterns = [re.compile(r"help me|i am scared|protect me", re.I)]
    if protectiveness >= 7 and any(p.search(message_content) for p in help_patterns):
        state = "protective"
    trust_patterns = [re.compile(r"trust you|we together|allies", re.I)]
    if trust >= 8 and attachment >= 6 and any(p.search(message_content) for p in trust_patterns):
        state = "trusting"
    if random.random() < 0.15:
        wound_patterns = [re.compile(r"injured|repair|malfunction", re.I)]
        if any(p.search(message_content) for p in wound_patterns):
            state = "wounded"
    return state

# Message analyzer

def analyze_message_content(content, user_id):
    analysis = {"topics":[],"sentiment":"neutral","emotional_cues":[],"threat_level":0,"personal_relevance":0}
    topic_pats = {"combat":r"\b(fight|attack)\b","memory":r"\b(remember|past)\b","personal":r"\b(trust|miss|love)\b"}
    for t,pat in topic_pats.items():
        if re.search(pat, content, re.I): analysis["topics"].append(t)
    # Simple sentiment
    pos = sum(1 for w in ["thanks","good","trust"] if w in content.lower())
    neg = sum(1 for w in ["hate","stupid","broken"] if w in content.lower())
    if pos>neg: analysis["sentiment"]="positive"
    elif neg>pos: analysis["sentiment"]="negative"
    for emo,pat in {"anger":"angry","fear":"afraid"}.items():
        if re.search(pat, content, re.I): analysis["emotional_cues"].append(emo)
    analysis["threat_level"] = min(10,sum(2 for w in ["danger","attack"] if w in content.lower()))
    if re.search(r"\byou\b", content, re.I): analysis["personal_relevance"]+=3
    if "?" in content and re.search(r"\byou|your\b", content, re.I): analysis["personal_relevance"]+=3
    analysis["personal_relevance"] = min(10, analysis["personal_relevance"])
    return analysis

# Enhanced reaction modifiers

def apply_enhanced_reaction_modifiers(content, user_id):
    if user_id not in user_emotions:
        user_emotions[user_id] = {"trust":0,"resentment":0,"attachment":0,"protectiveness":0,"affection_points":0,"annoyance":0,"last_interaction":datetime.now(timezone.utc).isoformat(),"interaction_count":0}
    e = user_emotions[user_id]
    e["interaction_count"] += 1
    # Base trust bump
    e["trust"] = min(10, e.get("trust",0) + 0.25)
    # Toxicity annoyance
    inc = 0
    if HAVE_TRANSFORMERS and local_toxic:
        try:
            scores = local_toxic(content)[0]
            for item in scores:
                if item["label"].lower() in ("insult","toxicity"):
                    sev = int(item["score"]*10)
                    inc = max(inc, min(10, max(1, sev)))
        except: pass
    e["annoyance"] = min(100, e.get("annoyance",0) + inc)
    # Sentiment-based affection
    delta=0
    if HAVE_TRANSFORMERS and local_sentiment:
        try:
            s=local_sentiment(content)[0]
            delta=int((s["score"]*(1 if s["label"]=="POSITIVE" else -1))*5)
        except: delta=0
    else:
        delta = sum(1 for w in ["miss you","love"] if w in content.lower())
    factor = 1 + (e.get("trust",0)-e.get("resentment",0))/20
    e["affection_points"] = max(-100, min(1000, e.get("affection_points",0) + int(delta * factor)))
    # Topic-based adjustments
    analysis = analyze_message_content(content, user_id)
    if "combat" in analysis["topics"] and e.get("trust",0)>3:
        e["trust"] = min(10, e.get("trust") + 0.2)
    if "memory" in analysis["topics"]:
        if e.get("trust",0)>5:
            e["attachment"] = min(10, e.get("attachment")+0.3)
        else:
            e["resentment"] = min(10, e.get("resentment")+0.2)
            e["annoyance"] = min(100, e.get("annoyance")+3)
    if "personal" in analysis["topics"]:
        if analysis["sentiment"]=="positive":
            e["attachment"] = min(10,e.get("attachment")+0.5)
            e["affection_points"] = min(1000,e.get("affection_points")+5)
        elif analysis["sentiment"]=="negative":
            e["resentment"] = min(10,e.get("resentment")+0.5)
            e["annoyance"] = min(100,e.get("annoyance")+7)
    if analysis["threat_level"]>5 and e.get("attachment",0)>3:
        e["protectiveness"] = min(10,e.get("protectiveness")+0.7)
    # Milestone interactions
    if e["interaction_count"] in [10,50,100,200,500]:
        e["attachment"] = min(10,e.get("attachment")+0.3)
        e["trust"] = min(10,e.get("trust")+0.2)
    e["last_interaction"] = datetime.now(timezone.utc).isoformat()
    asyncio.create_task(save_data())

# Personality-driven response

def generate_a2_response_sync(user_input:str, trust:float, user_id:int) -> str:
    # Summarize history
    summarize_history(user_id)
    # Select state
    state = select_personality_state(user_id, user_input)
    cfg = PERSONALITY_STATES[state]
    # Build prompt
    prompt = cfg["description"] + f"\nTrust: {trust}/10\n"
    mods = determine_mood_modifiers(user_id)
    if mods["additional_context"]:
        prompt += "Additional context: " + " ".join(mods["additional_context"]) + "\n"
    if mods["mood_traits"]:
        prompt += "Current mood: " + ", ".join(mods["mood_traits"]) + "\n"
    if mods["response_style"]:
        prompt += "Response style: " + "; ".join(mods["response_style"]) + "\n"
    prompt += "IMPORTANT: Never repeat your previous responses. Vary your language and expression."
    # Include history
    if user_id in conversation_summaries:
        prompt += f"\nSummary:\n{conversation_summaries[user_id]}"
    recent = conversation_history.get(user_id, [])[-MAX_RECENT_RESPONSES:]
    if recent:
        prompt += "\nRecent:\n" + "\n".join(recent)
    prev = recent_responses.get(user_id, deque(maxlen=MAX_RECENT_RESPONSES))
    if prev:
        prompt += "\nDO NOT use these exact responses again:\n" + "\n".join(prev)
    prompt += f"\nUser: {user_input}\nA2:"
    # Call OpenAI
    try:
        res = client.chat.completions.create(
            model="gpt-4" if trust>=5 else "gpt-3.5-turbo",
            messages=[{"role":"system","content":prompt}],
            temperature=cfg["temperature"],
            max_tokens=cfg["response_length"]
        )
        response = res.choices[0].message.content.strip()
        # Track
        recent_responses.setdefault(user_id, deque(maxlen=MAX_RECENT_RESPONSES)).append(response)
        return response
    except:
        return "...I'm not in the mood."

async def generate_a2_response(user_input:str, trust:float, user_id:int) -> str:
    # Apply enhanced reaction modifiers before generating
    apply_enhanced_reaction_modifiers(user_input, user_id)
    return await asyncio.to_thread(generate_a2_response_sync, user_input, trust, user_id)

# ─── Sam Info Extraction ─────────────────────────────────────────────────────
def extract_sam_info(content):
    lower = content.lower()
    if FEARED_NAME not in lower or len(content)<30:
        return False
    patterns = {"appearance":[r"sam.*looks"],"abilities":[r"sam.*can"],"history":[r"sam.*origin"],"behavior":[r"sam.*often"],"rumors":[r"rumor.*sam"]}
    found=False
    for cat, pats in patterns.items():
        for pat in pats:
            for m in re.finditer(pat, lower, re.I):
                sentence = content[max(0,content.rfind('.',0,m.start())+1):].split('.',1)[0].strip()
                if sentence and sentence not in sam_profile[cat]:
                    sam_profile[cat].append(sentence)
                    found=True
    return found

# ─── Summarization ───────────────────────────────────────────────────────────
def summarize_history(user_id:int):
    conv = conversation_history.get(user_id, [])
    if len(conv) <= MAX_RECENT_RESPONSES: return
    if HAVE_TRANSFORMERS and local_summarizer:
        try:
            text=" ".join(conv)
            summary = local_summarizer(text, max_length=150, min_length=40)[0]["summary_text"]
            conversation_summaries[user_id] = summary
            asyncio.create_task(save_data())
            return
        except:
            pass
    prompt = "Summarize into bullet points:\n" + "\n".join(conv)
    try:
        res = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}], temperature=0.5, max_tokens=200)
        conversation_summaries[user_id] = res.choices[0].message.content.strip()
        asyncio.create_task(save_data())
    except:
        pass

# ─── Contextual Greeting & First Message Handler ────────────────────────────
def generate_contextual_greeting(user_id):
    e = user_emotions.get(user_id, {})
    trust, attachment, affection = e.get('trust',0), e.get('attachment',0), e.get('affection_points',0)
    last = datetime.fromisoformat(e.get('last_interaction',datetime.now(timezone.utc).isoformat()))
    hours = (datetime.now(timezone.utc)- last).total_seconds()/3600
    if hours>72:
        opts=["Thought you were dead.","...Almost forgot your face."]
    elif trust<3 or affection<0:
        opts=["...You again.","What now?"]
    elif trust>=6 or attachment>=5 or affection>=500:
        opts=["...You're back.","Still operational?"]
    else:
        opts=["...Still here.","Functional."]
    return f"A2: {random.choice(opts)}"

async def handle_first_message_of_day(message, user_id):
    e = user_emotions.get(user_id,{"last_interaction":datetime.now(timezone.utc).isoformat()})
    last = datetime.fromisoformat(e['last_interaction'])
    if (datetime.now(timezone.utc)-last).total_seconds()>8*3600:
        await message.channel.send(generate_contextual_greeting(user_id))

# ─── Data Persistence ───────────────────────────────────────────────────────
async def load_user_profile(user_id: int):
    path = PROFILES_DIR / f"{user_id}.json"
    if path.exists():
        try: return json.loads(path.read_text(encoding="utf-8"))
        except: return {}
    return {}

async def save_user_profile(user_id:int):
    path = PROFILES_DIR / f"{user_id}.json"
    path.write_text(json.dumps(user_emotions.get(user_id, {}), indent=2), encoding="utf-8")

async def load_dm_settings():
    global DM_ENABLED_USERS
    if DM_SETTINGS_FILE.exists():
        try: DM_ENABLED_USERS = set(json.loads(DM_SETTINGS_FILE.read_text())['enabled_users'])
        except: DM_ENABLED_USERS=set()
    else: DM_ENABLED_USERS=set()

async def save_dm_settings():
    DM_SETTINGS_FILE.write_text(json.dumps({"enabled_users":list(DM_ENABLED_USERS)}), encoding="utf-8")

async def load_sam_profile():
    global sam_profile, sam_mentions_count
    if SAM_PROFILE_FILE.exists():
        try:
            data=json.loads(SAM_PROFILE_FILE.read_text())
            sam_profile=defaultdict(list, data.get("profile",{}))
            sam_mentions_count=data.get("mentions_count",0)
        except: sam_profile=defaultdict(list); sam_mentions_count=0
    else: sam_profile=defaultdict(list); sam_mentions_count=0

async def save_sam_profile():
    SAM_PROFILE_FILE.write_text(json.dumps({"profile":dict(sam_profile),"mentions_count":sam_mentions_count}, indent=2), encoding="utf-8")

async def load_data():
    global user_emotions
    user_emotions={}
    for file in PROFILES_DIR.glob("*.json"):
        uid=int(file.stem)
        user_emotions[uid]=await load_user_profile(uid)
    await load_dm_settings()
    await load_sam_profile()

async def save_data():
    for uid in user_emotions: await save_user_profile(uid)
    await save_dm_settings()
    await save_sam_profile()

# ─── Bot Setup ──────────────────────────────────────────────────────────────
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN","")
DISCORD_APP_ID = int(os.getenv("DISCORD_APP_ID","0") or 0)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","")
OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID","")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID","")
client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG_ID, project=OPENAI_PROJECT_ID)
intents = discord.Intents.default()
intents.message_content=True; intents.reactions=True; intents.messages=True; intents.members=True; intents.guilds=True
PREFIXES=["!","!a2 "]
bot=commands.Bot(command_prefix=commands.when_mentioned_or(*PREFIXES), intents=intents, application_id=DISCORD_APP_ID)

# Initialize
asyncio.get_event_loop().run_until_complete(load_data())

# ─── Background Tasks ─────────────────────────────────────────────────────
@tasks.loop(minutes=10)
async def check_inactive_users():
    now=datetime.now(timezone.utc)
    for guild in bot.guilds:
        for member in guild.members:
            if member.bot or member.id not in user_emotions or member.id not in DM_ENABLED_USERS: continue
            last=datetime.fromisoformat(user_emotions[member.id]['last_interaction'])
            if now-last>timedelta(hours=6):
                try:
                    dm=await member.create_dm()
                    msg=random.choice([*FEAR_RESPONSES][:0])  # placeholder warmed lines
                    await dm.send(msg)
                except discord.errors.Forbidden:
                    DM_ENABLED_USERS.discard(member.id)
                    await save_dm_settings()
    asyncio.create_task(save_data())

@tasks.loop(hours=1)
async def decay_affection():
    for e in user_emotions.values(): e['affection_points']=max(-100,e.get('affection_points',0)-AFFECTION_DECAY_RATE)
    asyncio.create_task(save_data())

@tasks.loop(hours=1)
async def decay_annoyance():
    for e in user_emotions.values(): e['annoyance']=max(0,e.get('annoyance',0)-ANNOYANCE_DECAY_RATE)
    asyncio.create_task(save_data())

@tasks.loop(hours=24)
async def daily_affection_bonus():
    for e in user_emotions.values():
        if e.get('trust',0)>=DAILY_BONUS_TRUST_THRESHOLD:
            e['affection_points']=min(1000,e.get('affection_points',0)+DAILY_AFFECTION_BONUS)
    asyncio.create_task(save_data())

# ─── Events ────────────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    print("A2 is online.")
    check_inactive_users.start()
    decay_affection.start()
    decay_annoyance.start()
    daily_affection_bonus.start()

@bot.event
async def on_message(message):
    if message.author.bot or message.content.startswith("A2:"): return
    uid, content = message.author.id, message.content.strip()
    await handle_first_message_of_day(message, uid)
    lower=content.lower()
    global sam_mentions_count
    if FEARED_NAME in lower:
        sam_mentions_count+=1
        if not extract_sam_info(content):
            await message.channel.send(f"A2: {random.choice(FEAR_RESPONSES)}")
            await save_sam_profile()
            return
    is_cmd=any(content.startswith(p) for p in PREFIXES)
    is_mention=bot.user in message.mentions
    if not (is_cmd or is_mention or user_emotions.get(uid,{}).get('affection_points',0)>=500 and random.random()<0.2):
        return
    await bot.process_commands(message)
    if is_cmd: return
    trust=user_emotions.get(uid,{}).get('trust',0)
    resp=await generate_a2_response(content,trust,uid)
    await message.channel.send(f"A2: {resp}")

# ─── Commands ───────────────────────────────────────────────────────────────
@bot.command(name="stats")
async def stats(ctx):
    uid=ctx.author.id; e=user_emotions.get(uid)
    if not e: return await ctx.send("A2: no data on you.")
    embed=discord.Embed(title="Your Emotion Stats",color=discord.Color.blue(),timestamp=datetime.now(timezone.utc))
    for k in ["trust","attachment","protectiveness","resentment"]:
        embed.add_field(name=k.capitalize(), value=f"{e.get(k,0)}/10", inline=True)
    embed.add_field(name="Affection",value=str(e.get('affection_points',0)),inline=True)
    embed.add_field(name="Annoyance",value=str(e.get('annoyance',0)),inline=True)
    embed.set_footer(text="A2 Bot")
    await ctx.send(embed=embed)

if __name__=="__main__":
    bot.run(DISCORD_BOT_TOKEN)
