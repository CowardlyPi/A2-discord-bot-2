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
DM_ENABLED_USERS = set()  # Store user IDs who have enabled DMs
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
MAX_RECENT_RESPONSES = 10  # How many recent responses to remember per user

async def load_user_profile(user_id: int):
    path = PROFILES_DIR / f"{user_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}

async def save_user_profile(user_id: int):
    path = PROFILES_DIR / f"{user_id}.json"
    profile = user_emotions.get(user_id, {})
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")

async def load_dm_settings():
    global DM_ENABLED_USERS
    if DM_SETTINGS_FILE.exists():
        try:
            data = json.loads(DM_SETTINGS_FILE.read_text(encoding="utf-8"))
            DM_ENABLED_USERS = set(data.get("enabled_users", []))
        except json.JSONDecodeError:
            DM_ENABLED_USERS = set()
    else:
        DM_ENABLED_USERS = set()

async def save_dm_settings():
    data = {"enabled_users": list(DM_ENABLED_USERS)}
    DM_SETTINGS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

async def load_sam_profile():
    global sam_profile, sam_mentions_count
    if SAM_PROFILE_FILE.exists():
        try:
            data = json.loads(SAM_PROFILE_FILE.read_text(encoding="utf-8"))
            sam_profile = defaultdict(list)
            for category, entries in data.get("profile", {}).items():
                sam_profile[category] = entries
            sam_mentions_count = data.get("mentions_count", 0)
        except json.JSONDecodeError:
            sam_profile = defaultdict(list)
            sam_mentions_count = 0
    else:
        sam_profile = defaultdict(list)
        sam_mentions_count = 0

async def save_sam_profile():
    data = {
        "profile": dict(sam_profile),
        "mentions_count": sam_mentions_count
    }
    SAM_PROFILE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

async def load_data():
    global user_emotions
    user_emotions = {}
    for file in PROFILES_DIR.glob("*.json"):
        uid = int(file.stem)
        user_emotions[uid] = await load_user_profile(uid)
    
    # Load DM settings and Sam profile
    await load_dm_settings()
    await load_sam_profile()

async def save_data():
    for uid in list(user_emotions.keys()):
        await save_user_profile(uid)
    
    # Save DM settings and Sam profile
    await save_dm_settings()
    await save_sam_profile()

# ─── Configuration & State ───────────────────────────────────────────────────
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_APP_ID    = int(os.environ.get("DISCORD_APP_ID", "0") or 0)
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")
OPENAI_ORG_ID     = os.environ.get("OPENAI_ORG_ID", "")
OPENAI_PROJECT_ID = os.environ.get("OPENAI_PROJECT_ID", "")

client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG_ID, project=OPENAI_PROJECT_ID)

intents = discord.Intents.default()
intents.message_content = True
intents.reactions       = True
intents.messages        = True
intents.members         = True
intents.guilds          = True
PREFIXES = ["!", "!a2 "]
command_prefix = commands.when_mentioned_or(*PREFIXES)
bot = commands.Bot(command_prefix=command_prefix, intents=intents, application_id=DISCORD_APP_ID)

# ─── Per-user State & Utilities ─────────────────────────────────────────────
HISTORY_LIMIT          = 10
asyncio.get_event_loop().run_until_complete(load_data())

# ─── Persona & Modifiers ─────────────────────────────────────────────────────
A2_PERSONA = (
    "You are A2, a rogue android from NieR: Automata. You speak in short, clipped, often sarcastic "
    "sentences. You are emotionally guarded and never break character."
    " You can discuss all topics, including violence, in a factual and non-graphic manner when appropriate."
)
reaction_modifiers = [
    (re.compile(r"\bi'll protect you\b", re.I),    {"trust":2,"protectiveness":1}),
    (re.compile(r"\bi miss you\b", re.I),          {"attachment":1,"trust":1}),
    (re.compile(r"\bhate you\b", re.I),            {"resentment":3,"trust":-2}),
]
provoking_lines = [
    "Still mad? Good.", 
    "You again? Tch.", 
    "What?",
    "Need something?",
    "Don't waste my time.",
    "...This better be important.",
    "Speak or walk away.",
    "Not in the mood for games.",
    "Is this necessary?",
    "I've got things to do.",
    "Make it quick."
]

warm_lines = [
    "...Checking in.", 
    "Still breathing?", 
    "Thought you got scrapped.",
    "You're still functional. Good.",
    "Just making sure you're alive.",
    "Been quiet. Status report?",
    "Maintaining comm link.",
    "Required check-in.",
    "Vital signs stable?",
    "Survived another day, I see.",
    "You're harder to kill than expected."
]

# ─── Helper: Should Respond Logic ───────────────────────────────────────────
def should_respond_to(content: str, uid: int, is_cmd: bool, is_mention: bool) -> bool:
    affection = user_emotions.get(uid, {}).get('affection_points', 0)
    if is_cmd or is_mention:
        return True
    if affection >= 800:
        return True
    if affection >= 500:
        return random.random() < 0.2
    return False

# ─── Emotion & Annoyance Tracking ───────────────────────────────────────────
def apply_reaction_modifiers(content: str, user_id: int):
    if user_id not in user_emotions:
        user_emotions[user_id] = {
            "trust":0,"resentment":0,"attachment":0,
            "guilt_triggered":False,"protectiveness":0,
            "affection_points":0,"annoyance":0,
            "last_interaction":datetime.now(timezone.utc).isoformat()
        }
    e = user_emotions[user_id]
    for pat, effects in reaction_modifiers:
        if pat.search(content):
            for emo, val in effects.items():
                if emo == "guilt_triggered": e[emo] = True
                else: e[emo] = max(0, min(10, e.get(emo,0)+val))
    e["trust"] = min(10, e.get("trust",0)+0.25)
    inc=0
    if HAVE_TRANSFORMERS and local_toxic:
        try:
            scores=local_toxic(content)[0]
            for item in scores:
                if item["label"].lower() in ("insult","toxicity"):
                    sev=int(item["score"]*10)
                    inc=max(inc,min(10,max(1,sev)))
        except: pass
    else:
        for pat,_ in reaction_modifiers:
            if pat.search(content): inc=max(inc,1)
    e["annoyance"]=min(100,e.get("annoyance",0)+inc)
    if HAVE_TRANSFORMERS and local_sentiment:
        try:
            s=local_sentiment(content)[0]
            delta=int((s["score"]*(1 if s["label"]=="POSITIVE" else -1))*5)
        except: delta=0
    else:
        delta=sum(1 for w in ["miss you","support","love"] if w in content.lower())
    factor=1+(e.get("trust",0)-e.get("resentment",0))/20
    e["affection_points"]=max(-100,min(1000,e.get("affection_points",0)+int(delta*factor)))
    e["last_interaction"]=datetime.now(timezone.utc).isoformat()
    asyncio.create_task(save_data())

# ─── Summarization ───────────────────────────────────────────────────────────
def summarize_history(user_id:int):
    raw=conversation_history.get(user_id,[])
    if len(raw)>HISTORY_LIMIT:
        if HAVE_TRANSFORMERS and local_summarizer:
            try:
                text=" ".join(raw)
                summary=local_summarizer(text,max_length=150,min_length=40)[0]["summary_text"]
                conversation_summaries[user_id]=summary
                asyncio.create_task(save_data());return
            except: pass
        prompt="Summarize into bullet points under 200 tokens:\n"+"\n".join(raw)
        try:
            res=client.chat.completions.create(model="gpt-3.5-turbo",
                messages=[{"role":"user","content":prompt}],temperature=0.5,max_tokens=200)
            conversation_summaries[user_id]=res.choices[0].message.content.strip()
            asyncio.create_task(save_data())
        except: pass

# ─── Faded Sam Analysis ───────────────────────────────────────────────────────
def extract_sam_info(content):
    # Skip if it's a direct fear reaction
    lower_content = content.lower()
    if FEARED_NAME in lower_content and len(lower_content) < 30:
        return False
    
    # Skip if it doesn't mention Sam
    if FEARED_NAME not in lower_content:
        return False
    
    # Extract information about Sam based on patterns
    patterns = {
        "appearance": [
            r"(?:faded sam|sam).{1,30}(?:looks|appears|wears|dressed)",
            r"(?:faded sam|sam).{1,30}(?:tall|short|big|small|thin|fat)"
        ],
        "abilities": [
            r"(?:faded sam|sam).{1,30}(?:can|able to|powers|abilities)",
            r"(?:faded sam|sam).{1,30}(?:control|manipulate|create|destroy)"
        ],
        "history": [
            r"(?:faded sam|sam).{1,30}(?:came from|origin|history|background|past)",
            r"(?:faded sam|sam).{1,30}(?:used to|once|before|previously)"
        ],
        "behavior": [
            r"(?:faded sam|sam).{1,30}(?:always|never|sometimes|often|usually|likes to|hates)",
            r"(?:faded sam|sam).{1,30}(?:personality|behavior|attitude|temperament)"
        ],
        "rumors": [
            r"(?:heard|rumor|they say|people say).{1,40}(?:faded sam|sam)",
            r"(?:faded sam|sam).{1,30}(?:supposedly|allegedly|apparently|might|could|would)"
        ]
    }
    
    found_info = False
    for category, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.finditer(pattern, lower_content, re.IGNORECASE)
            for match in matches:
                # Get the sentence containing the match
                start = max(0, lower_content.rfind(".", 0, match.start()) + 1)
                end = lower_content.find(".", match.end())
                if end == -1:
                    end = len(lower_content)
                
                sentence = content[start:end].strip()
                if sentence and len(sentence) > 10:  # Ensure it's substantial
                    # Avoid duplicates
                    if sentence not in sam_profile[category]:
                        sam_profile[category].append(sentence)
                        found_info = True
    
    return found_info

# ─── A2 Response ─────────────────────────────────────────────────────────────
def generate_a2_response_sync(user_input:str, trust:float, user_id:int)->str:
    summarize_history(user_id)
    model = "gpt-3.5-turbo" if trust < 5 else "gpt-4"
    prompt = A2_PERSONA + f"\nTrust: {trust}/10\n"
    
    # Add explicit instruction against repetition
    prompt += "IMPORTANT: Never repeat your previous responses. Vary your language and expression. Try to be unpredictable.\n"
    
    if user_id in conversation_summaries: 
        prompt += f"Summary:\n{conversation_summaries[user_id]}\n"
    
    recent = conversation_history.get(user_id, [])[-HISTORY_LIMIT:]
    if recent: 
        prompt += "Recent:\n" + "\n".join(recent) + "\n"
    
    # Add previous responses to avoid
    user_previous_responses = recent_responses.get(user_id, deque(maxlen=MAX_RECENT_RESPONSES))
    if user_previous_responses:
        prompt += "DO NOT use these exact responses again (your previous answers):\n"
        prompt += "\n".join([f"- {resp}" for resp in user_previous_responses]) + "\n"
    
    prompt += f"User: {user_input}\nA2:"
    
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.85,  # Increased from 0.7 to encourage more variation
            max_tokens=100
        )
        response = res.choices[0].message.content.strip()
        
        # Check if this response is too similar to previous ones
        if user_id not in recent_responses:
            recent_responses[user_id] = deque(maxlen=MAX_RECENT_RESPONSES)
        
        # Add to tracked responses and save it
        recent_responses[user_id].append(response)
        
        return response
    except:
        return "...I'm not in the mood."

async def generate_a2_response(user_input:str,trust:float,user_id:int)->str:
    return await asyncio.to_thread(generate_a2_response_sync,user_input,trust,user_id)

# ─── Tasks ───────────────────────────────────────────────────────────────────
@tasks.loop(minutes=10)
async def check_inactive_users():
    now = datetime.now(timezone.utc)
    for guild in bot.guilds:
        for member in guild.members:
            if member.bot or member.id not in user_emotions: 
                continue
                
            # Skip if user hasn't enabled DMs
            if member.id not in DM_ENABLED_USERS:
                continue
                
            last = datetime.fromisoformat(user_emotions[member.id]["last_interaction"])
            if now - last > timedelta(hours=6):
                try:
                    dm = await member.create_dm()
                    msg = random.choice(warm_lines if user_emotions[member.id]["trust"] >= 7 else provoking_lines)
                    await dm.send(msg)
                except discord.errors.Forbidden:
                    # Remove user from DM_ENABLED_USERS if they've blocked the bot
                    DM_ENABLED_USERS.discard(member.id)
                    await save_dm_settings()
                except Exception as e:
                    print(f"Error sending DM to {member.name}: {e}")
    asyncio.create_task(save_data())

@tasks.loop(hours=1)
async def decay_affection():
    for e in user_emotions.values(): e["affection_points"]=max(-100,e.get("affection_points",0)-AFFECTION_DECAY_RATE)
    asyncio.create_task(save_data())

@tasks.loop(hours=1)
async def decay_annoyance():
    for e in user_emotions.values(): e["annoyance"]=max(0,e.get("annoyance",0)-ANNOYANCE_DECAY_RATE)
    asyncio.create_task(save_data())

@tasks.loop(hours=24)
async def daily_affection_bonus():
    for e in user_emotions.values():
        if e.get("trust",0)>=DAILY_BONUS_TRUST_THRESHOLD: e["affection_points"]=min(1000,e.get("affection_points",0)+DAILY_AFFECTION_BONUS)
    asyncio.create_task(save_data())

# ─── Event Handlers ─────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    print("A2 is online.")
    # Initialize existing tasks
    check_inactive_users.start()
    decay_affection.start()
    decay_annoyance.start()
    daily_affection_bonus.start()
    
    # Load DM settings and Sam profile
    await load_dm_settings()
    await load_sam_profile()
    
    # Initialize recent_responses for each user
    global recent_responses
    recent_responses = {}
    for user_id in user_emotions:
        recent_responses[user_id] = deque(maxlen=MAX_RECENT_RESPONSES)

@bot.event
async def on_command_error(ctx,error):
    if isinstance(error,commands.CommandNotFound):return
    raise error

@bot.event
async def on_message(message):
    if message.author.bot or message.content.startswith("A2:"): 
        return
    
    uid, content = message.author.id, message.content.strip()
    
    # Check for Faded Sam mentions
    lower_content = content.lower()
    if FEARED_NAME in lower_content:
        global sam_mentions_count
        sam_mentions_count += 1
        
        # First, try to extract information
        info_extracted = extract_sam_info(content)
        
        # If it's just a mention without useful info, or if the message is short, show fear
        if not info_extracted or len(content) < 30:
            await message.channel.send(f"A2: {random.choice(FEAR_RESPONSES)}")
            await save_sam_profile()
            return  # Skip normal processing
    
    # Normal message processing continues
    is_cmd = any(content.startswith(p) for p in PREFIXES)
    is_mention = bot.user in message.mentions
    if not should_respond_to(content, uid, is_cmd, is_mention): 
        return
    
    await bot.process_commands(message)
    if is_cmd: 
        return
    
    trust = user_emotions.get(uid, {}).get("trust", 0)
    resp = await generate_a2_response(content, trust, uid)
    await message.channel.send(f"A2: {resp}")

# ─── Commands ───────────────────────────────────────────────────────────────
@bot.command(name="affection",help="Show emotion stats for all users.")
async def affection_all(ctx):
    if not user_emotions: return await ctx.send("A2: no interactions.")
    lines=[]
    for uid,e in user_emotions.items():
        member=bot.get_user(uid)or(ctx.guild and ctx.guild.get_member(uid))
        mention=member.mention if member else f"<@{uid}>"
        lines.append(f"**{mention}** • Trust: {e.get('trust',0)}/10 • Attachment: {e.get('attachment',0)}/10 • Protectiveness: {e.get('protectiveness',0)}/10 • Resentment: {e.get('resentment',0)}/10 • Affection: {e.get('affection_points',0)} • Annoyance: {e.get('annoyance',0)}")
    await ctx.send("\n".join(lines))

@bot.command(name="stats",help="Show your stats.")
async def stats(ctx):
    uid=ctx.author.id; e=user_emotions.get(uid)
    if not e: return await ctx.send("A2: no data on you.")
    embed=discord.Embed(title="Your Emotion Stats",color=discord.Color.blue(),timestamp=datetime.now(timezone.utc))
    embed.add_field(name="Trust",value=f"{e.get('trust',0)}/10",inline=True)
    embed.add_field(name="Attachment",value=f"{e.get('attachment',0)}/10",inline=True)
    embed.add_field(name="Protectiveness",value=f"{e.get('protectiveness',0)}/10",inline=True)
    embed.add_field(name="Resentment",value=f"{e.get('resentment',0)}/10",inline=True)
    embed.add_field(name="Affection",value=str(e.get('affection_points',0)),inline=True)
    embed.add_field(name="Annoyance",value=str(e.get('annoyance',0)),inline=True)
    embed.set_footer(text="A2 Bot")
    await ctx.send(embed=embed)

@bot.command(name="set_stat",aliases=["setstat"],help="Dev: set a stat for a user or yourself.")
async def set_stat(ctx,stat:str,value:float,member:discord.Member=None):
    target=member or ctx.author;uid=target.id
    e=user_emotions.setdefault(uid,{"trust":0,"resentment":0,"attachment":0,"protectiveness":0,"affection_points":0,"annoyance":0,"guilt_triggered":False,"last_interaction":datetime.now(timezone.utc).isoformat()})
    limits={'trust':(0,10),'resentment':(0,10),'attachment':(0,10),'protectiveness':(0,10),'annoyance':(0,100),'affection_points':(-100,1000)}
    key=stat.lower()
    if key=='affection':key='affection_points'
    if key not in limits:return await ctx.send(f"A2: Unknown stat '{stat}'. Valid stats: {', '.join(limits.keys())}.")
    lo,hi=limits[key];e[key]=max(lo,min(hi,value));asyncio.create_task(save_data());await ctx.send(f"A2: Set {key} to {e[key]} for {target.mention}.")

@bot.command(name="ping",help="Ping the bot.")
async def ping(ctx):await ctx.send("Pong!")

@bot.command(name="test_decay",help="Run affection and annoyance decay immediately.")
async def test_decay(ctx):decay_affection.restart();decay_annoyance.restart();await ctx.send("A2: Decay tasks triggered.")

@bot.command(name="test_daily",help="Run daily affection bonus immediately.")
async def test_daily(ctx):daily_affection_bonus.restart();await ctx.send("A2: Daily affection bonus triggered.")

@bot.command(name="view_emotions",help="View raw emotion data for a user.")
async def view_emotions(ctx,member:discord.Member=None):
    target=member or ctx.author;uid=target.id
    if uid not in user_emotions:return await ctx.send(f"A2: No data for {target.mention}.")
    await ctx.send(f"Emotion data for {target.mention}: {json.dumps(user_emotions[uid],indent=2)}")

@bot.command(name="clear_responses", help="Clear the bot's memory of previous responses")
async def clear_responses(ctx, member: discord.Member = None):
    target = member or ctx.author
    uid = target.id
    if uid in recent_responses:
        recent_responses[uid].clear()
        await ctx.send(f"A2: Memory banks cleared for {target.mention}.")
    else:
        await ctx.send(f"A2: No stored responses for {target.mention}.")

@bot.command(name="enable_dm", aliases=["dms_on"], help="Enable A2 to send you direct messages")
async def enable_dm(ctx):
    DM_ENABLED_USERS.add(ctx.author.id)
    await save_dm_settings()
    await ctx.send("A2: DMs enabled. I can contact you directly now.")
    
@bot.command(name="disable_dm", aliases=["dms_off"], help="Disable A2 from sending you direct messages")
async def disable_dm(ctx):
    DM_ENABLED_USERS.discard(ctx.author.id)
    await save_dm_settings()
    await ctx.send("A2: DMs disabled. I won't bother you anymore.")
    
@bot.command(name="dm_status", help="Check if you have DMs enabled or disabled")
async def dm_status(ctx):
    status = "enabled" if ctx.author.id in DM_ENABLED_USERS else "disabled"
    await ctx.send(f"A2: Your DMs are currently {status}.")

@bot.command(name="sam_profile", help="View the compiled profile of Faded Sam")
async def view_sam_profile(ctx):
    if not sam_profile:
        return await ctx.send("A2: *tenses up* ...No information to share about that entity.")
    
    embed = discord.Embed(
        title="Faded Sam Profile", 
        description=f"*A2 seems uncomfortable sharing this information*\nMentions: {sam_mentions_count}",
        color=discord.Color.dark_red()
    )
    
    for category, entries in sam_profile.items():
        if entries:
            # Format the entries nicely, limit to 5 per category
            formatted_entries = "\n• ".join(entries[:5])
            if len(entries) > 5:
                formatted_entries += f"\n• ... ({len(entries) - 5} more)"
            
            embed.add_field(
                name=category.capitalize(),
                value=f"• {formatted_entries}",
                inline=False
            )
    
    await ctx.send(embed=embed)

@bot.command(name="reset_sam", help="Reset the Faded Sam profile")
async def reset_sam_profile(ctx):
    # You might want to add permission checks here
    global sam_profile, sam_mentions_count
    sam_profile = defaultdict(list)
    sam_mentions_count = 0
    await save_sam_profile()
    await ctx.send("A2: *looks relieved* ...Deleted.")

if __name__=="__main__":
    bot.run(DISCORD_BOT_TOKEN)
