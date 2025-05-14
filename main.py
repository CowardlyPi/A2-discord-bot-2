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
from collections import defaultdict, deque
import time

# --- Memory System Improvements ---
class MemoryManager:
    def __init__(self, data_dir, max_history=20, summary_threshold=10):
        self.data_dir = Path(data_dir)
        self.memories_dir = self.data_dir / "memories"
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory settings
        self.max_history = max_history          # Maximum raw messages to store
        self.summary_threshold = summary_threshold  # When to summarize conversation
        
        # Memory cache
        self.raw_history = defaultdict(lambda: deque(maxlen=self.max_history))
        self.summaries = {}                     # Long-term summaries
        self.key_memories = defaultdict(list)   # Important memories per user
        self.interests = defaultdict(set)       # Topics user has shown interest in
        self.last_topics = defaultdict(str)     # Last conversation topic
        self.memory_index = defaultdict(dict)   # Topic/keyword lookup for memories
        self.last_summarized = {}               # Track when last summarized
        
        # Persistence
        self.dirty_users = set()                # Users with unsaved changes
    
    async def load_all_memories(self):
        """Load all user memories from disk"""
        for file in self.memories_dir.glob("*.json"):
            user_id = int(file.stem)
            await self.load_user_memories(user_id)
    
    async def load_user_memories(self, user_id):
        """Load a specific user's memories"""
        path = self.memories_dir / f"{user_id}.json"
        if not path.exists():
            return
        
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            
            # Load history
            if "history" in data:
                self.raw_history[user_id] = deque(data["history"], maxlen=self.max_history)
            
            # Load summaries
            if "summaries" in data:
                self.summaries[user_id] = data["summaries"]
            
            # Load key memories
            if "key_memories" in data:
                self.key_memories[user_id] = data["key_memories"]
            
            # Load interests
            if "interests" in data:
                self.interests[user_id] = set(data["interests"])
            
            # Load last topic
            if "last_topic" in data:
                self.last_topics[user_id] = data["last_topic"]
            
            # Load memory index
            if "memory_index" in data:
                self.memory_index[user_id] = data["memory_index"]
                
            # Load last summarized timestamp
            if "last_summarized" in data:
                self.last_summarized[user_id] = data["last_summarized"]
                
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error loading memories for {user_id}: {e}")
    
    async def save_user_memories(self, user_id):
        """Save a specific user's memories to disk"""
        path = self.memories_dir / f"{user_id}.json"
        
        # Prepare data structure for saving
        data = {
            "history": list(self.raw_history[user_id]),
            "summaries": self.summaries.get(user_id, ""),
            "key_memories": self.key_memories.get(user_id, []),
            "interests": list(self.interests[user_id]),
            "last_topic": self.last_topics.get(user_id, ""),
            "memory_index": self.memory_index.get(user_id, {}),
            "last_summarized": self.last_summarized.get(user_id, 0)
        }
        
        # Write to disk
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        
        # Remove from dirty set
        if user_id in self.dirty_users:
            self.dirty_users.remove(user_id)
    
    async def save_all_dirty(self):
        """Save all users with unsaved changes"""
        for user_id in list(self.dirty_users):
            await self.save_user_memories(user_id)
    
    async def add_message(self, user_id, author, content):
        """Add a message to a user's history"""
        timestamp = datetime.now(timezone.utc).isoformat()
        message = {
            "author": author,
            "content": content,
            "timestamp": timestamp
        }
        
        # Add to raw history
        self.raw_history[user_id].append(message)
        
        # Mark for saving
        self.dirty_users.add(user_id)
        
        # Check if we need to summarize
        if len(self.raw_history[user_id]) >= self.summary_threshold:
            last_sum = self.last_summarized.get(user_id, 0)
            current_time = time.time()
            
            # Only summarize if it's been at least 30 minutes
            if current_time - last_sum > 1800:  # 30 minutes in seconds
                await self.generate_summary(user_id)
                self.last_summarized[user_id] = current_time
    
    async def extract_interests(self, user_id, content):
        """Extract topics of interest from user message"""
        # This could be enhanced with more sophisticated NLP
        # Simple keyword extraction for now
        common_topics = [
            "combat", "android", "emotions", "yorha", "machine", "humanity",
            "existence", "war", "memory", "programming", "games"
        ]
        
        content_lower = content.lower()
        found = set()
        
        for topic in common_topics:
            if topic in content_lower:
                found.add(topic)
        
        if found:
            self.interests[user_id].update(found)
            self.dirty_users.add(user_id)
    
    async def add_key_memory(self, user_id, memory, topics=None):
        """Add an important memory for a user"""
        if not topics:
            topics = []
            
        timestamp = datetime.now(timezone.utc).isoformat()
        key_memory = {
            "content": memory,
            "timestamp": timestamp,
            "topics": topics
        }
        
        # Add to key memories list
        self.key_memories[user_id].append(key_memory)
        
        # Update index for each topic
        for topic in topics:
            if topic not in self.memory_index[user_id]:
                self.memory_index[user_id][topic] = []
            self.memory_index[user_id][topic].append(len(self.key_memories[user_id]) - 1)
        
        # Mark for saving
        self.dirty_users.add(user_id)
    
    async def generate_summary(self, user_id):
        """Generate or update conversation summary"""
        # If we don't have at least the threshold number of messages, skip
        if len(self.raw_history[user_id]) < self.summary_threshold:
            return
        
        # Use local summarizer if available, otherwise use OpenAI
        messages = self.format_history_for_summarization(user_id)
        
        if HAVE_TRANSFORMERS and local_summarizer:
            try:
                text = " ".join(msg.get("content", "") for msg in messages)
                summary = local_summarizer(text, max_length=150, min_length=40)[0]["summary_text"]
                
                # Store the summary
                if user_id not in self.summaries:
                    self.summaries[user_id] = summary
                else:
                    # Combine with existing summary
                    self.summaries[user_id] = f"{self.summaries[user_id]}\n\nRecent: {summary}"
                
                # Mark for saving
                self.dirty_users.add(user_id)
                return
            except Exception as e:
                print(f"Local summarization failed: {e}")
        
        # Fall back to OpenAI
        try:
            # Format for API
            prompt = "Summarize this conversation into bullet points focusing on key facts about the user and important events, under 200 tokens:\n"
            prompt += "\n".join([f"{msg['author']}: {msg['content']}" for msg in messages])
            
            res = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=200
            )
            
            summary = res.choices[0].message.content.strip()
            
            # Update the user's summary
            if user_id not in self.summaries:
                self.summaries[user_id] = summary
            else:
                # Check and remove duplicated information
                existing_points = set(self.summaries[user_id].split("\n"))
                new_points = summary.split("\n")
                unique_new_points = [p for p in new_points if p not in existing_points]
                
                if unique_new_points:
                    combined = f"{self.summaries[user_id]}\n" + "\n".join(unique_new_points)
                    self.summaries[user_id] = combined
            
            # Mark for saving
            self.dirty_users.add(user_id)
            
        except Exception as e:
            print(f"OpenAI summarization failed: {e}")
    
    def format_history_for_summarization(self, user_id):
        """Format the raw history for summarization"""
        return list(self.raw_history[user_id])
    
    async def detect_significant_event(self, user_id, message, bot_response):
        """Detect if a conversation exchange represents a significant memory"""
        # This could use sentiment analysis or other heuristics
        # For now, we'll use simple keyword detection
        
        significant_keywords = [
            "promise", "never forget", "remember", "important", "trust",
            "secret", "help me", "always", "forever", "thank you"
        ]
        
        message_lower = message.lower()
        response_lower = bot_response.lower()
        
        # Check for significant keywords
        if any(keyword in message_lower or keyword in response_lower for keyword in significant_keywords):
            # Create a memory entry
            memory = f"User said: '{message}' and A2 responded: '{bot_response}'"
            
            # Extract potential topics
            potential_topics = []
            for topic in self.interests[user_id]:
                if topic.lower() in message_lower or topic.lower() in response_lower:
                    potential_topics.append(topic)
            
            # Add as key memory
            await self.add_key_memory(user_id, memory, topics=potential_topics)
    
    async def get_relevant_memories(self, user_id, current_message):
        """Retrieve memories relevant to the current conversation"""
        if user_id not in self.key_memories or not self.key_memories[user_id]:
            return None
        
        # Simple relevance check based on keywords
        message_lower = current_message.lower()
        relevant_memories = []
        
        # Check if any topic from the memory index appears in the message
        for topic, indices in self.memory_index[user_id].items():
            if topic.lower() in message_lower:
                for idx in indices:
                    if idx < len(self.key_memories[user_id]):
                        relevant_memories.append(self.key_memories[user_id][idx])
        
        # If we have too many, select the most recent ones
        if len(relevant_memories) > 3:
            # Sort by timestamp (descending) and take top 3
            relevant_memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            relevant_memories = relevant_memories[:3]
        
        return relevant_memories if relevant_memories else None
    
    def get_memory_context(self, user_id):
        """Get formatted memory context for generating responses"""
        context = []
        
        # Add summary if available
        if user_id in self.summaries and self.summaries[user_id]:
            context.append(f"Memory summary:\n{self.summaries[user_id]}")
        
        # Add interests if available
        if user_id in self.interests and self.interests[user_id]:
            context.append(f"User interests: {', '.join(self.interests[user_id])}")
        
        # Add last topic if available
        if user_id in self.last_topics and self.last_topics[user_id]:
            context.append(f"Last conversation topic: {self.last_topics[user_id]}")
        
        return "\n\n".join(context) if context else ""
    
    async def update_conversation_topic(self, user_id, message, ai_response=None):
        """Update the tracked conversation topic"""
        # This could be more sophisticated with topic modeling
        # Simple implementation for now
        try:
            prompt = f"Identify the main topic of this conversation in 3 words or less:\nHuman: {message}"
            if ai_response:
                prompt += f"\nAI: {ai_response}"
            
            res = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=10
            )
            
            topic = res.choices[0].message.content.strip()
            self.last_topics[user_id] = topic
            self.dirty_users.add(user_id)
            
        except Exception as e:
            print(f"Topic extraction failed: {e}")

# --- Rest of the code remains largely the same ---

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

# Initialize Memory Manager
memory_manager = MemoryManager(DATA_DIR)

# Global variables (legacy)
conversation_summaries = {}
conversation_history = {}
user_emotions = {}
last_bot_responses = {}

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

async def load_data():
    global user_emotions
    user_emotions = {}
    for file in PROFILES_DIR.glob("*.json"):
        uid = int(file.stem)
        user_emotions[uid] = await load_user_profile(uid)
    
    # Load memories as well
    await memory_manager.load_all_memories()

async def save_data():
    for uid in list(user_emotions.keys()):
        await save_user_profile(uid)
    
    # Save memories as well
    await memory_manager.save_all_dirty()

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
provoking_lines = ["Still mad? Good.", "You again? Tch.", "What?"]
warm_lines      = ["...Checking in.", "Still breathing?", "Thought you got scrapped."]

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

# ─── A2 Response ─────────────────────────────────────────────────────────────
async def generate_a2_response(user_input:str, trust:float, user_id:int) -> str:
    # Get memory context
    memory_context = memory_manager.get_memory_context(user_id)
    
    # Get relevant memories for this conversation
    relevant_memories = await memory_manager.get_relevant_memories(user_id, user_input)
    specific_memories = ""
    if relevant_memories:
        memories_text = "\n".join([f"- {m['content']}" for m in relevant_memories])
        specific_memories = f"Relevant memories:\n{memories_text}\n"
    
    # Extract possible topics of interest
    await memory_manager.extract_interests(user_id, user_input)
    
    # Use the better model for high trust users
    model = "gpt-3.5-turbo" if trust < 5 else "gpt-4"
    
    # Construct the prompt with memory integration
    prompt = A2_PERSONA + f"\nTrust: {trust}/10\n"
    
    # Add memory context if available
    if memory_context:
        prompt += f"{memory_context}\n"
    
    # Add specific memories if available
    if specific_memories:
        prompt += f"{specific_memories}\n"
    
    # Add recent conversation history
    recent_history = []
    if user_id in memory_manager.raw_history:
        recent = list(memory_manager.raw_history[user_id])[-HISTORY_LIMIT:]
        for msg in recent:
            if msg["author"] == "User":
                recent_history.append(f"User: {msg['content']}")
            else:
                recent_history.append(f"A2: {msg['content']}")
    
    if recent_history:
        prompt += "Recent conversation:\n" + "\n".join(recent_history) + "\n"
    
    # Add the current user input
    prompt += f"User: {user_input}\nA2:"
    
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_tokens=100
        )
        response = res.choices[0].message.content.strip()
        
        # Store the message exchange
        await memory_manager.add_message(user_id, "User", user_input)
        await memory_manager.add_message(user_id, "A2", response)
        
        # Update the conversation topic
        await memory_manager.update_conversation_topic(user_id, user_input, response)
        
        # Check if this is a significant memory
        await memory_manager.detect_significant_event(user_id, user_input, response)
        
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "...I'm not in the mood."

# ─── Tasks ───────────────────────────────────────────────────────────────────
@tasks.loop(minutes=10)
async def check_inactive_users():
    now=datetime.now(timezone.utc)
    for guild in bot.guilds:
        for member in guild.members:
            if member.bot or member.id not in user_emotions: continue
            last=datetime.fromisoformat(user_emotions[member.id]["last_interaction"])
            if now-last>timedelta(hours=6):
                dm=await member.create_dm()
                msg=random.choice(warm_lines if user_emotions[member.id]["trust"]>=7 else provoking_lines)
                await dm.send(msg)
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

@tasks.loop(hours=1)
async def save_memory_task():
    """Periodically save memory data"""
    await memory_manager.save_all_dirty()

# ─── Event Handlers ─────────────────────────────────────────────────────────
@bot.event
async def on_ready():
    print("A2 is online.")
    check_inactive_users.start()
    decay_affection.start()
    decay_annoyance.start()
    daily_affection_bonus.start()
    save_memory_task.start()  # Start the memory saving task

@bot.event
async def on_command_error(ctx,error):
    if isinstance(error,commands.CommandNotFound):return
    raise error

@bot.event
async def on_message(message):
    if message.author.bot or message.content.startswith("A2:"):
        return

    uid, content = message.author.id, message.content.strip()
    is_cmd = any(content.startswith(p) for p in PREFIXES)
    is_mention = bot.user in message.mentions
    if not should_respond_to(content, uid, is_cmd, is_mention):
        return

    await bot.process_commands(message)
    if is_cmd:
        return

    trust = user_emotions.get(uid, {}).get("trust", 0)
    apply_reaction_modifiers(content, uid)  # Update emotions
    
    resp = await generate_a2_response(content, trust, uid)
    # Prevent sending the exact same response twice in a row
    if last_bot_responses.get(uid) == resp:
        return

    last_bot_responses[uid] = resp
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
async def set_stat(ctx,stat:str,value:float,member
