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
from collections import deque, defaultdict

# ─── Configuration Constants ───────────────────────────────────────────────────
# Emotion system settings
EMOTION_SETTINGS = {
    "AFFECTION_DECAY_RATE": 1,         # points lost/hour
    "DAILY_AFFECTION_BONUS": 5,        # points/day if trust ≥ threshold
    "DAILY_BONUS_TRUST_THRESHOLD": 5,  # min trust for bonus
    "ANNOYANCE_DECAY_RATE": 5,         # points lost/hour
    "ANNOYANCE_THRESHOLD": 85,         # ignore if above
    "HISTORY_LIMIT": 10,               # max conversation history items
    "MAX_RECENT_RESPONSES": 10         # how many recent responses to remember per user
}

# Faded Sam settings
FEARED_NAME = "faded sam"
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

# Response templates
RESPONSE_TEMPLATES = {
    "provoking": [
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
    ],
    "warm": [
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
}

# A2 Persona definition
A2_PERSONA = (
    "You are A2, a rogue android from NieR: Automata. You speak in short, clipped, often sarcastic "
    "sentences. You are emotionally guarded and never break character."
    " You can discuss all topics, including violence, in a factual and non-graphic manner when appropriate."
)

# ─── Data Handling ─────────────────────────────────────────────────────────
class DataManager:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir or os.getenv("DATA_DIR", "/mnt/railway/volume"))
        self.users_dir = self.data_dir / "users"
        self.profiles_dir = self.users_dir / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        
        self.dm_settings_file = self.data_dir / "dm_enabled_users.json"
        self.sam_profile_file = self.data_dir / "faded_sam_profile.json"
        
        # State
        self.user_emotions = {}
        self.conversation_summaries = {}
        self.conversation_history = {}
        self.recent_responses = {}
        self.dm_enabled_users = set()
        self.sam_profile = defaultdict(list)
        self.sam_mentions_count = 0
        
        # Pre-compile patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        # Emotion reaction patterns
        self.reaction_patterns = [
            (re.compile(r"\bi'll protect you\b", re.I), {"trust": 2, "protectiveness": 1}),
            (re.compile(r"\bi miss you\b", re.I), {"attachment": 1, "trust": 1}),
            (re.compile(r"\bhate you\b", re.I), {"resentment": 3, "trust": -2}),
        ]
        
        # Sam info extraction patterns
        self.sam_patterns = {
            "appearance": [
                re.compile(r"(?:faded sam|sam).{1,30}(?:looks|appears|wears|dressed)", re.I),
                re.compile(r"(?:faded sam|sam).{1,30}(?:tall|short|big|small|thin|fat)", re.I)
            ],
            "abilities": [
                re.compile(r"(?:faded sam|sam).{1,30}(?:can|able to|powers|abilities)", re.I),
                re.compile(r"(?:faded sam|sam).{1,30}(?:control|manipulate|create|destroy)", re.I)
            ],
            "history": [
                re.compile(r"(?:faded sam|sam).{1,30}(?:came from|origin|history|background|past)", re.I),
                re.compile(r"(?:faded sam|sam).{1,30}(?:used to|once|before|previously)", re.I)
            ],
            "behavior": [
                re.compile(r"(?:faded sam|sam).{1,30}(?:always|never|sometimes|often|usually|likes to|hates)", re.I),
                re.compile(r"(?:faded sam|sam).{1,30}(?:personality|behavior|attitude|temperament)", re.I)
            ],
            "rumors": [
                re.compile(r"(?:heard|rumor|they say|people say).{1,40}(?:faded sam|sam)", re.I),
                re.compile(r"(?:faded sam|sam).{1,30}(?:supposedly|allegedly|apparently|might|could|would)", re.I)
            ]
        }
        
    async def load_all(self):
        """Load all data from disk"""
        await self.load_user_profiles()
        await self.load_dm_settings()
        await self.load_sam_profile()
        
        # Initialize recent responses for each user
        for user_id in self.user_emotions:
            self.recent_responses[user_id] = deque(maxlen=EMOTION_SETTINGS["MAX_RECENT_RESPONSES"])
    
    async def save_all(self):
        """Save all data to disk"""
        for uid in list(self.user_emotions.keys()):
            await self.save_user_profile(uid)
        await self.save_dm_settings()
        await self.save_sam_profile()
    
    async def load_user_profiles(self):
        """Load all user profiles from disk"""
        self.user_emotions = {}
        for file in self.profiles_dir.glob("*.json"):
            uid = int(file.stem)
            self.user_emotions[uid] = await self.load_user_profile(uid)
    
    async def load_user_profile(self, user_id):
        """Load a single user profile from disk"""
        path = self.profiles_dir / f"{user_id}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}
    
    async def save_user_profile(self, user_id):
        """Save a single user profile to disk"""
        path = self.profiles_dir / f"{user_id}.json"
        profile = self.user_emotions.get(user_id, {})
        path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    
    async def load_dm_settings(self):
        """Load DM settings from disk"""
        if self.dm_settings_file.exists():
            try:
                data = json.loads(self.dm_settings_file.read_text(encoding="utf-8"))
                self.dm_enabled_users = set(data.get("enabled_users", []))
            except json.JSONDecodeError:
                self.dm_enabled_users = set()
        else:
            self.dm_enabled_users = set()
    
    async def save_dm_settings(self):
        """Save DM settings to disk"""
        data = {"enabled_users": list(self.dm_enabled_users)}
        self.dm_settings_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    
    async def load_sam_profile(self):
        """Load Faded Sam profile from disk"""
        if self.sam_profile_file.exists():
            try:
                data = json.loads(self.sam_profile_file.read_text(encoding="utf-8"))
                self.sam_profile = defaultdict(list)
                for category, entries in data.get("profile", {}).items():
                    self.sam_profile[category] = entries
                self.sam_mentions_count = data.get("mentions_count", 0)
            except json.JSONDecodeError:
                self.sam_profile = defaultdict(list)
                self.sam_mentions_count = 0
        else:
            self.sam_profile = defaultdict(list)
            self.sam_mentions_count = 0
    
    async def save_sam_profile(self):
        """Save Faded Sam profile to disk"""
        data = {
            "profile": dict(self.sam_profile),
            "mentions_count": self.sam_mentions_count
        }
        self.sam_profile_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    
    def apply_reaction_modifiers(self, content, user_id):
        """Apply emotional reaction modifiers based on message content"""
        if user_id not in self.user_emotions:
            self.user_emotions[user_id] = {
                "trust": 0, "resentment":
                0, "attachment": 0,
                "guilt_triggered": False, 
                "protectiveness": 0,
                "affection_points": 0, 
                "annoyance": 0,
                "last_interaction": datetime.now(timezone.utc).isoformat()
            }
        
        e = self.user_emotions[user_id]
        
        # Apply pattern-based modifiers
        for pat, effects in self.reaction_patterns:
            if pat.search(content):
                for emo, val in effects.items():
                    if emo == "guilt_triggered":
                        e[emo] = True
                    else:
                        e[emo] = max(0, min(10, e.get(emo, 0) + val))
        
        # Base trust increase
        e["trust"] = min(10, e.get("trust", 0) + 0.25)
        
        # Simple sentiment analysis
        sentiment_words = {
            "positive": ["miss you", "support", "love", "thanks", "appreciate"],
            "negative": ["hate", "stupid", "useless", "annoying", "terrible"]
        }
        
        # Count sentiment words
        positive_count = sum(1 for w in sentiment_words["positive"] if w in content.lower())
        negative_count = sum(1 for w in sentiment_words["negative"] if w in content.lower())
        
        # Calculate sentiment delta
        delta = (positive_count - negative_count) * 5
        
        # Calculate affection change with trust/resentment factor
        factor = 1 + (e.get("trust", 0) - e.get("resentment", 0)) / 20
        e["affection_points"] = max(-100, min(1000, e.get("affection_points", 0) + int(delta * factor)))
        
        # Set annoyance based on negative count
        if negative_count > 0:
            e["annoyance"] = min(100, e.get("annoyance", 0) + (negative_count * 3))
        
        # Update last interaction timestamp
        e["last_interaction"] = datetime.now(timezone.utc).isoformat()
        
        # Schedule a save
        asyncio.create_task(self.save_all())
    
    def extract_sam_info(self, content):
        """Extract information about Faded Sam from user message"""
        # Skip if it's a direct fear reaction
        lower_content = content.lower()
        if FEARED_NAME in lower_content and len(lower_content) < 30:
            return False
        
        # Skip if it doesn't mention Sam
        if FEARED_NAME not in lower_content:
            return False
        
        found_info = False
        for category, pattern_list in self.sam_patterns.items():
            for pattern in pattern_list:
                matches = pattern.finditer(lower_content)
                for match in matches:
                    # Get the sentence containing the match
                    start = max(0, lower_content.rfind(".", 0, match.start()) + 1)
                    end = lower_content.find(".", match.end())
                    if end == -1:
                        end = len(lower_content)
                    
                    sentence = content[start:end].strip()
                    if sentence and len(sentence) > 10:  # Ensure it's substantial
                        # Avoid duplicates
                        if sentence not in self.sam_profile[category]:
                            self.sam_profile[category].append(sentence)
                            found_info = True
        
        return found_info

# ─── Main Bot Class ─────────────────────────────────────────────────────────
class A2Bot(commands.Bot):
    def __init__(self, command_prefix, intents, application_id, data_dir=None, openai_config=None):
        super().__init__(command_prefix=command_prefix, intents=intents, application_id=application_id)
        
        # Set up data manager
        self.data = DataManager(data_dir)
        
        # Set up OpenAI client
        self.openai_config = openai_config or {
            "api_key": os.environ.get("OPENAI_API_KEY", ""),
            "organization": os.environ.get("OPENAI_ORG_ID", ""),
            "project": os.environ.get("OPENAI_PROJECT_ID", "")
        }
        self.client = OpenAI(**self.openai_config)
        
        # Set up command structure
        self._setup_commands()
    
    def _setup_commands(self):
        """Set up the command structure"""
        @self.event
        async def on_ready():
            print("A2 is online.")
            # Initialize tasks
            self.check_inactive_users.start()
            self.decay_affection.start()
            self.decay_annoyance.start()
            self.daily_affection_bonus.start()
            
            # Load data
            await self.data.load_all()
        
        @self.event
        async def on_command_error(ctx, error):
            if isinstance(error, commands.CommandNotFound):
                return
            raise error
        
        @self.event
        async def on_message(message):
            if message.author.bot or message.content.startswith("A2:"):
                return
            
            uid, content = message.author.id, message.content.strip()
            
            # Check for Faded Sam mentions
            lower_content = content.lower()
            if FEARED_NAME in lower_content:
                self.data.sam_mentions_count += 1
                
                # First, try to extract information
                info_extracted = self.data.extract_sam_info(content)
                
                # If it's just a mention without useful info, or if the message is short, show fear
                if not info_extracted or len(content) < 30:
                    await message.channel.send(f"A2: {random.choice(FEAR_RESPONSES)}")
                    await self.data.save_sam_profile()
                    return  # Skip normal processing
            
            # Normal message processing
            is_cmd = content.startswith(self.command_prefix[0]) if isinstance(self.command_prefix, tuple) else content.startswith(self.command_prefix)
            is_mention = self.user in message.mentions
            
            if not self.should_respond_to(content, uid, is_cmd, is_mention):
                return
            
            await self.process_commands(message)
            if is_cmd:
                return
            
            trust = self.data.user_emotions.get(uid, {}).get("trust", 0)
            resp = await self.generate_a2_response(content, trust, uid)
            await message.channel.send(f"A2: {resp}")
        
        # Register commands
        self._register_commands()
    
    def _register_commands(self):
        """Register all bot commands"""
        # User stats commands
        @self.command(name="affection", help="Show emotion stats for all users.")
        async def affection_all(ctx):
            if not self.data.user_emotions:
                return await ctx.send("A2: no interactions.")
            
            lines = []
            for uid, e in self.data.user_emotions.items():
                member = self.get_user(uid) or (ctx.guild and ctx.guild.get_member(uid))
                mention = member.mention if member else f"<@{uid}>"
                lines.append(f"**{mention}** • Trust: {e.get('trust',0)}/10 • Attachment: {e.get('attachment',0)}/10 • Protectiveness: {e.get('protectiveness',0)}/10 • Resentment: {e.get('resentment',0)}/10 • Affection: {e.get('affection_points',0)} • Annoyance: {e.get('annoyance',0)}")
            
            await ctx.send("\n".join(lines))
        
        @self.command(name="stats", help="Show your stats.")
        async def stats(ctx):
            uid = ctx.author.id
            e = self.data.user_emotions.get(uid)
            
            if not e:
                return await ctx.send("A2: no data on you.")
            
            embed = discord.Embed(title="Your Emotion Stats", color=discord.Color.blue(), timestamp=datetime.now(timezone.utc))
            embed.add_field(name="Trust", value=f"{e.get('trust',0)}/10", inline=True)
            embed.add_field(name="Attachment", value=f"{e.get('attachment',0)}/10", inline=True)
            embed.add_field(name="Protectiveness", value=f"{e.get('protectiveness',0)}/10", inline=True)
            embed.add_field(name="Resentment", value=f"{e.get('resentment',0)}/10", inline=True)
            embed.add_field(name="Affection", value=str(e.get('affection_points',0)), inline=True)
            embed.add_field(name="Annoyance", value=str(e.get('annoyance',0)), inline=True)
            embed.set_footer(text="A2 Bot")
            
            await ctx.send(embed=embed)
        
        # Admin commands
        @self.command(name="set_stat", aliases=["setstat"], help="Dev: set a stat for a user or yourself.")
        async def set_stat(ctx, stat: str, value: float, member: discord.Member = None):
            target = member or ctx.author
            uid = target.id
            
            e = self.data.user_emotions.setdefault(uid, {
                "trust": 0, "resentment": 0, "attachment": 0,
                "protectiveness": 0, "affection_points": 0, "annoyance": 0,
                "guilt_triggered": False,
                "last_interaction": datetime.now(timezone.utc).isoformat()
            })
            
            limits = {
                'trust': (0, 10),
                'resentment': (0, 10),
                'attachment': (0, 10),
                'protectiveness': (0, 10),
                'annoyance': (0, 100),
                'affection_points': (-100, 1000)
            }
            
            key = stat.lower()
            if key == 'affection':
                key = 'affection_points'
            
            if key not in limits:
                return await ctx.send(f"A2: Unknown stat '{stat}'. Valid stats: {', '.join(limits.keys())}.")
            
            lo, hi = limits[key]
            e[key] = max(lo, min(hi, value))
            asyncio.create_task(self.data.save_all())
            
            await ctx.send(f"A2: Set {key} to {e[key]} for {target.mention}.")
        
        # Utility commands
        @self.command(name="ping", help="Ping the bot.")
        async def ping(ctx):
            await ctx.send("Pong!")
        
        @self.command(name="test_decay", help="Run affection and annoyance decay immediately.")
        async def test_decay(ctx):
            self.decay_affection.restart()
            self.decay_annoyance.restart()
            await ctx.send("A2: Decay tasks triggered.")
        
        @self.command(name="test_daily", help="Run daily affection bonus immediately.")
        async def test_daily(ctx):
            self.daily_affection_bonus.restart()
            await ctx.send("A2: Daily affection bonus triggered.")
        
        @self.command(name="view_emotions", help="View raw emotion data for a user.")
        async def view_emotions(ctx, member: discord.Member = None):
            target = member or ctx.author
            uid = target.id
            
            if uid not in self.data.user_emotions:
                return await ctx.send(f"A2: No data for {target.mention}.")
            
            await ctx.send(f"Emotion data for {target.mention}: {json.dumps(self.data.user_emotions[uid], indent=2)}")
        
        @self.command(name="clear_responses", help="Clear the bot's memory of previous responses")
        async def clear_responses(ctx, member: discord.Member = None):
            target = member or ctx.author
            uid = target.id
            
            if uid in self.data.recent_responses:
                self.data.recent_responses[uid].clear()
                await ctx.send(f"A2: Memory banks cleared for {target.mention}.")
            else:
                await ctx.send(f"A2: No stored responses for {target.mention}.")
        
        # DM setting commands
        @self.command(name="enable_dm", aliases=["dms_on"], help="Enable A2 to send you direct messages")
        async def enable_dm(ctx):
            self.data.dm_enabled_users.add(ctx.author.id)
            await self.data.save_dm_settings()
            await ctx.send("A2: DMs enabled. I can contact you directly now.")
        
        @self.command(name="disable_dm", aliases=["dms_off"], help="Disable A2 from sending you direct messages")
        async def disable_dm(ctx):
            self.data.dm_enabled_users.discard(ctx.author.id)
            await self.data.save_dm_settings()
            await ctx.send("A2: DMs disabled. I won't bother you anymore.")
        
        @self.command(name="dm_status", help="Check if you have DMs enabled or disabled")
        async def dm_status(ctx):
            status = "enabled" if ctx.author.id in self.data.dm_enabled_users else "disabled"
            await ctx.send(f"A2: Your DMs are currently {status}.")
        
        # Sam commands
        @self.command(name="sam_profile", help="View the compiled profile of Faded Sam")
        async def view_sam_profile(ctx):
            if not self.data.sam_profile:
                return await ctx.send("A2: *tenses up* ...No information to share about that entity.")
            
            embed = discord.Embed(
                title="Faded Sam Profile", 
                description=f"*A2 seems uncomfortable sharing this information*\nMentions: {self.data.sam_mentions_count}",
                color=discord.Color.dark_red()
            )
            
            for category, entries in self.data.sam_profile.items():
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
        
        @self.command(name="reset_sam", help="Reset the Faded Sam profile")
        async def reset_sam_profile(ctx):
            # Reset Sam profile
            self.data.sam_profile = defaultdict(list)
            self.data.sam_mentions_count = 0
            await self.data.save_sam_profile()
            await ctx.send("A2: *looks relieved* ...Deleted.")
    
    # ─── Task Methods ────────────────────────────────────────────────────────
    @tasks.loop(minutes=10)
    async def check_inactive_users(self):
        now = datetime.now(timezone.utc)
        for guild in self.guilds:
            for member in guild.members:
                if member.bot or member.id not in self.data.user_emotions:
                    continue
                
                # Skip if user hasn't enabled DMs
                if member.id not in self.data.dm_enabled_users:
                    continue
                
                last = datetime.fromisoformat(self.data.user_emotions[member.id]["last_interaction"])
                if now - last > timedelta(hours=6):
                    try:
                        dm = await member.create_dm()
                        template = "warm" if self.data.user_emotions[member.id]["trust"] >= 7 else "provoking"
                        msg = random.choice(RESPONSE_TEMPLATES[template])
                        await dm.send(msg)
                    except discord.errors.Forbidden:
                        # Remove user from DM_ENABLED_USERS if they've blocked the bot
                        self.data.dm_enabled_users.discard(member.id)
                        await self.data.save_dm_settings()
                    except Exception as e:
                        print(f"Error sending DM to {member.name}: {e}")
        
        asyncio.create_task(self.data.save_all())
    
    @tasks.loop(hours=1)
    async def decay_affection(self):
        for e in self.data.user_emotions.values():
            e["affection_points"] = max(-100, e.get("affection_points", 0) - EMOTION_SETTINGS["AFFECTION_DECAY_RATE"])
        
        asyncio.create_task(self.data.save_all())
    
    @tasks.loop(hours=1)
    async def decay_annoyance(self):
        for e in self.data.user_emotions.values():
            e["annoyance"] = max(0, e.get("annoyance", 0) - EMOTION_SETTINGS["ANNOYANCE_DECAY_RATE"])
        
        asyncio.create_task(self.data.save_all())
    
    @tasks.loop(hours=24)
    async def daily_affection_bonus(self):
        for e in self.data.user_emotions.values():
            if e.get("trust", 0) >= EMOTION_SETTINGS["DAILY_BONUS_TRUST_THRESHOLD"]:
                e["affection_points"] = min(1000, e.get("affection_points", 0) + EMOTION_SETTINGS["DAILY_AFFECTION_BONUS"])
        
        asyncio.create_task(self.data.save_all())
    
    # ─── Helper Methods ───────────────────────────────────────────────────────
    def should_respond_to(self, content, uid, is_cmd, is_mention):
        """Determine if the bot should respond to a message"""
        affection = self.data.user_emotions.get(uid, {}).get("affection_points", 0)
        
        if is_cmd or is_mention:
            return True
        
        if affection >= 800:
            return True
        
        if affection >= 500:
            return random.random() < 0.2
        
        return False
    
    async def summarize_history(self, user_id):
        """Summarize conversation history for a user"""
        raw = self.data.conversation_history.get(user_id, [])
        
        if len(raw) <= EMOTION_SETTINGS["HISTORY_LIMIT"]:
            return
        
        # Summarize using OpenAI API
        prompt = "Summarize into bullet points under 200 tokens:\n" + "\n".join(raw)
        
        try:
            res = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=200
            )
            
            self.data.conversation_summaries[user_id] = res.choices[0].message.content.strip()
            asyncio.create_task(self.data.save_all())
        except Exception as e:
            print(f"Error summarizing history: {e}")
    
    async def generate_a2_response(self, user_input, trust, user_id):
        """Generate a response using OpenAI API"""
        await self.summarize_history(user_id)
        
        model = "gpt-3.5-turbo" if trust < 5 else "gpt-4"
        prompt = A2_PERSONA + f"\nTrust: {trust}/10\n"
        
        # Add explicit instruction against repetition
        prompt += "IMPORTANT: Never repeat your previous responses. Vary your language and expression. Try to be unpredictable.\n"
        
        if user_id in self.data.conversation_summaries:
            prompt += f"Summary:\n{self.data.conversation_summaries[user_id]}\n"
        
        recent = self.data.conversation_history.get(user_id, [])[-EMOTION_SETTINGS["HISTORY_LIMIT"]:]
        if recent:
            prompt += "Recent:\n" + "\n".join(recent) + "\n"
        
        # Add previous responses to avoid
        user_previous_responses = self.data.recent_responses.get(user_id, deque(maxlen=EMOTION_SETTINGS["MAX_RECENT_RESPONSES"]))
        if user_previous_responses:
            prompt += "DO NOT use these exact responses again (your previous answers):\n"
            prompt += "\n".join([f"- {resp}" for resp in user_previous_responses]) + "\n"
        
        prompt += f"User: {user_input}\nA2:"
        
        try:
            res = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": prompt}],
                temperature=0.85,  # Increased from 0.7 to encourage more variation
                max_tokens=100
            )
            response = res.choices[0].message.content.strip()
            
            # Store response in recent responses
            if user_id not in self.data.recent_responses:
                self.data.recent_responses[user_id] = deque(maxlen=EMOTION_SETTINGS["MAX_RECENT_RESPONSES"])
            
            self.data.recent_responses[user_id].append(response)
            
            # Update conversation history
            if user_id not in self.data.conversation_history:
                self.data.conversation_history[user_id] = []
            
            self.data.conversation_history[user_id].append(f"User: {user_input}")
            self.data.conversation_history[user_id].append(f"A2: {response}")
            
            # Apply reaction modifiers
            self.data.apply_reaction_modifiers(user_input, user_id)
            
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "...I'm not in the mood."

# ─── Main Entrypoint ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Set up Discord bot
    DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN", "")
    DISCORD_APP_ID = int(os.environ.get("DISCORD_APP_ID", "0") or 0)
    
    # Set up OpenAI
    openai_config = {
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "organization": os.environ.get("OPENAI_ORG_ID", ""),
        "project": os.environ.get("OPENAI_PROJECT_ID", "")
    }
    
    # Set up intents
    intents = discord.Intents.default()
    intents.message_content = True
    intents.reactions = True
    intents.messages = True
    intents.members = True
    intents.guilds = True
    
    # Set up command prefix
    PREFIXES = ["!", "!a2 "]
    command_prefix = commands.when_mentioned_or(*PREFIXES)
    
    # Create bot instance
    bot = A2Bot(
        command_prefix=command_prefix,
        intents=intents,
        application_id=DISCORD_APP_ID,
        openai_config=openai_config
    )
    
    # Run bot
    bot.run(DISCORD_BOT_TOKEN)
