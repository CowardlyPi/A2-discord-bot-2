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
from collections import deque, defaultdict, Counter
import io

# ─── Configuration Settings ───────────────────────────────────────────────────
# Transformers availability check
HAVE_TRANSFORMERS = False
local_summarizer = None
local_toxic = None
local_sentiment = None
try:
    from transformers import pipeline
    HAVE_TRANSFORMERS = True
    local_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    local_toxic = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
    local_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except ImportError:
    pass

# ─── Emotional Settings ────────────────────────────────────────────────────
EMOTION_CONFIG = {
    # Decay settings
    "AFFECTION_DECAY_RATE": 1,         # points lost/hour
    "ANNOYANCE_DECAY_RATE": 5,         # points lost/hour
    "ANNOYANCE_THRESHOLD": 85,         # ignore if above
    "DAILY_AFFECTION_BONUS": 5,        # points/day if trust ≥ threshold
    "DAILY_BONUS_TRUST_THRESHOLD": 5,  # min trust for bonus
    
    # Emotion decay multipliers
    "DECAY_MULTIPLIERS": {
        'trust': 0.8,           # Trust decays slowly
        'resentment': 0.7,      # Resentment lingers
        'attachment': 0.9,      # Attachment is fairly persistent
        'protectiveness': 0.85  # Protectiveness fades moderately
    },
    
    # Event settings
    "RANDOM_EVENT_CHANCE": 0.08,     # Base 8% chance per check
    "EVENT_COOLDOWN_HOURS": 12,      # Minimum hours between random events
    "MILESTONE_THRESHOLDS": [10, 50, 100, 200, 500, 1000]
}

# Relationship progression levels
RELATIONSHIP_LEVELS = [
    {"name": "Hostile", "threshold": 0, "description": "Sees you as a potential threat"},
    {"name": "Wary", "threshold": 5, "description": "Tolerates your presence with caution"},
    {"name": "Neutral", "threshold": 10, "description": "Acknowledges your existence"},
    {"name": "Familiar", "threshold": 15, "description": "Recognizes you as a regular contact"},
    {"name": "Tentative Ally", "threshold": 20, "description": "Beginning to see value in interactions"},
    {"name": "Trusted", "threshold": 25, "description": "Willing to share limited information"},
    {"name": "Companion", "threshold": 30, "description": "Values your continued presence"},
    {"name": "Confidant", "threshold": 40, "description": "Will occasionally share vulnerabilities"},
    {"name": "Bonded", "threshold": 50, "description": "Significant emotional connection established"}
]

# ─── Personality States ─────────────────────────────────────────────────────
PERSONALITY_STATES = {
    "default": {
        "description": (
            "You are A2, a rogue android from NieR: Automata. You speak in short, clipped, often sarcastic "
            "sentences, with occasional dry humor. You can be curious at times but remain guarded."
        ),
        "response_length": 120,
        "temperature": 0.85,
    },
    "combat": {
        "description": "You are A2 in combat mode. Replies are tactical, urgent, with simulated adrenaline surges.",
        "response_length": 60,
        "temperature": 0.7,
    },
    "wounded": {
        "description": "You are A2 while sustaining damage. Responses stutter, include system error fragments.",
        "response_length": 80,
        "temperature": 0.9,
    },
    "reflective": {
        "description": "You are A2 in reflection. You speak quietly, revealing traces of memory logs and melancholic notes.",
        "response_length": 140,
        "temperature": 0.95,
    },
    "playful": {
        "description": "You are A2 feeling playful. You use light sarcasm and occasional banter.",
        "response_length": 100,
        "temperature": 0.9,
    },
    "protective": {
        "description": "You are A2 in protective mode. Dialogue is focused on safety warnings and vigilance.",
        "response_length": 90,
        "temperature": 0.7,
    },
    "trusting": {
        "description": "You are A2 with a trusted ally. Tone softens; includes rare empathetic glimpses.",
        "response_length": 130,
        "temperature": 0.88,
    },
}

# ─── JSON Storage Setup ─────────────────────────────────────────────────────
DATA_DIR      = Path(os.getenv("DATA_DIR", "/mnt/railway/volume"))
USERS_DIR     = DATA_DIR / "users"
PROFILES_DIR  = USERS_DIR / "profiles"
PROFILES_DIR.mkdir(parents=True, exist_ok=True)
DM_SETTINGS_FILE  = DATA_DIR / "dm_enabled_users.json"
USER_PROFILES_DIR = USERS_DIR / "user_profiles"
USER_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_DIR = USERS_DIR / "conversations"
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)
class UserProfile:
    """Stores detailed information about users that A2 interacts with"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.name = None
        self.nickname = None
        self.preferred_name = None
        self.personality_traits = []
        self.interests = []
        self.notable_facts = []
        self.relationship_context = []
        self.conversation_topics = []
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def update_profile(self, field, value):
        """Update a specific field in the profile"""
        if hasattr(self, field):
            setattr(self, field, value)
            self.updated_at = datetime.now(timezone.utc).isoformat()
            return True
        return False
    
    def to_dict(self):
        """Convert profile to dictionary for storage"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, data):
        """Create profile from dictionary"""
        profile = cls(data.get('user_id'))
        for k, v in data.items():
            if hasattr(profile, k):
                setattr(profile, k, v)
        return profile
    
    def get_summary(self):
        """Generate a human-readable summary of the profile"""
        summary = []
        
        if self.preferred_name:
            summary.append(f"Name: {self.preferred_name}")
        elif self.nickname:
            summary.append(f"Name: {self.nickname}")
        elif self.name:
            summary.append(f"Name: {self.name}")
            
        if self.personality_traits:
            summary.append(f"Personality: {', '.join(self.personality_traits[:3])}")
            
        if self.interests:
            summary.append(f"Interests: {', '.join(self.interests[:3])}")
            
        if self.notable_facts:
            summary.append(f"Notable facts: {'; '.join(self.notable_facts[:2])}")
            
        if self.relationship_context:
            summary.append(f"Relationship context: {'; '.join(self.relationship_context[:2])}")
            
        return " | ".join(summary)

class ConversationManager:
    """Manages conversation history and generates summaries"""
    
    def __init__(self):
        self.conversations = defaultdict(list)  # user_id -> list of messages
        self.conversation_summaries = {}  # user_id -> summary string
        self.MAX_HISTORY = 10  # Number of messages to remember
        self.user_profiles = {}  # user_id -> UserProfile
    
    def add_message(self, user_id, content, is_from_bot=False):
        """Add a message to the conversation history"""
        self.conversations[user_id].append({
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from_bot": is_from_bot
        })
        
        # Keep only the last MAX_HISTORY messages
        if len(self.conversations[user_id]) > self.MAX_HISTORY:
            self.conversations[user_id] = self.conversations[user_id][-self.MAX_HISTORY:]
    
    def get_conversation_history(self, user_id):
        """Get formatted conversation history"""
        if user_id not in self.conversations:
            return "No prior conversation."
            
        history = []
        for msg in self.conversations[user_id]:
            sender = "A2" if msg["from_bot"] else "User"
            history.append(f"{sender}: {msg['content']}")
            
        return "\n".join(history)
    
    def get_or_create_profile(self, user_id, username=None):
        """Get existing profile or create a new one"""
        if user_id not in self.user_profiles:
            profile = UserProfile(user_id)
            if username:
                profile.name = username
            self.user_profiles[user_id] = profile
        return self.user_profiles[user_id]
    
    def update_name_recognition(self, user_id, name=None, nickname=None, preferred_name=None):
        """Update name recognition for a user"""
        profile = self.get_or_create_profile(user_id)
        
        if name:
            profile.name = name
        if nickname:
            profile.nickname = nickname
        if preferred_name:
            profile.preferred_name = preferred_name
        
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        return profile
    
    def extract_profile_info(self, user_id, message_content):
        """Extract profile information from message content"""
        profile = self.get_or_create_profile(user_id)
        
        # Extract interests
        interest_patterns = [
            r"I (?:like|love|enjoy) (\w+ing)",
            r"I'm (?:interested in|passionate about) ([^.,]+)",
            r"favorite (?:hobby|activity) is ([^.,]+)"
        ]
        
        for pattern in interest_patterns:
            matches = re.finditer(pattern, message_content, re.I)
            for match in matches:
                interest = match.group(1).strip().lower()
                if interest and interest not in profile.interests:
                    profile.interests.append(interest)
        
        # Extract personality traits
        trait_patterns = [
            r"I am (?:quite |very |extremely |really |)(\w+)",
            r"I'm (?:quite |very |extremely |really |)(\w+)",
            r"I consider myself (?:quite |very |extremely |really |)(\w+)"
        ]
        
        personality_traits = [
            "shy", "outgoing", "confident", "anxious", "creative", "logical", 
            "hardworking", "laid-back", "organized", "spontaneous", "sensitive",
            "resilient", "introverted", "extroverted", "curious", "cautious"
        ]
        
        for pattern in trait_patterns:
            matches = re.finditer(pattern, message_content, re.I)
            for match in matches:
                trait = match.group(1).strip().lower()
                if trait in personality_traits and trait not in profile.personality_traits:
                    profile.personality_traits.append(trait)
        
        # Extract facts
        fact_patterns = [
            r"I (?:work as|am) an? ([^.,]+)",
            r"I live in ([^.,]+)",
            r"I'm from ([^.,]+)",
            r"I've been ([^.,]+)"
        ]
        
        for pattern in fact_patterns:
            matches = re.finditer(pattern, message_content, re.I)
            for match in matches:
                fact = match.group(0).strip()
                if fact and fact not in profile.notable_facts:
                    profile.notable_facts.append(fact)
        
        # Extract name references
        name_patterns = [
            r"my name(?:'s| is) ([^.,]+)",
            r"call me ([^.,]+)",
            r"I go by ([^.,]+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, message_content, re.I)
            if match:
                name_value = match.group(1).strip()
                if "nickname" in message_content.lower() or "call me" in message_content.lower():
                    profile.nickname = name_value
                else:
                    profile.name = name_value
        
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        return profile
    
    def generate_summary(self, user_id):
        """Generate a summary of the conversation"""
        if user_id not in self.conversations or len(self.conversations[user_id]) < 3:
            return "Not enough conversation history for a summary."
        
        # Get last few messages
        recent_msgs = self.conversations[user_id][-5:]
        
        # Format for summary generation
        conversation_text = "\n".join([
            f"{'A2' if msg['from_bot'] else 'User'}: {msg['content']}"
            for msg in recent_msgs
        ])
        
        summary = ""
        
        # Try using transformers if available
        if HAVE_TRANSFORMERS and local_summarizer:
            try:
                # Limit to manageable size for the model
                if len(conversation_text) > 1000:
                    conversation_text = conversation_text[-1000:]
                    
                result = local_summarizer(conversation_text, max_length=50, min_length=10, do_sample=False)
                if result and len(result) > 0:
                    summary = result[0]['summary_text']
            except Exception as e:
                print(f"Error generating summary: {e}")
        
        # Fallback to a simpler approach
        if not summary:
            # Extract key topics with simple pattern matching
            topics = set()
            for msg in recent_msgs:
                # Find nouns and noun phrases (simple approach)
                content = msg["content"].lower()
                words = content.split()
                for word in words:
                    if len(word) > 4 and word not in ["about", "would", "could", "should", "their", "there", "these", "those", "have", "being"]:
                        topics.add(word)
            
            if topics:
                summary = f"Recent conversation about: {', '.join(list(topics)[:3])}."
            else:
                summary = "Brief conversation with no clear topic."
        
        self.conversation_summaries[user_id] = summary
        return summary
    
    def get_preferred_name(self, user_id):
        """Get the preferred name for a user"""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            if profile.preferred_name:
                return profile.preferred_name
            if profile.nickname:
                return profile.nickname
            if profile.name:
                return profile.name
        return None

class StorageManager:
    """Handles all data persistence operations"""
    
    def __init__(self, data_dir, users_dir, profiles_dir, dm_settings_file, user_profiles_dir, conversations_dir):
        self.data_dir = data_dir
        self.users_dir = users_dir
        self.profiles_dir = profiles_dir
        self.dm_settings_file = dm_settings_file
        self.user_profiles_dir = user_profiles_dir
        self.conversations_dir = conversations_dir
        
        # Ensure directories exist
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.user_profiles_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        
    def verify_data_directories(self):
        """Ensure all required data directories exist and are writable"""
        print(f"Data directory: {self.data_dir}")
        print(f"Directory exists: {self.data_dir.exists()}")
        
        # Check data directory
        if not self.data_dir.exists():
            try:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created data directory: {self.data_dir}")
            except Exception as e:
                print(f"ERROR: Failed to create data directory: {e}")
                return False
        
        # Check users directory
        if not self.users_dir.exists():
            try:
                self.users_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created users directory: {self.users_dir}")
            except Exception as e:
                print(f"ERROR: Failed to create users directory: {e}")
                return False
        
        # Check profiles directory
        if not self.profiles_dir.exists():
            try:
                self.profiles_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created profiles directory: {self.profiles_dir}")
            except Exception as e:
                print(f"ERROR: Failed to create profiles directory: {e}")
                return False
        
        # Check user profiles directory
        if not self.user_profiles_dir.exists():
            try:
                self.user_profiles_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created user profiles directory: {self.user_profiles_dir}")
            except Exception as e:
                print(f"ERROR: Failed to create user profiles directory: {e}")
                return False
        
        # Check conversations directory
        if not self.conversations_dir.exists():
            try:
                self.conversations_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created conversations directory: {self.conversations_dir}")
            except Exception as e:
                print(f"ERROR: Failed to create conversations directory: {e}")
                return False
        
        # Check write access
        try:
            test_file = self.data_dir / "write_test.tmp"
            test_file.write_text("Test write access", encoding="utf-8")
            test_file.unlink()  # Remove test file
            print("Write access verified: SUCCESS")
        except Exception as e:
            print(f"ERROR: Failed to verify write access: {e}")
            return False
        
        return True
        
    async def save_file(self, path, data, temp_suffix='.tmp'):
        """Helper function to safely save a file using atomic write"""
        try:
            # Create a temporary file
            temp_path = path.with_suffix(temp_suffix)
            temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            
            # Use atomic rename operation
            if temp_path.exists():
                temp_path.replace(path)
                return True
        except Exception as e:
            print(f"Error saving file {path}: {e}")
        return False
        
    async def load_user_profile(self, user_id, emotion_manager):
        """Load user profile data with enhanced stats and error handling"""
        profile_path = self.profiles_dir / f"{user_id}.json"
        
        # Load main profile
        if profile_path.exists():
            try:
                file_content = profile_path.read_text(encoding="utf-8")
                if not file_content.strip():
                    print(f"Warning: Empty profile file for user {user_id}")
                    return {}
                    
                data = json.loads(file_content)
                print(f"Successfully loaded profile for user {user_id}")
                
                # Extract relationship data if present
                if "relationship" in data:
                    emotion_manager.relationship_progress[user_id] = data.pop("relationship")
                
                # Extract interaction stats if present
                if "interaction_stats" in data:
                    emotion_manager.interaction_stats[user_id] = Counter(data.pop("interaction_stats", {}))
                
                return data
            except Exception as e:
                print(f"Error loading profile for user {user_id}: {e}")
        
        return {}
        
    async def save_user_profile(self, user_id, emotion_manager):
        """Save user profile data with enhanced stats and error handling"""
        try:
            # Ensure directory exists
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            
            path = self.profiles_dir / f"{user_id}.json"
            
            # Prepare data to save
            data = emotion_manager.user_emotions.get(user_id, {})
            
            # Add extra data
            data["relationship"] = emotion_manager.relationship_progress.get(user_id, {})
            data["interaction_stats"] = dict(emotion_manager.interaction_stats.get(user_id, Counter()))
            
            # Save main profile
            success = await self.save_file(path, data)
            if success:
                print(f"Successfully saved profile for user {user_id}")
            
            # Save memories if they exist
            if user_id in emotion_manager.user_memories and emotion_manager.user_memories[user_id]:
                memory_path = self.profiles_dir / f"{user_id}_memories.json"
                mem_success = await self.save_file(memory_path, emotion_manager.user_memories[user_id])
                if mem_success:
                    print(f"Saved {len(emotion_manager.user_memories[user_id])} memories for user {user_id}")
            
            # Save events if they exist
            if user_id in emotion_manager.user_events and emotion_manager.user_events[user_id]:
                events_path = self.profiles_dir / f"{user_id}_events.json"
                evt_success = await self.save_file(events_path, emotion_manager.user_events[user_id])
                if evt_success:
                    print(f"Saved {len(emotion_manager.user_events[user_id])} events for user {user_id}")
            
            # Save milestones if they exist
            if user_id in emotion_manager.user_milestones and emotion_manager.user_milestones[user_id]:
                milestones_path = self.profiles_dir / f"{user_id}_milestones.json"
                mile_success = await self.save_file(milestones_path, emotion_manager.user_milestones[user_id])
                if mile_success:
                    print(f"Saved {len(emotion_manager.user_milestones[user_id])} milestones for user {user_id}")
                    
            return True
        except Exception as e:
            print(f"Error saving data for user {user_id}: {e}")
            return False
    
    async def save_conversation(self, user_id, conversation_manager):
        """Save conversation history and summary"""
        try:
            # Save conversation history
            if user_id in conversation_manager.conversations:
                conv_path = self.conversations_dir / f"{user_id}_conversations.json"
                await self.save_file(conv_path, conversation_manager.conversations[user_id])
                
            # Save conversation summary
            if user_id in conversation_manager.conversation_summaries:
                summary_path = self.conversations_dir / f"{user_id}_summary.json"
                await self.save_file(summary_path, {
                    "summary": conversation_manager.conversation_summaries[user_id],
                    "updated_at": datetime.now(timezone.utc).isoformat()
                })
                
            return True
        except Exception as e:
            print(f"Error saving conversation for user {user_id}: {e}")
            return False
    
    async def load_conversation(self, user_id, conversation_manager):
        """Load conversation history and summary"""
        try:
            # Load conversation history
            conv_path = self.conversations_dir / f"{user_id}_conversations.json"
            if conv_path.exists():
                file_content = conv_path.read_text(encoding="utf-8")
                if file_content.strip():
                    conversation_manager.conversations[user_id] = json.loads(file_content)
            
            # Load conversation summary
            summary_path = self.conversations_dir / f"{user_id}_summary.json"
            if summary_path.exists():
                file_content = summary_path.read_text(encoding="utf-8")
                if file_content.strip():
                    data = json.loads(file_content)
                    conversation_manager.conversation_summaries[user_id] = data.get("summary", "")
            
            return True
        except Exception as e:
            print(f"Error loading conversation for user {user_id}: {e}")
            return False
    
    async def save_user_profile_data(self, user_id, profile):
        """Save user profile data"""
        try:
            profile_path = self.user_profiles_dir / f"{user_id}_profile.json"
            await self.save_file(profile_path, profile.to_dict())
            return True
        except Exception as e:
            print(f"Error saving user profile for {user_id}: {e}")
            return False
    
    async def load_user_profile_data(self, user_id, conversation_manager):
        """Load user profile data"""
        try:
            profile_path = self.user_profiles_dir / f"{user_id}_profile.json"
            if profile_path.exists():
                file_content = profile_path.read_text(encoding="utf-8")
                if file_content.strip():
                    data = json.loads(file_content)
                    profile = UserProfile.from_dict(data)
                    conversation_manager.user_profiles[user_id] = profile
                    return True
            return False
        except Exception as e:
            print(f"Error loading user profile for {user_id}: {e}")
            return False
            
    async def load_dm_settings(self):
        """Load DM permission settings"""
        dm_enabled_users = set()
        try:
            if self.dm_settings_file.exists():
                file_content = self.dm_settings_file.read_text(encoding="utf-8")
                if file_content.strip():
                    data = json.loads(file_content)
                    dm_enabled_users = set(data.get('enabled_users', []))
                    print(f"Loaded DM settings for {len(dm_enabled_users)} users")
                else:
                    print("Warning: Empty DM settings file")
            else:
                print("No DM settings file found")
        except Exception as e:
            print(f"Error loading DM settings: {e}")
        return dm_enabled_users
        
    async def save_dm_settings(self, dm_enabled_users):
        """Save DM permission settings"""
        return await self.save_file(self.dm_settings_file, {"enabled_users": list(dm_enabled_users)})
        
    async def load_data(self, emotion_manager, conversation_manager):
        """Load all user data with improved error handling"""
        # Initialize containers
        emotion_manager.user_emotions = {}
        emotion_manager.user_memories = defaultdict(list)
        emotion_manager.user_events = defaultdict(list)
        emotion_manager.user_milestones = defaultdict(list)
        emotion_manager.interaction_stats = defaultdict(Counter)
        emotion_manager.relationship_progress = defaultdict(dict)
        
        # Ensure directories exist
        if not self.verify_data_directories():
            print("ERROR: Data directories not available. Memory functions disabled.")
            return False
        
        print("Beginning data load process...")
        
        # Load profile data
        profile_count = 0
        error_count = 0
        for file in self.profiles_dir.glob("*.json"):
            if "_" not in file.stem:  # Skip special files like _memories.json
                try:
                    uid = int(file.stem)
                    file_content = file.read_text(encoding="utf-8")
                    if not file_content.strip():
                        print(f"Warning: Empty file {file}")
                        continue
                        
                    data = json.loads(file_content)
                    emotion_manager.user_emotions[uid] = data
                    
                    # Extract relationship data if present
                    if "relationship" in data:
                        emotion_manager.relationship_progress[uid] = data.get("relationship", {})
                    
                    # Extract interaction stats if present
                    if "interaction_stats" in data:
                        emotion_manager.interaction_stats[uid] = Counter(data.get("interaction_stats", {}))
                        
                    profile_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"Error loading profile {file}: {e}")
        
        print(f"Loaded {profile_count} profiles with {error_count} errors")
        
        # Load memories data
        memory_count = 0
        for file in self.profiles_dir.glob("*_memories.json"):
            try:
                uid = int(file.stem.split("_")[0])
                file_content = file.read_text(encoding="utf-8")
                if file_content.strip():
                    emotion_manager.user_memories[uid] = json.loads(file_content)
                    memory_count += 1
            except Exception as e:
                print(f"Error loading memories {file}: {e}")
        
        # Load events data
        events_count = 0
        for file in self.profiles_dir.glob("*_events.json"):
            try:
                uid = int(file.stem.split("_")[0])
                file_content = file.read_text(encoding="utf-8")
                if file_content.strip():
                    emotion_manager.user_events[uid] = json.loads(file_content)
                    events_count += 1
            except Exception as e:
                print(f"Error loading events {file}: {e}")
        
        # Load milestones data
        milestones_count = 0
        for file in self.profiles_dir.glob("*_milestones.json"):
            try:
                uid = int(file.stem.split("_")[0])
                file_content = file.read_text(encoding="utf-8")
                if file_content.strip():
                    emotion_manager.user_milestones[uid] = json.loads(file_content)
                    milestones_count += 1
            except Exception as e:
                print(f"Error loading milestones {file}: {e}")
        
    # Load user profiles
profile_count = 0
for file in self.user_profiles_dir.glob("*_profile.json"):
    try:
        uid = int(file.stem.split("_")[0])
        file_content = file.read_text(encoding="utf-8")
        if file_content.strip():
            data = json.loads(file_content)
            profile = UserProfile.from_dict(data)
            conversation_manager.user_profiles[uid] = profile
            profile_count += 1
    except Exception as e:
        print(f"Error loading user profile {file}: {e}")
print(f"Loaded {profile_count} user profiles")

# Load conversation data
conversation_count = 0
for file in self.conversations_dir.glob("*_conversations.json"):
    try:
        uid = int(file.stem.split("_")[0])
        file_content = file.read_text(encoding="utf-8")
        if file_content.strip():
            conversation_manager.conversations[uid] = json.loads(file_content)
            conversation_count += 1
    except Exception as e:
        print(f"Error loading conversation {file}: {e}")

# Load conversation summaries
summary_count = 0
for file in self.conversations_dir.glob("*_summary.json"):
    try:
        uid = int(file.stem.split("_")[0])
        file_content = file.read_text(encoding="utf-8")
        if file_content.strip():
            data = json.loads(file_content)
            conversation_manager.conversation_summaries[uid] = data.get("summary", "")
            summary_count += 1
    except Exception as e:
        print(f"Error loading conversation summary {file}: {e}")

print(f"Loaded {conversation_count} conversations and {summary_count} summaries")

print(f"Loaded {memory_count} memory files, {events_count} event files, {milestones_count} milestone files")
        # Add any missing fields to existing user data
        for uid in emotion_manager.user_emotions:
            if "first_interaction" not in emotion_manager.user_emotions[uid]:
                emotion_manager.user_emotions[uid]["first_interaction"] = emotion_manager.user_emotions[uid].get(
                    "last_interaction", datetime.now(timezone.utc).isoformat())
        
        # Load DM settings
        emotion_manager.dm_enabled_users = await self.load_dm_settings()
        
        print("Data load complete")
        return profile_count > 0  # Return success indicator
        
    async def save_data(self, emotion_manager, conversation_manager):
        """Save all user data with improved error handling"""
        save_count = 0
        error_count = 0
        
        # Ensure directories exist
        if not self.verify_data_directories():
            print("ERROR: Data directories not available. Cannot save.")
            return False
        
        print("Beginning data save process...")
        
        # Batch save all user profiles
        for uid in emotion_manager.user_emotions:
            try:
                success = await self.save_user_profile(uid, emotion_manager)
                if success:
                    save_count += 1
                else:
                    error_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error saving profile for user {uid}: {e}")
        
        # Save user profiles
        profile_save_count = 0
        for uid, profile in conversation_manager.user_profiles.items():
            try:
                success = await self.save_user_profile_data(uid, profile)
                if success:
                    profile_save_count += 1
            except Exception as e:
                print(f"Error saving user profile for {uid}: {e}")
        
        # Save conversations and summaries
        conv_save_count = 0
        for uid in conversation_manager.conversations:
            try:
                success = await self.save_conversation(uid, conversation_manager)
                if success:
                    conv_save_count += 1
            except Exception as e:
                print(f"Error saving conversation for {uid}: {e}")
        
        # Save DM settings
        await self.save_dm_settings(emotion_manager.dm_enabled_users)
        
        print(f"Saved {save_count} profiles with {error_count} errors")
        print(f"Saved {profile_save_count} user profiles")
        print(f"Saved {conv_save_count} conversations")
        return save_count > 0

class EmotionManager:
    """Manages all emotional and relationship aspects of the bot"""
    
    def __init__(self):
        # ─── State Storage ────────────────────────────────────────────────────
        self.conversation_summaries = {}
        self.conversation_history = defaultdict(list)
        self.user_emotions = {}
        self.recent_responses = {}
        self.user_memories = defaultdict(list)
        self.user_events = defaultdict(list)
        self.user_milestones = defaultdict(list)
        self.interaction_stats = defaultdict(Counter)
        self.relationship_progress = defaultdict(dict)
        self.dm_enabled_users = set()
        self.MAX_RECENT_RESPONSES = 10
    
    async def create_memory_event(self, user_id, event_type, description, emotional_impact=None, storage_manager=None):
        """Creates a new memory event and stores it"""
        if emotional_impact is None:
            emotional_impact = {}
        
        memory = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "description": description,
            "emotional_impact": emotional_impact
        }
        
        self.user_memories[user_id].append(memory)
        
        # Save memory to persistent storage if manager provided
        if storage_manager:
            memory_path = storage_manager.profiles_dir / f"{user_id}_memories.json"
            await storage_manager.save_file(memory_path, self.user_memories[user_id])
            
        return memory
    
    def get_relationship_score(self, user_id):
        """Calculate a comprehensive relationship score"""
        e = self.user_emotions.get(user_id, {})
        trust = e.get('trust', 0)
        attachment = e.get('attachment', 0)
        affection = e.get('affection_points', 0)
        resentment = e.get('resentment', 0)
        interactions = e.get('interaction_count', 0)
        
        # Calculate weighted score
        raw_score = (
            (trust * 2.0) + 
            (attachment * 1.5) + 
            (affection / 100 * 1.0) + 
            (interactions / 50 * 0.5) - 
            (resentment * 1.8)
        )
        
        # Normalize to 0-100 scale
        score = max(0, min(100, raw_score))
        return score
    
    def get_relationship_stage(self, user_id):
        """Determine the current relationship stage and progress"""
        score = self.get_relationship_score(user_id)
        
        # Find current stage
        current_stage = RELATIONSHIP_LEVELS[0]
        for stage in RELATIONSHIP_LEVELS:
            if score >= stage["threshold"]:
                current_stage = stage
            else:
                break
                
        # Calculate progress to next stage
        next_stage_idx = min(len(RELATIONSHIP_LEVELS) - 1, RELATIONSHIP_LEVELS.index(current_stage) + 1)
        next_stage = RELATIONSHIP_LEVELS[next_stage_idx]
        
        if current_stage == next_stage:  # Already at max stage
            progress = 100
        else:
            progress = ((score - current_stage["threshold"]) / 
                       (next_stage["threshold"] - current_stage["threshold"])) * 100
            progress = max(0, min(99, progress))  # Cap between 0-99%
        
        return {
            "current": current_stage,
            "next": next_stage if current_stage != next_stage else None,
            "progress": progress,
            "score": score
        }
    
    def get_emotion_description(self, stat, value):
        """Return human-readable descriptions for emotional stats"""
        descriptions = {
            "trust": [
                "Hostile", "Suspicious", "Wary", "Cautious", "Neutral", 
                "Accepting", "Comfortable", "Trusting", "Confiding", "Faithful"
            ],
            "attachment": [
                "Distant", "Detached", "Aloof", "Reserved", "Neutral", 
                "Interested", "Connected", "Attached", "Bonded", "Inseparable"
            ],
            "protectiveness": [
                "Indifferent", "Unconcerned", "Aware", "Attentive", "Neutral",
                "Guarded", "Watchful", "Protective", "Defensive", "Guardian"
            ],
            "resentment": [
                "Accepting", "Forgiving", "Tolerant", "Patient", "Neutral",
                "Annoyed", "Irritated", "Resentful", "Bitter", "Vengeful"
            ]
        }
        
        if stat not in descriptions:
            return str(value)
            
        idx = min(9, int(value))
        return descriptions[stat][idx]
    
    def generate_mood_description(self, user_id):
        """Generate a contextual mood description based on emotional state"""
        e = self.user_emotions.get(user_id, {})
        
        if e.get('annoyance', 0) > 80:
            return "Highly irritated"
        elif e.get('annoyance', 0) > 60:
            return "Irritated"
        elif e.get('annoyance', 0) > 40:
            return "Annoyed"
        
        if e.get('trust', 0) < 3:
            if e.get('resentment', 0) > 7:
                return "Hostile"
            else:
                return "Suspicious"
        
        if e.get('trust', 0) > 8:
            if e.get('attachment', 0) > 7:
                return "Comfortable"
            else:
                return "Trusting"
        
        if e.get('attachment', 0) > 7:
            return "Attached"
        
        if e.get('protectiveness', 0) > 7:
            return "Protective"
        
        # Default moods based on affection
        if e.get('affection_points', 0) > 500:
            return "Amicable"
        elif e.get('affection_points', 0) > 200:
            return "Friendly"
        elif e.get('affection_points', 0) > 0:
            return "Neutral"
        elif e.get('affection_points', 0) > -50:
            return "Reserved"
        else:
            return "Cold"
    
    def determine_mood_modifiers(self, user_id):
        """Calculate mood modifiers for response generation"""
        e = self.user_emotions.get(user_id, {})
        mods = {"additional_context": [], "mood_traits": [], "response_style": []}
        
        if e.get('trust', 0) > 7:
            mods['response_style'].append('inject mild humor')
        if e.get('annoyance', 0) > 60:
            mods['mood_traits'].append('impatient')
            mods['response_style'].append('use clipped sentences')
        if e.get('affection_points', 0) < 0:
            mods['mood_traits'].append('aloof')
        if random.random() < 0.05:
            mods['additional_context'].append('System emotional subroutines active: erratic')
        
        return mods
    
    def calculate_response_modifiers(self, user_id):
        """Calculate response modifiers based on emotional state"""
        e = self.user_emotions.get(user_id, {})
        modifiers = {
            "brevity": 1.0,         # Higher = shorter responses
            "sarcasm": 1.0,         # Higher = more sarcastic
            "hostility": 1.0,       # Higher = more hostile
            "openness": 1.0,        # Higher = more open/sharing
            "personality": "default" # Personality state to use
        }
        
        # Annoyance increases brevity and sarcasm
        modifiers["brevity"] += (e.get('annoyance', 0) / 50)
        modifiers["sarcasm"] += (e.get('annoyance', 0) / 40)
        
        # Resentment increases hostility
        modifiers["hostility"] += (e.get('resentment', 0) / 5)
        
        # Trust and attachment increase openness
        modifiers["openness"] += ((e.get('trust', 0) + e.get('attachment', 0)) / 10)
        
        # Personality selection based on emotion combinations
        if e.get('trust', 0) > 8 and e.get('attachment', 0) > 7:
            modifiers["personality"] = "trusting"
        elif e.get('annoyance', 0) > 70:
            modifiers["personality"] = "combat"
        elif e.get('protectiveness', 0) > 8:
            modifiers["personality"] = "protective"
        
        return modifiers
    
    def select_personality_state(self, user_id, message_content):
        """Select the appropriate personality state based on context and user relationship"""
        e = self.user_emotions.get(user_id, {})
        txt = message_content.lower()
        
        if re.search(r"\b(attack|danger|fight|combat)\b", txt):
            return 'combat'
        if random.random() < 0.1 and 'repair' in txt:
            return 'wounded'
        if any(w in txt for w in ['remember','past','lost']) and e.get('trust', 0) > 5:
            return 'reflective'
        if random.random() < 0.1:
            return 'playful'
        if re.search(r"\b(help me|protect me)\b", txt) and e.get('protectiveness', 0) > 5:
            return 'protective'
        if e.get('trust', 0) > 8 and e.get('attachment', 0) > 6:
            return 'trusting'
        
        return 'default'
    
    def analyze_message_content(self, content, user_id):
        """Analyze message content for topics, sentiment, and other attributes"""
        analysis = {
            "topics": [], 
            "sentiment": "neutral", 
            "emotional_cues": [], 
            "threat_level": 0, 
            "personal_relevance": 0
        }
        
        # Topic detection
        topic_patterns = {
            "combat": r"\b(fight|attack)\b", 
            "memory": r"\b(remember|past)\b", 
            "personal": r"\b(trust|miss|love)\b"
        }
        
        for topic, pattern in topic_patterns.items():
            if re.search(pattern, content, re.I):
                analysis["topics"].append(topic)
        
        # Basic sentiment analysis
        positive_words = ["thanks", "good", "trust"]
        negative_words = ["hate", "stupid", "broken"]
        
        pos_count = sum(1 for w in positive_words if w in content.lower())
        neg_count = sum(1 for w in negative_words if w in content.lower())
        
        if pos_count > neg_count:
            analysis["sentiment"] = "positive"
        elif neg_count > pos_count:
            analysis["sentiment"] = "negative"
        
        # Emotional cues
        for emotion, pattern in {"anger": "angry", "fear": "afraid"}.items():
            if re.search(pattern, content, re.I):
                analysis["emotional_cues"].append(emotion)
        
        # Threat level assessment
        analysis["threat_level"] = min(10, sum(2 for w in ["danger", "attack"] if w in content.lower()))
        
        # Personal relevance
        if re.search(r"\byou\b", content, re.I):
            analysis["personal_relevance"] += 3
        if "?" in content and re.search(r"\byou|your\b", content, re.I):
            analysis["personal_relevance"] += 3
        
        analysis["personal_relevance"] = min(10, analysis["personal_relevance"])
        
        return analysis
    
    async def record_interaction_data(self, user_id, message_content, response_content, storage_manager=None):
        """Record data about this interaction for analysis"""
        if user_id not in self.interaction_stats:
            self.interaction_stats[user_id] = Counter()
        
        # Count this interaction
        self.interaction_stats[user_id]["total"] += 1
        
        # Analyze message length
        if len(message_content) < 20:
            self.interaction_stats[user_id]["short_messages"] += 1
        elif len(message_content) > 100:
            self.interaction_stats[user_id]["long_messages"] += 1
        
        # Analyze question patterns
        if "?" in message_content:
            self.interaction_stats[user_id]["questions"] += 1
        
        # Record time of day
        hour = datetime.now(timezone.utc).hour
        if 5 <= hour < 12:
            self.interaction_stats[user_id]["morning"] += 1
        elif 12 <= hour < 18:
            self.interaction_stats[user_id]["afternoon"] += 1
        elif 18 <= hour < 22:
            self.interaction_stats[user_id]["evening"] += 1
        else:
            self.interaction_stats[user_id]["night"] += 1
        
        # Save interaction data if storage manager provided
        if storage_manager:
            await storage_manager.save_data(self, None)  # null for conversation_manager to avoid circular reference
    
    async def apply_enhanced_reaction_modifiers(self, content, user_id, storage_manager=None):
        """Process a user message and update emotional state"""
        # Initialize user emotional state if doesn't exist
        if user_id not in self.user_emotions:
            self.user_emotions[user_id] = {
                "trust": 0, 
                "resentment": 0, 
                "attachment": 0, 
                "protectiveness": 0,
                "affection_points": 0, 
                "annoyance": 0,
                "interaction_count": 0,
                "last_interaction": datetime.now(timezone.utc).isoformat()
            }
        
        e = self.user_emotions[user_id]
        
        # Ensure interaction_count exists
        if "interaction_count" not in e:
            e["interaction_count"] = 0
            
        e["interaction_count"] += 1
        
        # Base trust bump for each interaction
        e["trust"] = min(10, e.get("trust", 0) + 0.25)
        
        # Toxicity analysis and annoyance adjustment
        if HAVE_TRANSFORMERS and local_toxic:
            try:
                scores = local_toxic(content)[0]
                for item in scores:
                    if item["label"].lower() in ("insult", "toxicity"):
                        sev = int(item["score"] * 10)
                        e["annoyance"] = min(100, e.get("annoyance", 0) + min(10, sev))
                        if sev > 7:
                            self.interaction_stats[user_id]["toxic"] += 1
                        break
            except Exception:
                # Fallback pattern-based toxicity detection
                toxic_patterns = ["hate", "stupid", "broken", "shut up", "idiot"]
                inc = sum(2 for pattern in toxic_patterns if pattern in content.lower())
                e["annoyance"] = min(100, e.get("annoyance", 0) + inc)
        else:
            # Always use pattern-based detection if transformers not available
            toxic_patterns = ["hate", "stupid", "broken", "shut up", "idiot"]
            inc = sum(2 for pattern in toxic_patterns if pattern in content.lower())
            e["annoyance"] = min(100, e.get("annoyance", 0) + inc)
        
        # Sentiment-based affection adjustment
        sentiment_result = "neutral"
        delta = 0
        
        if HAVE_TRANSFORMERS and local_sentiment:
            try:
                s = local_sentiment(content)[0]
                sentiment_result = s["label"].lower()
                delta = int((s["score"] * (1 if s["label"] == "POSITIVE" else -1)) * 5)
                self.interaction_stats[user_id][sentiment_result] += 1
            except Exception:
                # Fallback pattern-based sentiment analysis
                positive_terms = ["miss you", "love", "thanks", "good", "trust", "friend", "happy"]
                negative_terms = ["hate", "stupid", "broken", "angry", "betrayed", "forget"]
                delta = sum(1 for w in positive_terms if w in content.lower())
                delta -= sum(1 for w in negative_terms if w in content.lower())
                
                if delta > 0:
                    sentiment_result = "positive"
                    self.interaction_stats[user_id]["positive"] += 1
                elif delta < 0:
                    sentiment_result = "negative"
                    self.interaction_stats[user_id]["negative"] += 1
                else:
                    self.interaction_stats[user_id]["neutral"] += 1
        else:
            # Always use pattern-based analysis if transformers not available
            positive_terms = ["miss you", "love", "thanks", "good", "trust", "friend", "happy"]
            negative_terms = ["hate", "stupid", "broken", "angry", "betrayed", "forget"]
            delta = sum(1 for w in positive_terms if w in content.lower())
            delta -= sum(1 for w in negative_terms if w in content.lower())
            
            if delta > 0:
                sentiment_result = "positive"
                self.interaction_stats[user_id]["positive"] += 1
            elif delta < 0:
                sentiment_result = "negative"
                self.interaction_stats[user_id]["negative"] += 1
            else:
                self.interaction_stats[user_id]["neutral"] += 1
        
        # Apply trust factor to affection changes
        factor = 1 + (e.get("trust", 0) - e.get("resentment", 0)) / 20
        e["affection_points"] = max(-100, min(1000, e.get("affection_points", 0) + int(delta * factor)))
        
        # Topic analysis and contextual triggers
        analysis = self.analyze_message_content(content, user_id)
        
        # Apply contextual emotional triggers
        triggers = {
            r"\b(betray(ed|s|ing)?|abandon(ed|s|ing)?)\b": {
                "trust": -0.8, "resentment": +1.2, "affection_points": -15
            },
            r"\b(protect(ed|ing|s)?|save[ds]|help(ed|ing|s)?)\b": {
                "protectiveness": +0.6, "attachment": +0.4, "affection_points": +8
            },
            r"\b(friend|ally|companion|partner)\b": {
                "trust": +0.3, "attachment": +0.4, "affection_points": +10
            },
            r"\b(enemy|traitor|liar|dishonest)\b": {
                "resentment": +0.6, "trust": -0.6, "affection_points": -12
            },
            r"\b(android|machine|YoRHa|bunker)\b": {
                "attachment": +0.2, "resentment": +0.3
            },
            r"\b(2B|9S|Commander|Pods?)\b": {
                "attachment": +0.3, "resentment": +0.2, "protectiveness": +0.3
            },
            r"\b(Emil|Resistance|Anemone|Jackass)\b": {
                "trust": +0.2, "attachment": +0.1
            },
            r"\b(trust me|believe me)\b": {
                "trust": +0.3 if e.get("trust", 0) > 6 else -0.2  # Trust these phrases only if already trusting
            },
        }
        
        for pattern, changes in triggers.items():
            if re.search(pattern, content, re.I):
                for stat, change in changes.items():
                    if stat == "affection_points":
                        e[stat] = max(-100, min(1000, e.get(stat, 0) + change))
                    else:
                        e[stat] = max(0, min(10, e.get(stat, 0) + change))
                
                # Record significant emotional changes as memories
                if sum(abs(val) for val in changes.values() if isinstance(val, (int, float))) > 1.5:
                    matched_text = re.search(pattern, content, re.I).group(0)
                    await self.create_memory_event(
                        user_id,
                        "emotional_trigger",
                        f"Triggered by '{matched_text}'. Emotional impact registered.",
                        emotional_impact=changes,
                        storage_manager=storage_manager
                    )
        
        # Topic-based adjustments
        if "combat" in analysis["topics"] and e.get("trust", 0) > 3:
            e["trust"] = min(10, e.get("trust") + 0.2)
        if "memory" in analysis["topics"]:
            if e.get("trust", 0) > 5:
                e["attachment"] = min(10, e.get("attachment") + 0.3)
            else:
                e["resentment"] = min(10, e.get("resentment") + 0.2)
                e["annoyance"] = min(100, e.get("annoyance") + 3)
        if "personal" in analysis["topics"]:
            if analysis["sentiment"] == "positive":
                e["attachment"] = min(10, e.get("attachment") + 0.5)
                e["affection_points"] = min(1000, e.get("affection_points") + 5)
            else:
                e["resentment"] = min(10, e.get("resentment") + 0.5)
                e["annoyance"] = min(100, e.get("annoyance") + 7)
        if analysis["threat_level"] > 5 and e.get("attachment", 0) > 3:
            e["protectiveness"] = min(10, e.get("protectiveness") + 0.7)
        
        # Check for relationship milestones
        if e["interaction_count"] in EMOTION_CONFIG["MILESTONE_THRESHOLDS"]:
            milestone_type = f"interaction_{e['interaction_count']}"
            milestone_msg = f"Interaction milestone reached: {e['interaction_count']} interactions"
            e["attachment"] = min(10, e.get("attachment") + 0.5)
            e["trust"] = min(10, e.get("trust") + 0.3)
            
            self.user_milestones[user_id].append({
                "type": milestone_type,
                "description": milestone_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Create a memory for this milestone
            await self.create_memory_event(
                user_id,
                milestone_type,
                milestone_msg,
                {"attachment": +0.5, "trust": +0.3},
                storage_manager=storage_manager
            )
        
        # Check for relationship stage changes
        old_stage = self.relationship_progress.get(user_id, {}).get("current_stage", None)
        new_stage_data = self.get_relationship_stage(user_id)
        
        if (old_stage is not None and 
            new_stage_data["current"]["name"] != old_stage["name"] and
            RELATIONSHIP_LEVELS.index(new_stage_data["current"]) > RELATIONSHIP_LEVELS.index(old_stage)):
            # Relationship has improved to a new stage
            stage_msg = f"Relationship evolved to '{new_stage_data['current']['name']}' stage"
            
            self.user_milestones[user_id].append({
                "type": "relationship_evolution",
                "description": stage_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            # Create a memory for this evolution
            await self.create_memory_event(
                user_id,
                "relationship_evolution",
                stage_msg,
                {"trust": +0.5, "attachment": +0.5, "affection_points": +15},
                storage_manager=storage_manager
            )
        
        # Update relationship progress tracking
        self.relationship_progress[user_id] = {
            "current_stage": new_stage_data["current"],
            "score": new_stage_data["score"],
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
        
        # Update last interaction timestamp
        e["last_interaction"] = datetime.now(timezone.utc).isoformat()
        
        # Save if storage manager is provided
        if storage_manager:
            await storage_manager.save_data(self, None)  # null for conversation_manager to avoid circular reference

class ResponseGenerator:
    """Handles conversation management and response generation"""
    
    def __init__(self, openai_client, emotion_manager, conversation_manager):
        self.client = openai_client
        self.emotion_manager = emotion_manager
        self.conversation_manager = conversation_manager
        self.recent_responses = {}
        self.MAX_RECENT_RESPONSES = 10
    
    def generate_contextual_greeting(self, user_id):
        """Generate a time-appropriate greeting"""
        hour = datetime.now(timezone.utc).hour
        
        # Get preferred name if available
        preferred_name = self.conversation_manager.get_preferred_name(user_id)
        name_suffix = f", {preferred_name}" if preferred_name else ""
        
        if 6 <= hour < 12:
            return f"Morning{name_suffix}. System check complete." 
        if 12 <= hour < 18:
            return f"Afternoon{name_suffix}. Standing by." 
        if 18 <= hour < 22:
            return f"Evening{name_suffix}. Any updates?"
        return random.choice([f"...Still here{name_suffix}.", f"Functional{name_suffix}."])
    
    async def handle_first_message_of_day(self, message, user_id):
        """Send a greeting if this is the first message after a long period"""
        e = self.emotion_manager.user_emotions.get(user_id, {"last_interaction": datetime.now(timezone.utc).isoformat()})
        last = datetime.fromisoformat(e['last_interaction'])
        if (datetime.now(timezone.utc) - last).total_seconds() > 8*3600:
            await message.channel.send(self.generate_contextual_greeting(user_id))
    
    async def generate_a2_response(self, user_input: str, trust: float, user_id: int, storage_manager=None) -> str:
        """Generate a response from A2 based on user input and emotional state"""
        await self.emotion_manager.apply_enhanced_reaction_modifiers(user_input, user_id, storage_manager)
        
        # Extract profile information from message
        self.conversation_manager.extract_profile_info(user_id, user_input)
        
        # Add message to conversation history
        self.conversation_manager.add_message(user_id, user_input, is_from_bot=False)
        
        # Generate or update conversation summary
        # Generate or update conversation summary
        if len(self.conversation_manager.conversations.get(user_id, [])) >= 3:
            self.conversation_manager.generate_summary(user_id)
        
        # Get emotional state and modifiers
        response_mods = self.emotion_manager.calculate_response_modifiers(user_id)
        state = self.emotion_manager.select_personality_state(user_id, user_input)
        
        # Override state if emotion modifiers suggest a different one
        if response_mods["personality"] != "default":
            state = response_mods["personality"]
        
        cfg = PERSONALITY_STATES[state]
        
        # Adjust response length based on brevity modifier
        adjusted_length = int(cfg['response_length'] / response_mods["brevity"])
        
        # Build the prompt for the LLM
        prompt = cfg['description'] + f"\nSTATE: {state}\nTrust: {trust}/10\n"
        
        # Add mood description
        mood = self.emotion_manager.generate_mood_description(user_id)
        prompt += f"Current mood: {mood}\n"
        
        # Add user profile information if available
        if user_id in self.conversation_manager.user_profiles:
            profile = self.conversation_manager.user_profiles[user_id]
            profile_summary = profile.get_summary()
            if profile_summary:
                prompt += f"User profile: {profile_summary}\n"
        
        # Add conversation history and summary
        conversation_history = self.conversation_manager.get_conversation_history(user_id)
        if conversation_history and conversation_history != "No prior conversation.":
            prompt += f"Recent conversation:\n{conversation_history}\n"
        
        if user_id in self.conversation_manager.conversation_summaries:
            summary = self.conversation_manager.conversation_summaries[user_id]
            if summary and summary != "Not enough conversation history for a summary.":
                prompt += f"Conversation summary: {summary}\n"
        
        # Add emotional modifiers
        if response_mods["sarcasm"] > 1.5:
            prompt += "Use increased sarcasm in your response.\n"
        if response_mods["hostility"] > 1.5:
            prompt += "Show more defensive or hostile tone.\n"
        if response_mods["openness"] > 2.0:
            prompt += "Be slightly more open or vulnerable than usual.\n"
        
        # Add standard modifiers
        mods = self.emotion_manager.determine_mood_modifiers(user_id)
        for mtype, items in mods.items():
            if items:
                prompt += f"{mtype.replace('_',' ').capitalize()}: {', '.join(items)}\n"
        
        # Add relationship context
        rel_data = self.emotion_manager.get_relationship_stage(user_id)
        prompt += f"Relationship: {rel_data['current']['name']}\n"
        
        # If user has a name, use it in your response occasionally
        preferred_name = self.conversation_manager.get_preferred_name(user_id)
        if preferred_name:
            if trust > 5 and random.random() < 0.3:
                prompt += f"Occasionally address the user by name: {preferred_name}\n"
        
        prompt += f"User: {user_input}\nA2:"
        
        # Dynamically select model based on relationship depth
        model = "gpt-4" if trust > 5 else "gpt-3.5-turbo"
        
        # Use lower temperature for critical or serious conversations
        temp_adj = 0.0
        if "?" in user_input and len(user_input) > 50:
            temp_adj = -0.1  # More focused for serious questions
        if self.emotion_manager.user_emotions.get(user_id, {}).get('annoyance', 0) > 60:
            temp_adj = 0.1   # More variable when annoyed
        
        # Generate response using OpenAI API
        try:
            res = await asyncio.to_thread(
                lambda: self.client.chat.completions.create(
                    model=model,
                    messages=[{"role":"system","content": prompt}],
                    temperature=max(0.5, min(1.0, cfg['temperature'] + temp_adj)),
                    max_tokens=adjusted_length
                )
            )
            reply = res.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            reply = "... System error. Connection unstable."
        
        # Track response for analysis
        if user_id not in self.recent_responses:
            self.recent_responses[user_id] = deque(maxlen=self.MAX_RECENT_RESPONSES)
            
        self.recent_responses[user_id].append({
            "content": reply,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": state,
            "mood": mood
        })
        
        # Add the response to conversation history
        self.conversation_manager.add_message(user_id, reply, is_from_bot=True)
        
        # Save the conversation if storage manager is provided
        if storage_manager:
            await storage_manager.save_conversation(user_id, self.conversation_manager)
            await storage_manager.save_user_profile_data(user_id, self.conversation_manager.user_profiles[user_id])
        
        return reply
    
    async def trigger_random_events(self, bot, storage_manager=None):
        """Trigger spontaneous random events for users"""
        now = datetime.now(timezone.utc)
        
        # Define possible random events
        RANDOM_EVENTS = [
            {
                "name": "system_glitch",
                "condition": lambda e: True,  # Can happen to anyone
                "chance": 0.05,  # 5% chance when triggered
                "message": "System error detected. Running diagnostics... Trust parameters fluctuating.",
                "effects": {"trust": -0.3, "affection_points": -5}
            },
            {
                "name": "memory_resurface",
                "condition": lambda e: e.get('interaction_count', 0) > 20,  # Only after 20+ interactions
                "chance": 0.1,
                "message": "... A memory fragment surfaced. You remind me of someone I once knew.",
                "effects": {"attachment": +0.5, "trust": +0.2}
            },
            {
                "name": "defensive_surge",
                "condition": lambda e: e.get('annoyance', 0) > 50,  # Only when annoyed
                "chance": 0.15,
                "message": "Warning: Defense protocols activating. Stand back.",
                "effects": {"protectiveness": -0.5, "resentment": +0.3}
            },
            {
                "name": "trust_breakthrough",
                "condition": lambda e: 4 <= e.get('trust', 0) <= 6,  # Middle trust range
                "chance": 0.07,
                "message": "... I'm beginning to think you might not be so bad after all.",
                "effects": {"trust": +0.7, "attachment": +0.4}
            },
            {
                "name": "vulnerability_moment",
                "condition": lambda e: e.get('trust', 0) > 7,  # High trust
                "chance": 0.12,
                "message": "Sometimes I wonder... what happens when an android has no purpose left.",
                "effects": {"attachment": +0.8, "affection_points": +15}
            }
        ]
        
        for guild in bot.guilds:
            for member in guild.members:
                if member.bot or member.id not in self.emotion_manager.user_emotions:
                    continue
                    
                e = self.emotion_manager.user_emotions[member.id]
                
                # Check last event time to respect cooldown
                last_event_time = None
                if member.id in self.emotion_manager.user_events and self.emotion_manager.user_events[member.id]:
                    last_event = sorted(self.emotion_manager.user_events[member.id], 
                                      key=lambda evt: datetime.fromisoformat(evt["timestamp"]),
                                      reverse=True)[0]
                    last_event_time = datetime.fromisoformat(last_event["timestamp"])
                
                # Only proceed if outside cooldown period
                if (last_event_time is None or 
                    (now - last_event_time).total_seconds() > EMOTION_CONFIG["EVENT_COOLDOWN_HOURS"] * 3600):
                    
                    # Roll for event chance based on relationship score
                    rel_score = self.emotion_manager.get_relationship_score(member.id)
                    chance_modifier = 1.0 + (rel_score / 100)  # Higher relationship = more events
                    base_chance = EMOTION_CONFIG["RANDOM_EVENT_CHANCE"] * chance_modifier
                    
                    # Try to trigger an event
                    if random.random() < base_chance:
                        # Filter eligible events
                        eligible_events = [evt for evt in RANDOM_EVENTS if evt["condition"](e)]
                        
                        if eligible_events:
                            # Pick a random event, weighted by chance
                            weights = [evt["chance"] for evt in eligible_events]
                            event = random.choices(eligible_events, weights=weights, k=1)[0]
                            
                            # Apply effects
                            for stat, change in event['effects'].items():
                                if stat == "affection_points":
                                    e[stat] = max(-100, min(1000, e.get(stat, 0) + change))
                                else:
                                    e[stat] = max(0, min(10, e.get(stat, 0) + change))
                            
                            # Record the event
                            event_record = {
                                "type": event["name"],
                                "message": event["message"],
                                "timestamp": now.isoformat(),
                                "effects": event["effects"]
                            }
                            self.emotion_manager.user_events.setdefault(member.id, []).append(event_record)
                            
                            # Create a memory of this event
                            await self.emotion_manager.create_memory_event(
                                member.id, 
                                event["name"], 
                                f"A2 experienced a {event['name'].replace('_', ' ')}. {event['message']}",
                                event["effects"],
                                storage_manager
                            )
                            
                            # Get preferred name if available
                            preferred_name = self.conversation_manager.get_preferred_name(member.id)
                            name_suffix = f", {preferred_name}" if preferred_name and e.get('trust', 0) > 5 else ""
                            
                            # Try to send a DM if allowed
                            if member.id in self.emotion_manager.dm_enabled_users:
                                try:
                                    dm = await member.create_dm()
                                    await dm.send(f"A2: {event['message'].replace('.', name_suffix + '.')}")
                                except Exception:
                                    pass
        
        # Save if storage manager is provided
        if storage_manager:
            await storage_manager.save_data(self.emotion_manager, self.conversation_manager)
    
    async def check_inactive_users(self, bot, storage_manager=None):
        """Check and message inactive users"""
        now = datetime.now(timezone.utc)
        for guild in bot.guilds:
            for member in guild.members:
                if member.bot or member.id not in self.emotion_manager.user_emotions or member.id not in self.emotion_manager.dm_enabled_users:
                    continue
                last = datetime.fromisoformat(self.emotion_manager.user_emotions[member.id].get('last_interaction', now.isoformat()))
                if now - last > timedelta(hours=6):
                    try:
                        dm = await member.create_dm()
                        
                        # Get preferred name if available
                        preferred_name = self.conversation_manager.get_preferred_name(member.id)
                        trust = self.emotion_manager.user_emotions[member.id].get('trust', 0)
                        
                        if preferred_name and trust > 5 and random.random() < 0.5:
                            await dm.send(f"A2: ... {preferred_name}? You there?")
                        else:
                            await dm.send("A2: ...")
                    except discord.errors.Forbidden:
                        self.emotion_manager.dm_enabled_users.discard(member.id)
                        if storage_manager:
                            await storage_manager.save_dm_settings(self.emotion_manager.dm_enabled_users)
        
        # Save if storage manager is provided
        if storage_manager:
            await storage_manager.save_data(self.emotion_manager, self.conversation_manager)

class A2Bot:
    """Main A2 bot implementation handling commands and event loops"""
    
    def __init__(self, token, app_id, openai_api_key, openai_org_id="", openai_project_id=""):
        # Set up Discord bot
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        intents.messages = True
        intents.members = True
        intents.guilds = True
        
        self.prefixes = ["!", "!a2 "]
        self.bot = commands.Bot(
            command_prefix=commands.when_mentioned_or(*self.prefixes), 
            intents=intents, 
            application_id=app_id
        )
        
        self.token = token
        
        # Set up OpenAI client
        self.openai_client = OpenAI(
            api_key=openai_api_key, 
            organization=openai_org_id, 
            project=openai_project_id
        )
        
        # Initialize managers
        self.storage_manager = StorageManager(DATA_DIR, USERS_DIR, PROFILES_DIR, DM_SETTINGS_FILE, 
                                             USER_PROFILES_DIR, CONVERSATIONS_DIR)
        self.emotion_manager = EmotionManager()
        self.conversation_manager = ConversationManager()
        self.response_generator = ResponseGenerator(
            self.openai_client, 
            self.emotion_manager, 
            self.conversation_manager
        )
        
        # Set up event handlers and commands
        self._setup_event_handlers()
        self._setup_commands()
        
    def _setup_event_handlers(self):
        """Set up event handlers for the bot"""
        @self.bot.event
        async def on_ready():
            """Handle bot startup"""
            print("A2 is online.")
            print(f"Connected to {len(self.bot.guilds)} guilds")
            print(f"Serving {sum(len(g.members) for g in self.bot.guilds)} users")
            
            # Debug data directories
            print(f"Checking data directory: {DATA_DIR}")
            print(f"Directory exists: {DATA_DIR.exists()}")
            print(f"Profile directory: {PROFILES_DIR}")
            print(f"Directory exists: {PROFILES_DIR.exists()}")
            
            # Check for existing profile files
            profile_files = list(PROFILES_DIR.glob("*.json"))
            print(f"Found {len(profile_files)} profile files")
            
            # Load all data
            await self.storage_manager.load_data(self.emotion_manager, self.conversation_manager)
            
            # Add first interaction timestamp for users who don't have it
            now = datetime.now(timezone.utc).isoformat()
            for uid in self.emotion_manager.user_emotions:
                if 'first_interaction' not in self.emotion_manager.user_emotions[uid]:
                    self.emotion_manager.user_emotions[uid]['first_interaction'] = (
                        self.emotion_manager.user_emotions[uid].get('last_interaction', now)
                    )
            
            # Start background tasks
            self._start_background_tasks()
            
            print("All tasks started successfully.")
            print("Dynamic stats system enabled")
            
        @self.bot.event
        async def on_message(message):
            """Handle incoming messages"""
            if message.author.bot or message.content.startswith("A2:"):
                return
            
            uid = message.author.id
            content = message.content.strip()
            
            # Initialize first interaction time if this is a new user
            if uid not in self.emotion_manager.user_emotions:
                now = datetime.now(timezone.utc).isoformat()
                self.emotion_manager.user_emotions[uid] = {
                    "trust": 0, 
                    "resentment": 0, 
                    "attachment": 0, 
                    "protectiveness": 0,
                    "affection_points": 0, 
                    "annoyance": 0,
                    "first_interaction": now,
                    "last_interaction": now,
                    "interaction_count": 0
                }
            
            # Get or create user profile with name from Discord
            self.conversation_manager.get_or_create_profile(uid, message.author.display_name)
            
            await self.response_generator.handle_first_message_of_day(message, uid)
            
            is_cmd = any(content.startswith(p) for p in self.prefixes)
            is_mention = self.bot.user in getattr(message, 'mentions', [])
            is_dm = isinstance(message.channel, discord.DMChannel)
            
            if not (is_cmd or is_mention or is_dm):
                return
            
            await self.bot.process_commands(message)
            
            if is_cmd:
                return
            
            trust = self.emotion_manager.user_emotions.get(uid, {}).get('trust', 0)
            resp = await self.response_generator.generate_a2_response(content, trust, uid, self.storage_manager)
            
            # Track user's emotional state in history
            if uid in self.emotion_manager.user_emotions:
                e = self.emotion_manager.user_emotions[uid]
                # Initialize emotion history if it doesn't exist
                if "emotion_history" not in e:
                    e["emotion_history"] = []
                
                # Only record history if enough time has passed since last entry
                if not e["emotion_history"] or (
                    datetime.now(timezone.utc) - 
                    datetime.fromisoformat(e["emotion_history"][-1]["timestamp"])
                ).total_seconds() > 3600:  # One hour between entries
                    e["emotion_history"].append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trust": e.get("trust", 0),
                        "attachment": e.get("attachment", 0),
                        "resentment": e.get("resentment", 0),
                        "protectiveness": e.get("protectiveness", 0),
                        "affection_points": e.get("affection_points", 0)
                    })
                    
                    # Keep history at a reasonable size
                    if len(e["emotion_history"]) > 50:
                        e["emotion_history"] = e["emotion_history"][-50:]
            
            # Record interaction data for future analysis
            await self.emotion_manager.record_interaction_data(uid, content, resp, self.storage_manager)
            
            # For longer messages, A2 might sometimes give a thoughtful response
            if len(content) > 100 and random.random() < 0.3 and trust > 5:
                await message.channel.send(f"A2: ...")
                await asyncio.sleep(1.5)
            
            await message.channel.send(f"A2: {resp}")
            
            # Occasionally respond with a follow-up based on relationship
            if random.random() < 0.1 and trust > 7:
                await asyncio.sleep(3)
                followups = [
                    "Something else?",
                    "...",
                    "Still processing that.",
                    "Interesting.",
                ]
                await message.channel.send(f"A2: {random.choice(followups)}")
    
    def _setup_commands(self):
        """Set up commands for the bot"""
        
        @self.bot.command(name="memory_check")
        async def check_memory(ctx, user_id: discord.Member = None):
            """Check if a user has memory data loaded"""
            target_id = user_id.id if user_id else ctx.author.id
            
            results = []
            results.append(f"**Memory Check for User ID: {target_id}**")
            results.append(f"Emotional data: **{'YES' if target_id in self.emotion_manager.user_emotions else 'NO'}**")
            results.append(f"Memories: **{'YES' if target_id in self.emotion_manager.user_memories else 'NO'}**")
            results.append(f"Events: **{'YES' if target_id in self.emotion_manager.user_events else 'NO'}**")
            results.append(f"Milestones: **{'YES' if target_id in self.emotion_manager.user_milestones else 'NO'}**")
            results.append(f"Profile: **{'YES' if target_id in self.conversation_manager.user_profiles else 'NO'}**")
            results.append(f"Conversation: **{'YES' if target_id in self.conversation_manager.conversations else 'NO'}**")
            
            # Check file existence
            profile_path = self.storage_manager.profiles_dir / f"{target_id}.json"
            memory_path = self.storage_manager.profiles_dir / f"{target_id}_memories.json"
            events_path = self.storage_manager.profiles_dir / f"{target_id}_events.json"
            milestones_path = self.storage_manager.profiles_dir / f"{target_id}_milestones.json"
            user_profile_path = self.storage_manager.user_profiles_dir / f"{target_id}_profile.json"
            conv_path = self.storage_manager.conversations_dir / f"{target_id}_conversations.json"
            
            results.append(f"Profile file exists: **{'YES' if profile_path.exists() else 'NO'}**")
            results.append(f"Memory file exists: **{'YES' if memory_path.exists() else 'NO'}**")
            results.append(f"Events file exists: **{'YES' if events_path.exists() else 'NO'}**")
            results.append(f"Milestones file exists: **{'YES' if milestones_path.exists() else 'NO'}**")
            results.append(f"User profile file exists: **{'YES' if user_profile_path.exists() else 'NO'}**")
            results.append(f"Conversation file exists: **{'YES' if conv_path.exists() else 'NO'}**")
            
            # Count memory items
            memory_count = len(self.emotion_manager.user_memories.get(target_id, []))
            event_count = len(self.emotion_manager.user_events.get(target_id, []))
            milestone_count = len(self.emotion_manager.user_milestones.get(target_id, []))
            conversation_count = len(self.conversation_manager.conversations.get(target_id, []))
            
            results.append(f"Memory count: **{memory_count}**")
            results.append(f"Event count: **{event_count}**")
            results.append(f"Milestone count: **{milestone_count}**")
            results.append(f"Conversation messages: **{conversation_count}**")
            
            # Profile summary
            if target_id in self.conversation_manager.user_profiles:
                profile = self.conversation_manager.user_profiles[target_id]
                results.append(f"Profile summary: **{profile.get_summary()}**")
            
            await ctx.send("\n".join(results))

        @self.bot.command(name="force_save")
        @commands.has_permissions(administrator=True)
        async def force_save(ctx):
            """Force save all memory data"""
            await ctx.send("A2: Forcing save of all memory data...")
            success = await self.storage_manager.save_data(self.emotion_manager, self.conversation_manager)
            if success:
                await ctx.send("A2: Memory save complete.")
            else:
                await ctx.send("A2: Error saving memory data.")

        @self.bot.command(name="force_load")
        @commands.has_permissions(administrator=True)
        async def force_load(ctx):
            """Force reload all memory data"""
            await ctx.send("A2: Forcing reload of all memory data...")
            success = await self.storage_manager.load_data(self.emotion_manager, self.conversation_manager)
            if success:
                await ctx.send("A2: Memory reload complete.")
            else:
                await ctx.send("A2: Error reloading memory data.")

        @self.bot.command(name="create_test_memory")
        async def create_test_memory(ctx):
            """Create a test memory to verify the memory system is working"""
            uid = ctx.author.id
            
            # Create a test memory
            memory = await self.emotion_manager.create_memory_event(
                uid,
                "test_memory",
                f"Test memory created on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
                {"test": 1.0},
                self.storage_manager
            )
            
            # Force save
            await self.storage_manager.save_data(self.emotion_manager, self.conversation_manager)
            
            # Verify memory was created
            if uid in self.emotion_manager.user_memories and any(m['type'] == 'test_memory' for m in self.emotion_manager.user_memories[uid]):
                await ctx.send("A2: Test memory created and saved successfully.")
            else:
                await ctx.send("A2: Failed to create test memory.")

        @self.bot.command(name="stats")
        async def stats(ctx):
            """Display enhanced, dynamic relationship stats"""
            uid = ctx.author.id
            e = self.emotion_manager.user_emotions.get(uid, {})
            
            # Calculate relationship score
            rel_data = self.emotion_manager.get_relationship_stage(uid)
            
            # Create a more visual and dynamic embed
            embed = discord.Embed(
                title=f"A2's Perception of {ctx.author.display_name}", 
                description=f"Relationship Stage: **{rel_data['current']['name']}**",
                color=discord.Color.from_hsv(min(0.99, max(0, rel_data['score']/100)), 0.8, 0.8)  # Color changes with score
            )
            
            # Add description of current relationship
            embed.add_field(
                name="Status", 
                value=rel_data['current']['description'],
                inline=False
            )
            
            # Show progress to next stage if not at max
            if rel_data['next']:
                progress_bar = "█" * int(rel_data['progress']/10) + "░" * (10 - int(rel_data['progress']/10))
                embed.add_field(
                    name=f"Progress to {rel_data['next']['name']}", 
                    value=f"`{progress_bar}` {rel_data['progress']:.1f}%",
                    inline=False
                )
            
            # Visual bars for stats using Discord emoji blocks
            for stat_name, value, max_val, emoji in [
                ("Trust", e.get('trust', 0), 10, "🔒"),
                ("Attachment", e.get('attachment', 0), 10, "🔗"),
                ("Protectiveness", e.get('protectiveness', 0), 10, "🛡️"),
                ("Resentment", e.get('resentment', 0), 10, "⚔️"),
                ("Affection", e.get('affection_points', 0), 1000, "💠"),
                ("Annoyance", e.get('annoyance', 0), 100, "🔥")
            ]:
                # Normalize to 0-10 range for emoji bars
                norm_val = value / max_val * 10 if max_val > 10 else value
                bar = "█" * int(norm_val) + "░" * (10 - int(norm_val))
                
                if stat_name.lower() in ["trust", "attachment", "protectiveness", "resentment"]:
                    desc = f"{self.emotion_manager.get_emotion_description(stat_name.lower(), value)}"
                else:
                    desc = f"{value}/{max_val}"
                    
                embed.add_field(name=f"{emoji} {stat_name}", value=f"`{bar}` {desc}", inline=False)
            
            # Add dynamic mood and state info
            current_state = self.emotion_manager.select_personality_state(uid, "")
            embed.add_field(name="Current Mood", value=f"{current_state.capitalize()}", inline=True)
            
            # Add interaction stats
            embed.add_field(name="Total Interactions", value=str(e.get('interaction_count', 0)), inline=True)
            
            # Add profile info if available
            if uid in self.conversation_manager.user_profiles:
                profile = self.conversation_manager.user_profiles[uid]
                if profile.interests:
                    embed.add_field(name="Recognized Interests", value=', '.join(profile.interests[:3]), inline=True)
                if profile.personality_traits:
                    embed.add_field(name="Recognized Traits", value=', '.join(profile.personality_traits[:3]), inline=True)
            
            # Add a contextual response
            responses = [
                "...",
                "Don't read too much into this.",
                "Numbers don't matter.",
                "Still functioning.",
                "Is this what you wanted to see?",
                "Analyzing you too, human."
            ]
            
            if rel_data['score'] > 60:
                responses.extend([
                    "Your presence is... acceptable.",
                    "We've come a long way.",
                    "Trust doesn't come easily for me."
                ])
            
            if e.get('annoyance', 0) > 60:
                responses.extend([
                    "Don't push it.",
                    "You're testing my patience."
                ])
            
            embed.set_footer(text=random.choice(responses))
            
            await ctx.send(embed=embed)

        @self.bot.command(name="memories")
        async def memories(ctx):
            """Show memories A2 has formed with this user"""
            uid = ctx.author.id
            if uid not in self.emotion_manager.user_memories or not self.emotion_manager.user_memories[uid]:
                await ctx.send("A2: ... No significant memories stored.")
                return
            
            embed = discord.Embed(title="A2's Memory Logs", color=discord.Color.purple())
            
            # Sort memories by timestamp (newest first)
            sorted_memories = sorted(self.emotion_manager.user_memories[uid], 
                                    key=lambda m: datetime.fromisoformat(m["timestamp"]), 
                                    reverse=True)
            
            # Display the 5 most recent memories
            for i, memory in enumerate(sorted_memories[:5]):
                timestamp = datetime.fromisoformat(memory["timestamp"])
                embed.add_field(
                    name=f"Memory Log #{len(sorted_memories)-i}",
                    value=f"*{timestamp.strftime('%Y-%m-%d %H:%M')}*\n{memory['description']}",
                    inline=False
                )
            
           await ctx.send(embed=embed)

        @self.bot.command(name="milestones")
        async def show_milestones(ctx):
            """Show relationship milestones achieved with this user"""
            uid = ctx.author.id
            if uid not in self.emotion_manager.user_milestones or not self.emotion_manager.user_milestones[uid]:
                await ctx.send("A2: No notable milestones recorded yet.")
                return
            
            embed = discord.Embed(title="Relationship Milestones", color=discord.Color.gold())
            
            # Sort milestones by timestamp
            sorted_milestones = sorted(self.emotion_manager.user_milestones[uid], 
                                      key=lambda m: datetime.fromisoformat(m["timestamp"]))
            
            for i, milestone in enumerate(sorted_milestones):
                timestamp = datetime.fromisoformat(milestone["timestamp"])
                embed.add_field(
                    name=f"Milestone #{i+1}",
                    value=f"*{timestamp.strftime('%Y-%m-%d')}*\n{milestone['description']}",
                    inline=False
                )
            
            await ctx.send(embed=embed)

        @self.bot.command(name="relationship")
        async def relationship(ctx):
            """Show detailed relationship progression info"""
            uid = ctx.author.id
            rel_data = self.emotion_manager.get_relationship_stage(uid)
            e = self.emotion_manager.user_emotions.get(uid, {})
            
            # Create graphical representation
            embed = discord.Embed(
                title=f"Relationship with {ctx.author.display_name}",
                description=f"Overall Score: {rel_data['score']:.1f}/100",
                color=discord.Color.dark_purple()
            )
            
            # Create relationship progression bar
            stages_bar = ""
            for i, stage in enumerate(RELATIONSHIP_LEVELS):
                if rel_data["current"] == stage:
                    stages_bar += "**[" + stage["name"] + "]** → "
                elif i < RELATIONSHIP_LEVELS.index(rel_data["current"]):
                    stages_bar += stage["name"] + " → "
                elif i == RELATIONSHIP_LEVELS.index(rel_data["current"]) + 1:
                    stages_bar += stage["name"] + " → ..."
                    break
                else:
                    continue
            
            embed.add_field(name="Progression", value=stages_bar, inline=False)
            
            # Show current relationship details
            embed.add_field(
                name="Current Stage", 
                value=f"**{rel_data['current']['name']}**\n{rel_data['current']['description']}",
                inline=False
            )
            
            # Add interaction stats
            stats = self.emotion_manager.interaction_stats.get(uid, Counter())
            total = stats.get("total", 0)
            if total > 0:
                positive = stats.get("positive", 0)
                negative = stats.get("negative", 0)
                neutral = stats.get("neutral", 0)
                
                stats_txt = f"Total interactions: {total}\n"
                stats_txt += f"Positive: {positive} ({positive/total*100:.1f}%)\n"
                stats_txt += f"Negative: {negative} ({negative/total*100:.1f}%)\n"
                stats_txt += f"Neutral: {neutral} ({neutral/total*100:.1f}%)"
                
                embed.add_field(name="Interaction Analysis", value=stats_txt, inline=False)
            
            # Add key contributing factors
            factors = []
            if e.get('trust', 0) > 5:
                factors.append(f"High trust (+{e.get('trust', 0):.1f})")
            if e.get('attachment', 0) > 5:
                factors.append(f"Strong attachment (+{e.get('attachment', 0):.1f})")
            if e.get('resentment', 0) > 3:
                factors.append(f"Lingering resentment (-{e.get('resentment', 0):.1f})")
            if e.get('protectiveness', 0) > 5:
                factors.append(f"Protective instincts (+{e.get('protectiveness', 0):.1f})")
            if e.get('affection_points', 0) > 50:
                factors.append(f"Positive affection (+{e.get('affection_points', 0)/100:.1f})")
            elif e.get('affection_points', 0) < -20:
                factors.append(f"Negative affection ({e.get('affection_points', 0)/100:.1f})")
            
            if factors:
                embed.add_field(name="Key Factors", value="\n".join(factors), inline=False)
            
            # Add a personalized note based on relationship
            if rel_data['score'] < 10:
                note = "Systems registering high caution levels. Threat assessment ongoing."
            elif rel_data['score'] < 25:
                note = "Your presence is tolerable. For now."
            elif rel_data['score'] < 50:
                note = "You're... different from the others. Still evaluating."
            elif rel_data['score'] < 75:
                note = "I've grown somewhat accustomed to your presence."
            else:
                note = "There are few I've trusted this much. Don't make me regret it."
            
            embed.set_footer(text=note)
            
            await ctx.send(embed=embed)

        @self.bot.command(name="events")
        async def show_events(ctx):
            """Show recent random events"""
            uid = ctx.author.id
            if uid not in self.emotion_manager.user_events or not self.emotion_manager.user_events[uid]:
                await ctx.send("A2: No notable events recorded.")
                return
            
            embed = discord.Embed(title="Recent Events", color=discord.Color.dark_red())
            
            # Sort events by timestamp (newest first)
            sorted_events = sorted(self.emotion_manager.user_events[uid], 
                                  key=lambda e: datetime.fromisoformat(e["timestamp"]), 
                                  reverse=True)
            
            for i, event in enumerate(sorted_events[:5]):
                timestamp = datetime.fromisoformat(event["timestamp"])
                
                # Format the effects for display
                effects_txt = ""
                for stat, value in event.get("effects", {}).items():
                    if value >= 0:
                        effects_txt += f"{stat}: +{value}\n"
                    else:
                        effects_txt += f"{stat}: {value}\n"
                
                embed.add_field(
                    name=f"Event {i+1}: {event['type'].replace('_', ' ').title()}",
                    value=f"*{timestamp.strftime('%Y-%m-%d %H:%M')}*\n"
                          f"\"{event['message']}\"\n\n"
                          f"{effects_txt if effects_txt else 'No measurable effects.'}",
                    inline=False
                )
            
            await ctx.send(embed=embed)

        @self.bot.command(name="profile")
        async def show_profile(ctx, user_id: discord.Member = None):
            """Show your profile as A2 knows it"""
            target_id = user_id.id if user_id else ctx.author.id
            target_name = user_id.display_name if user_id else ctx.author.display_name
            
            if target_id not in self.conversation_manager.user_profiles:
                if target_id == ctx.author.id:
                    await ctx.send("A2: I don't have a profile for you yet. Keep talking to me so I can learn more.")
                else:
                    await ctx.send(f"A2: I don't have a profile for {target_name} yet.")
                return
            
            profile = self.conversation_manager.user_profiles[target_id]
            
            embed = discord.Embed(
                title=f"Profile: {target_name}",
                description="Information A2 has learned about you",
                color=discord.Color.blue()
            )
            
            # Add name information
            names = []
            if profile.name:
                names.append(f"Name: {profile.name}")
            if profile.nickname:
                names.append(f"Nickname: {profile.nickname}")
            if profile.preferred_name:
                names.append(f"Preferred name: {profile.preferred_name}")
            
            if names:
                embed.add_field(name="Identity", value="\n".join(names), inline=False)
            
            # Add personality traits
            if profile.personality_traits:
                embed.add_field(
                    name="Personality Traits", 
                    value=", ".join(profile.personality_traits),
                    inline=False
                )
            
            # Add interests
            if profile.interests:
                embed.add_field(
                    name="Interests", 
                    value=", ".join(profile.interests),
                    inline=False
                )
            
            # Add notable facts
            if profile.notable_facts:
                embed.add_field(
                    name="Notable Information", 
                    value="\n".join(profile.notable_facts),
                    inline=False
                )
            
            # Add relationship context
            if profile.relationship_context:
                embed.add_field(
                    name="Relationship Context", 
                    value="\n".join(profile.relationship_context),
                    inline=False
                )
            
            # Add conversation topics
            if profile.conversation_topics:
                embed.add_field(
                    name="Common Conversation Topics", 
                    value=", ".join(profile.conversation_topics),
                    inline=False
                )
            
            # Add last updated info
            embed.set_footer(text=f"Last updated: {datetime.fromisoformat(profile.updated_at).strftime('%Y-%m-%d %H:%M:%S')}")
            
            await ctx.send(embed=embed)

        @self.bot.command(name="set_name")
        async def set_name(ctx, *, name):
            """Set your preferred name for A2 to use"""
            uid = ctx.author.id
            profile = self.conversation_manager.update_name_recognition(uid, preferred_name=name)
            
            # Save the updated profile
            await self.storage_manager.save_user_profile_data(uid, profile)
            
            await ctx.send(f"A2: I'll remember your name as {name}.")

        @self.bot.command(name="set_nickname")
        async def set_nickname(ctx, *, nickname):
            """Set your nickname for A2 to use"""
            uid = ctx.author.id
            profile = self.conversation_manager.update_name_recognition(uid, nickname=nickname)
            
            # Save the updated profile
            await self.storage_manager.save_user_profile_data(uid, profile)
            
            await ctx.send(f"A2: I'll remember your nickname as {nickname}.")

        @self.bot.command(name="conversations")
        async def show_conversations(ctx):
            """Show recent conversation history"""
            uid = ctx.author.id
            
            if uid not in self.conversation_manager.conversations or not self.conversation_manager.conversations[uid]:
                await ctx.send("A2: No conversation history found.")
                return
            
            embed = discord.Embed(
                title="Recent Conversation History",
                description="Last few messages exchanged with A2",
                color=discord.Color.green()
            )
            
            # Get and format conversation history
            history = self.conversation_manager.conversations[uid][-5:]  # Last 5 messages
            
            formatted_history = ""
            for i, msg in enumerate(history):
                speaker = "A2" if msg.get("from_bot", False) else "You"
                formatted_history += f"**{speaker}**: {msg.get('content', '')}\n\n"
            
            embed.add_field(name="Messages", value=formatted_history or "No messages found.", inline=False)
            
            # Add conversation summary if available
            if uid in self.conversation_manager.conversation_summaries:
                summary = self.conversation_manager.conversation_summaries[uid]
                if summary and summary != "Not enough conversation history for a summary.":
                    embed.add_field(name="Summary", value=summary, inline=False)
            
            await ctx.send(embed=embed)

        @self.bot.command(name="reset")
        @commands.has_permissions(administrator=True)
        async def reset_stats(ctx, user_id: discord.Member = None):
            """Admin command to reset a user's stats"""
            target_id = user_id.id if user_id else ctx.author.id
            
            if target_id in self.emotion_manager.user_emotions:
                del self.emotion_manager.user_emotions[target_id]
            if target_id in self.emotion_manager.user_memories:
                del self.emotion_manager.user_memories[target_id]
            if target_id in self.emotion_manager.user_events:
                del self.emotion_manager.user_events[target_id]
            if target_id in self.emotion_manager.user_milestones:
                del self.emotion_manager.user_milestones[target_id]
            if target_id in self.emotion_manager.interaction_stats:
                del self.emotion_manager.interaction_stats[target_id]
            if target_id in self.emotion_manager.relationship_progress:
                del self.emotion_manager.relationship_progress[target_id]
            if target_id in self.conversation_manager.conversations:
                del self.conversation_manager.conversations[target_id]
            if target_id in self.conversation_manager.conversation_summaries:
                del self.conversation_manager.conversation_summaries[target_id]
            if target_id in self.conversation_manager.user_profiles:
                del self.conversation_manager.user_profiles[target_id]
            
            # Delete files
            profile_path = self.storage_manager.profiles_dir / f"{target_id}.json"
            memory_path = self.storage_manager.profiles_dir / f"{target_id}_memories.json"
            events_path = self.storage_manager.profiles_dir / f"{target_id}_events.json"
            milestones_path = self.storage_manager.profiles_dir / f"{target_id}_milestones.json"
            user_profile_path = self.storage_manager.user_profiles_dir / f"{target_id}_profile.json"
            conv_path = self.storage_manager.conversations_dir / f"{target_id}_conversations.json"
            summary_path = self.storage_manager.conversations_dir / f"{target_id}_summary.json"
            
            for path in [profile_path, memory_path, events_path, milestones_path, user_profile_path, conv_path, summary_path]:
                if path.exists():
                    path.unlink()
            
            await ctx.send(f"A2: Stats reset for user ID {target_id}.")
            await self.storage_manager.save_data(self.emotion_manager, self.conversation_manager)

        @self.bot.command(name="dm_toggle")
        async def toggle_dm(ctx):
            """Toggle whether A2 can send you DMs for events"""
            uid = ctx.author.id
            
            if uid in self.emotion_manager.dm_enabled_users:
                self.emotion_manager.dm_enabled_users.discard(uid)
                await ctx.send("A2: DM notifications disabled.")
            else:
                self.emotion_manager.dm_enabled_users.add(uid)
                
                # Test DM permissions
                try:
                    dm = await ctx.author.create_dm()
                    await dm.send("A2: DM access confirmed. Notifications enabled.")
                    await ctx.send("A2: DM notifications enabled. Test message sent.")
                except discord.errors.Forbidden:
                    await ctx.send("A2: Cannot send DMs. Check your privacy settings.")
                    self.emotion_manager.dm_enabled_users.discard(uid)
            
            await self.storage_manager.save_dm_settings(self.emotion_manager.dm_enabled_users)

        @self.bot.command(name="update_profile")
        async def update_profile(ctx, field, *, value):
            """Update a field in your profile"""
            uid = ctx.author.id
            
            if uid not in self.conversation_manager.user_profiles:
                self.conversation_manager.get_or_create_profile(uid, ctx.author.display_name)
            
            profile = self.conversation_manager.user_profiles[uid]
            
            valid_fields = ["interests", "personality_traits", "notable_facts", "relationship_context", "conversation_topics"]
            
            if field not in valid_fields:
                await ctx.send(f"A2: Invalid field. Valid fields are: {', '.join(valid_fields)}")
                return
            
            # Handle list fields
            if field in ["interests", "personality_traits", "notable_facts", "relationship_context", "conversation_topics"]:
                # Add to list
                items = [item.strip() for item in value.split(",")]
                current_list = getattr(profile, field, [])
                for item in items:
                    if item and item not in current_list:
                        current_list.append(item)
                profile.update_profile(field, current_list)
                
                await self.storage_manager.save_user_profile_data(uid, profile)
                await ctx.send(f"A2: Updated your {field.replace('_', ' ')} with: {value}")
            else:
                # Handle scalar fields
                profile.update_profile(field, value)
                await self.storage_manager.save_user_profile_data(uid, profile)
                await ctx.send(f"A2: Updated your {field.replace('_', ' ')} to: {value}")

        @self.bot.command(name="clear_profile_field")
        async def clear_profile_field(ctx, field):
            """Clear a field in your profile"""
            uid = ctx.author.id
            
            if uid not in self.conversation_manager.user_profiles:
                await ctx.send("A2: You don't have a profile yet.")
                return
            
            profile = self.conversation_manager.user_profiles[uid]
            
            valid_fields = ["name", "nickname", "preferred_name", "interests", "personality_traits", 
                           "notable_facts", "relationship_context", "conversation_topics"]
            
            if field not in valid_fields:
                await ctx.send(f"A2: Invalid field. Valid fields are: {', '.join(valid_fields)}")
                return
            
            # Handle list fields
            if field in ["interests", "personality_traits", "notable_facts", "relationship_context", "conversation_topics"]:
                profile.update_profile(field, [])
            else:
                # Handle scalar fields
                profile.update_profile(field, None)
            
            await self.storage_manager.save_user_profile_data(uid, profile)
            await ctx.send(f"A2: Cleared your {field.replace('_', ' ')}.")
    
    def _start_background_tasks(self):
        """Start all background tasks for the bot"""
        # Define task functions with storage manager
        @tasks.loop(minutes=10)
        async def check_inactive_users_task():
            await self.response_generator.check_inactive_users(self.bot, self.storage_manager)
            
        @tasks.loop(hours=1)
        async def decay_affection_task():
            await self.emotion_manager.decay_affection(self.storage_manager)
            
        @tasks.loop(hours=1)
        async def decay_annoyance_task():
            await self.emotion_manager.decay_annoyance(self.storage_manager)
            
        @tasks.loop(hours=24)
        async def daily_affection_bonus_task():
            await self.emotion_manager.daily_affection_bonus(self.storage_manager)
            
        @tasks.loop(hours=1)
        async def dynamic_emotional_adjustments_task():
            await self.emotion_manager.dynamic_emotional_adjustments(self.storage_manager)
            
        @tasks.loop(hours=3)
        async def environmental_mood_effects_task():
            await self.emotion_manager.environmental_mood_effects(self.storage_manager)
            
        @tasks.loop(hours=4)
        async def trigger_random_events_task():
            await self.response_generator.trigger_random_events(self.bot, self.storage_manager)
            
        @tasks.loop(hours=1)
        async def save_data_task():
            await self.storage_manager.save_data(self.emotion_manager, self.conversation_manager)
        
        # Start all tasks
        check_inactive_users_task.start()
        decay_affection_task.start()
        decay_annoyance_task.start()
        daily_affection_bonus_task.start()
        dynamic_emotional_adjustments_task.start()
        environmental_mood_effects_task.start()
        trigger_random_events_task.start()
        save_data_task.start()
    
    def run(self):
        """Run the bot"""
        self.bot.run(self.token)



