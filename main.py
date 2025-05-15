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

# ─── State Storage ────────────────────────────────────────────────────────
conversation_summaries = {}
conversation_history   = defaultdict(list)
user_emotions          = {}
recent_responses       = {}
user_memories = defaultdict(list)
user_events = defaultdict(list)
user_milestones = defaultdict(list)
interaction_stats = defaultdict(Counter)
relationship_progress = defaultdict(dict)
DM_ENABLED_USERS  = set()
MAX_RECENT_RESPONSES   = 10

# ─── Utility Functions ────────────────────────────────────────────────────
def verify_data_directories():
    """Ensure all required data directories exist and are writable"""
    print(f"Data directory: {DATA_DIR}")
    print(f"Directory exists: {DATA_DIR.exists()}")
    
    # Check data directory
    if not DATA_DIR.exists():
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Created data directory: {DATA_DIR}")
        except Exception as e:
            print(f"ERROR: Failed to create data directory: {e}")
            return False
    
    # Check users directory
    if not USERS_DIR.exists():
        try:
            USERS_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Created users directory: {USERS_DIR}")
        except Exception as e:
            print(f"ERROR: Failed to create users directory: {e}")
            return False
    
    # Check profiles directory
    if not PROFILES_DIR.exists():
        try:
            PROFILES_DIR.mkdir(parents=True, exist_ok=True)
            print(f"Created profiles directory: {PROFILES_DIR}")
        except Exception as e:
            print(f"ERROR: Failed to create profiles directory: {e}")
            return False
    
    # Check write access
    try:
        test_file = DATA_DIR / "write_test.tmp"
        test_file.write_text("Test write access", encoding="utf-8")
        test_file.unlink()  # Remove test file
        print("Write access verified: SUCCESS")
    except Exception as e:
        print(f"ERROR: Failed to verify write access: {e}")
        return False
    
    return True

# ─── Relationship & Emotion Functions ───────────────────────────────────────
async def create_memory_event(user_id, event_type, description, emotional_impact=None):
    """Creates a new memory event and stores it"""
    if emotional_impact is None:
        emotional_impact = {}
    
    memory = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "description": description,
        "emotional_impact": emotional_impact
    }
    
    user_memories[user_id].append(memory)
    
    # Save memory to persistent storage
    memory_path = PROFILES_DIR / f"{user_id}_memories.json"
    memory_path.write_text(json.dumps(user_memories[user_id], indent=2), encoding="utf-8")
    return memory

def get_relationship_score(user_id):
    """Calculate a comprehensive relationship score"""
    e = user_emotions.get(user_id, {})
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

def get_relationship_stage(user_id):
    """Determine the current relationship stage and progress"""
    score = get_relationship_score(user_id)
    
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

def get_emotion_description(stat, value):
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

def generate_mood_description(user_id):
    """Generate a contextual mood description based on emotional state"""
    e = user_emotions.get(user_id, {})
    
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

def determine_mood_modifiers(user_id):
    """Calculate mood modifiers for response generation"""
    e = user_emotions.get(user_id, {})
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

def calculate_response_modifiers(user_id):
    """Calculate response modifiers based on emotional state"""
    e = user_emotions.get(user_id, {})
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

def select_personality_state(user_id, message_content):
    """Select the appropriate personality state based on context and user relationship"""
    e = user_emotions.get(user_id, {})
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

def analyze_message_content(content, user_id):
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

async def record_interaction_data(user_id, message_content, response_content):
    """Record data about this interaction for analysis"""
    if user_id not in interaction_stats:
        interaction_stats[user_id] = Counter()
    
    # Count this interaction
    interaction_stats[user_id]["total"] += 1
    
    # Analyze message length
    if len(message_content) < 20:
        interaction_stats[user_id]["short_messages"] += 1
    elif len(message_content) > 100:
        interaction_stats[user_id]["long_messages"] += 1
    
    # Analyze question patterns
    if "?" in message_content:
        interaction_stats[user_id]["questions"] += 1
    
    # Record time of day
    hour = datetime.now(timezone.utc).hour
    if 5 <= hour < 12:
        interaction_stats[user_id]["morning"] += 1
    elif 12 <= hour < 18:
        interaction_stats[user_id]["afternoon"] += 1
    elif 18 <= hour < 22:
        interaction_stats[user_id]["evening"] += 1
    else:
        interaction_stats[user_id]["night"] += 1
    
    # Save interaction data
    await save_data()

# ─── Enhanced Emotional Processing ───────────────────────────────────────────
async def apply_enhanced_reaction_modifiers(content, user_id):
    """Process a user message and update emotional state"""
    # Initialize user emotional state if doesn't exist
    if user_id not in user_emotions:
        user_emotions[user_id] = {
            "trust": 0, 
            "resentment": 0, 
            "attachment": 0, 
            "protectiveness": 0,
            "affection_points": 0, 
            "annoyance": 0,
            "interaction_count": 0,
            "last_interaction": datetime.now(timezone.utc).isoformat()
        }
    
    e = user_emotions[user_id]
    
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
                        interaction_stats[user_id]["toxic"] += 1
                    break
        except Exception:
            # Fallback pattern-based toxicity detection
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
            interaction_stats[user_id][sentiment_result] += 1
        except Exception:
            # Fallback pattern-based sentiment analysis
            positive_terms = ["miss you", "love", "thanks", "good", "trust", "friend", "happy"]
            negative_terms = ["hate", "stupid", "broken", "angry", "betrayed", "forget"]
            delta = sum(1 for w in positive_terms if w in content.lower())
            delta -= sum(1 for w in negative_terms if w in content.lower())
            
            if delta > 0:
                sentiment_result = "positive"
                interaction_stats[user_id]["positive"] += 1
            elif delta < 0:
                sentiment_result = "negative"
                interaction_stats[user_id]["negative"] += 1
            else:
                interaction_stats[user_id]["neutral"] += 1
    else:
        # Always use pattern-based analysis if transformers not available
        positive_terms = ["miss you", "love", "thanks", "good", "trust", "friend", "happy"]
        negative_terms = ["hate", "stupid", "broken", "angry", "betrayed", "forget"]
        delta = sum(1 for w in positive_terms if w in content.lower())
        delta -= sum(1 for w in negative_terms if w in content.lower())
        
        if delta > 0:
            sentiment_result = "positive"
            interaction_stats[user_id]["positive"] += 1
        elif delta < 0:
            sentiment_result = "negative"
            interaction_stats[user_id]["negative"] += 1
        else:
            interaction_stats[user_id]["neutral"] += 1
    
    # Apply trust factor to affection changes
    factor = 1 + (e.get("trust", 0) - e.get("resentment", 0)) / 20
    e["affection_points"] = max(-100, min(1000, e.get("affection_points", 0) + int(delta * factor)))
    
    # Topic analysis and contextual triggers
    analysis = analyze_message_content(content, user_id)
    
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
                await create_memory_event(
                    user_id,
                    "emotional_trigger",
                    f"Triggered by '{matched_text}'. Emotional impact registered.",
                    emotional_impact=changes
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
        
        user_milestones[user_id].append({
            "type": milestone_type,
            "description": milestone_msg,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Create a memory for this milestone
        await create_memory_event(
            user_id,
            milestone_type,
            milestone_msg,
            {"attachment": +0.5, "trust": +0.3}
        )
    
    # Check for relationship stage changes
    old_stage = relationship_progress.get(user_id, {}).get("current_stage", None)
    new_stage_data = get_relationship_stage(user_id)
    
    if (old_stage is not None and 
        new_stage_data["current"]["name"] != old_stage["name"] and
        RELATIONSHIP_LEVELS.index(new_stage_data["current"]) > RELATIONSHIP_LEVELS.index(old_stage)):
        # Relationship has improved to a new stage
        stage_msg = f"Relationship evolved to '{new_stage_data['current']['name']}' stage"
        
        user_milestones[user_id].append({
            "type": "relationship_evolution",
            "description": stage_msg,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Create a memory for this evolution
        await create_memory_event(
            user_id,
            "relationship_evolution",
            stage_msg,
            {"trust": +0.5, "attachment": +0.5, "affection_points": +15}
        )
    
    # Update relationship progress tracking
    relationship_progress[user_id] = {
        "current_stage": new_stage_data["current"],
        "score": new_stage_data["score"],
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    
    # Update last interaction timestamp
    e["last_interaction"] = datetime.now(timezone.utc).isoformat()
    await save_data()

# ─── Response Generation ─────────────────────────────────────────────────────
async def generate_a2_response(user_input: str, trust: float, user_id: int) -> str:
    """Generate a response from A2 based on user input and emotional state"""
    await apply_enhanced_reaction_modifiers(user_input, user_id)
    
    # Get emotional state and modifiers
    response_mods = calculate_response_modifiers(user_id)
    state = select_personality_state(user_id, user_input)
    
    # Override state if emotion modifiers suggest a different one
    if response_mods["personality"] != "default":
        state = response_mods["personality"]
    
    cfg = PERSONALITY_STATES[state]
    
    # Adjust response length based on brevity modifier
    adjusted_length = int(cfg['response_length'] / response_mods["brevity"])
    
    # Build the prompt for the LLM
    prompt = cfg['description'] + f"\nSTATE: {state}\nTrust: {trust}/10\n"
    
    # Add mood description
    mood = generate_mood_description(user_id)
    prompt += f"Current mood: {mood}\n"
    
    # Add emotional modifiers
    if response_mods["sarcasm"] > 1.5:
        prompt += "Use increased sarcasm in your response.\n"
    if response_mods["hostility"] > 1.5:
        prompt += "Show more defensive or hostile tone.\n"
    if response_mods["openness"] > 2.0:
        prompt += "Be slightly more open or vulnerable than usual.\n"
    
    # Add standard modifiers
    mods = determine_mood_modifiers(user_id)
    for mtype, items in mods.items():
        if items:
            prompt += f"{mtype.replace('_',' ').capitalize()}: {', '.join(items)}\n"
    
    # Add relationship context
    rel_data = get_relationship_stage(user_id)
    prompt += f"Relationship: {rel_data['current']['name']}\n"
    
    # Add conversation history if available
    if conversation_summaries.get(user_id):
        prompt += f"History summary: {conversation_summaries[user_id]}\n"
    
    prompt += f"User: {user_input}\nA2:"
    
    # Dynamically select model based on relationship depth
    model = "gpt-4" if trust > 5 else "gpt-3.5-turbo"
    
    # Use lower temperature for critical or serious conversations
    temp_adj = 0.0
    if "?" in user_input and len(user_input) > 50:
        temp_adj = -0.1  # More focused for serious questions
    if user_emotions.get(user_id, {}).get('annoyance', 0) > 60:
        temp_adj = 0.1   # More variable when annoyed
    
    # Generate response using OpenAI API
    try:
        res = await asyncio.to_thread(
            lambda: client.chat.completions.create(
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
    recent_responses.setdefault(user_id, deque(maxlen=MAX_RECENT_RESPONSES)).append({
        "content": reply,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "state": state,
        "mood": mood
    })
    
    return reply

def generate_contextual_greeting(user_id):
    """Generate a time-appropriate greeting"""
    hour = datetime.now(timezone.utc).hour
    if 6 <= hour < 12:
        return "Morning. System check complete." 
    if 12 <= hour < 18:
        return "Afternoon. Standing by." 
    if 18 <= hour < 22:
        return "Evening. Any updates?"
    return random.choice(["...Still here.", "Functional."])

async def handle_first_message_of_day(message, user_id):
    """Send a greeting if this is the first message after a long period"""
    e = user_emotions.get(user_id, {"last_interaction": datetime.now(timezone.utc).isoformat()})
    last = datetime.fromisoformat(e['last_interaction'])
    if (datetime.now(timezone.utc) - last).total_seconds() > 8*3600:
        await message.channel.send(generate_contextual_greeting(user_id))

# ─── Data Persistence ────────────────────────────────────────────────────────
async def save_file(path, data, temp_suffix='.tmp'):
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

async def load_user_profile(user_id):
    """Load user profile data with enhanced stats and error handling"""
    profile_path = PROFILES_DIR / f"{user_id}.json"
    
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
                relationship_progress[user_id] = data.pop("relationship")
            
            # Extract interaction stats if present
            if "interaction_stats" in data:
                interaction_stats[user_id] = Counter(data.pop("interaction_stats"))
            
            return data
        except Exception as e:
            print(f"Error loading profile for user {user_id}: {e}")
    
    return {}

async def save_user_profile(user_id):
    """Save user profile data with enhanced stats and error handling"""
    try:
        # Ensure directory exists
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        
        path = PROFILES_DIR / f"{user_id}.json"
        
        # Prepare data to save
        data = user_emotions.get(user_id, {})
        
        # Add extra data
        data["relationship"] = relationship_progress.get(user_id, {})
        data["interaction_stats"] = dict(interaction_stats.get(user_id, Counter()))
        
        # Save main profile
        success = await save_file(path, data)
        if success:
            print(f"Successfully saved profile for user {user_id}")
        
        # Save memories if they exist
        if user_id in user_memories and user_memories[user_id]:
            memory_path = PROFILES_DIR / f"{user_id}_memories.json"
            mem_success = await save_file(memory_path, user_memories[user_id])
            if mem_success:
                print(f"Saved {len(user_memories[user_id])} memories for user {user_id}")
        
        # Save events if they exist
        if user_id in user_events and user_events[user_id]:
            events_path = PROFILES_DIR / f"{user_id}_events.json"
            evt_success = await save_file(events_path, user_events[user_id])
            if evt_success:
                print(f"Saved {len(user_events[user_id])} events for user {user_id}")
        
        # Save milestones if they exist
        if user_id in user_milestones and user_milestones[user_id]:
            milestones_path = PROFILES_DIR / f"{user_id}_milestones.json"
            mile_success = await save_file(milestones_path, user_milestones[user_id])
            if mile_success:
                print(f"Saved {len(user_milestones[user_id])} milestones for user {user_id}")
                
        return True
    except Exception as e:
        print(f"Error saving data for user {user_id}: {e}")
        return False

async def load_dm_settings():
    """Load DM permission settings"""
    global DM_ENABLED_USERS
    try:
        if DM_SETTINGS_FILE.exists():
            file_content = DM_SETTINGS_FILE.read_text(encoding="utf-8")
            if file_content.strip():
                data = json.loads(file_content)
                DM_ENABLED_USERS = set(data.get('enabled_users', []))
                print(f"Loaded DM settings for {len(DM_ENABLED_USERS)} users")
            else:
                print("Warning: Empty DM settings file")
        else:
            print("No DM settings file found")
    except Exception as e:
        print(f"Error loading DM settings: {e}")

async def save_dm_settings():
    """Save DM permission settings"""
    return await save_file(DM_SETTINGS_FILE, {"enabled_users": list(DM_ENABLED_USERS)})

async def load_data():
    """Load all user data with improved error handling"""
    global user_emotions, user_memories, user_events, user_milestones, interaction_stats, relationship_progress
    
    # Initialize containers
    user_emotions = {}
    user_memories = defaultdict(list)
    user_events = defaultdict(list)
    user_milestones = defaultdict(list)
    interaction_stats = defaultdict(Counter)
    relationship_progress = defaultdict(dict)
    
    # Ensure directories exist
    if not verify_data_directories():
        print("ERROR: Data directories not available. Memory functions disabled.")
        return False
    
    print("Beginning data load process...")
    
    # Load profile data
    profile_count = 0
    error_count = 0
    for file in PROFILES_DIR.glob("*.json"):
        if "_" not in file.stem:  # Skip special files like _memories.json
            try:
                uid = int(file.stem)
                file_content = file.read_text(encoding="utf-8")
                if not file_content.strip():
                    print(f"Warning: Empty file {file}")
                    continue
                    
                data = json.loads(file_content)
                user_emotions[uid] = data
                
                # Extract relationship data if present
                if "relationship" in data:
                    relationship_progress[uid] = data.get("relationship", {})
                
                # Extract interaction stats if present
                if "interaction_stats" in data:
                    interaction_stats[uid] = Counter(data.get("interaction_stats", {}))
                    
                profile_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error loading profile {file}: {e}")
    
    print(f"Loaded {profile_count} profiles with {error_count} errors")
    
    # Load memories data
    memory_count = 0
    for file in PROFILES_DIR.glob("*_memories.json"):
        try:
            uid = int(file.stem.split("_")[0])
            file_content = file.read_text(encoding="utf-8")
            if file_content.strip():
                user_memories[uid] = json.loads(file_content)
                memory_count += 1
        except Exception as e:
            print(f"Error loading memories {file}: {e}")
    
    # Load events data
    events_count = 0
    for file in PROFILES_DIR.glob("*_events.json"):
        try:
            uid = int(file.stem.split("_")[0])
            file_content = file.read_text(encoding="utf-8")
            if file_content.strip():
                user_events[uid] = json.loads(file_content)
                events_count += 1
        except Exception as e:
            print(f"Error loading events {file}: {e}")
    
    # Load milestones data
    milestones_count = 0
    for file in PROFILES_DIR.glob("*_milestones.json"):
        try:
            uid = int(file.stem.split("_")[0])
            file_content = file.read_text(encoding="utf-8")
            if file_content.strip():
                user_milestones[uid] = json.loads(file_content)
                milestones_count += 1
        except Exception as e:
            print(f"Error loading milestones {file}: {e}")
    
    print(f"Loaded {memory_count} memory files, {events_count} event files, {milestones_count} milestone files")
    
    # Add any missing fields to existing user data
    for uid in user_emotions:
        if "first_interaction" not in user_emotions[uid]:
            user_emotions[uid]["first_interaction"] = user_emotions[uid].get("last_interaction", 
                                                    datetime.now(timezone.utc).isoformat())
    
    # Load DM settings
    await load_dm_settings()
    
    print("Data load complete")
    return profile_count > 0  # Return success indicator

async def save_data():
    """Save all user data with improved error handling"""
    save_count = 0
    error_count = 0
    
    # Ensure directories exist
    if not verify_data_directories():
        print("ERROR: Data directories not available. Cannot save.")
        return False
    
    print("Beginning data save process...")
    
    # Batch save all user profiles
    for uid in user_emotions:
        try:
            success = await save_user_profile(uid)
            if success:
                save_count += 1
            else:
                error_count += 1
        except Exception as e:
            error_count += 1
            print(f"Error saving profile for user {uid}: {e}")
    
    # Save DM settings
    await save_dm_settings()
    
    print(f"Saved {save_count} profiles with {error_count} errors")
    return save_count > 0

# ─── Bot Setup ──────────────────────────────────────────────────────────────
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN", "")
DISCORD_APP_ID    = int(os.getenv("DISCORD_APP_ID", "0") or 0)
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
OPENAI_ORG_ID     = os.getenv("OPENAI_ORG_ID", "")
OPENAI_PROJECT_ID = os.getenv("OPENAI_PROJECT_ID", "")
client = OpenAI(api_key=OPENAI_API_KEY, organization=OPENAI_ORG_ID, project=OPENAI_PROJECT_ID)

# Setup Discord intents
intents = discord.Intents.default()
intents.message_content = True
intents.reactions       = True
intents.messages        = True
intents.members         = True
intents.guilds          = True

PREFIXES = ["!", "!a2 "]
bot = commands.Bot(command_prefix=commands.when_mentioned_or(*PREFIXES), intents=intents, application_id=DISCORD_APP_ID)

# ─── Background Tasks ──────────────────────────────────────────────────────
@tasks.loop(hours=1)
async def dynamic_emotional_adjustments():
    """Complex emotional decay and interactions between emotions"""
    now = datetime.now(timezone.utc)
    for uid, e in user_emotions.items():
        # Get time since last interaction
        last = datetime.fromisoformat(e.get('last_interaction', now.isoformat()))
        hours_since = (now - last).total_seconds() / 3600
        
        # Only decay after 12 hours of inactivity
        if hours_since > 12:
            # Each stat decays at its own rate
            for stat, multiplier in EMOTION_CONFIG["DECAY_MULTIPLIERS"].items():
                if stat in e:
                    # Higher values decay slower (more stable once established)
                    decay_rate = 0.1 * multiplier * (1 - (e[stat] / 15))
                    # Longer absence increases decay
                    time_factor = min(1.0, hours_since / 72)  # Caps at 3 days
                    total_decay = decay_rate * time_factor
                    e[stat] = max(0, e[stat] - total_decay)
        
        # Emotional interactions
        if e.get('resentment', 0) > 7 and e.get('trust', 0) > 3:
            # High resentment slowly erodes trust
            e['trust'] = max(0, e.get('trust', 0) - 0.05)
        
        if e.get('trust', 0) > 8 and e.get('resentment', 0) > 0:
            # High trust slowly erodes resentment
            e['resentment'] = max(0, e.get('resentment', 0) - 0.05)
        
        if e.get('trust', 0) > 7 and e.get('attachment', 0) < 5:
            # High trust gradually increases attachment
            e['attachment'] = min(10, e.get('attachment', 0) + 0.05)
    
    await save_data()

@tasks.loop(hours=3)
async def environmental_mood_effects():
    """Time-based and environmental effects on mood"""
    now = datetime.now(timezone.utc)
    hour = now.hour
    
    # Time of day affects mood
    if 2 <= hour < 5:  # Late night
        mood_modifier = {"trust": -0.1, "resentment": +0.1}  # More guarded at night
    elif 6 <= hour < 10:  # Morning
        mood_modifier = {"trust": +0.1, "attachment": +0.1}  # Slightly more open in morning
    elif 17 <= hour < 22:  # Evening
        mood_modifier = {"attachment": +0.1}  # More relaxed in evening
    else:
        mood_modifier = {}
    
    # Apply to users who have interacted recently (last 24h)
    for uid, e in user_emotions.items():
        last = datetime.fromisoformat(e.get('last_interaction', now.isoformat()))
        if (now - last).total_seconds() < 86400:  # Within last 24h
            for stat, change in mood_modifier.items():
                e[stat] = max(0, min(10, e.get(stat, 0) + change))
    
    await save_data()

@tasks.loop(hours=4)
async def trigger_random_events():
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
            if member.bot or member.id not in user_emotions:
                continue
                
            e = user_emotions[member.id]
            
            # Check last event time to respect cooldown
            last_event_time = None
            if member.id in user_events and user_events[member.id]:
                last_event = sorted(user_events[member.id], 
                                  key=lambda evt: datetime.fromisoformat(evt["timestamp"]),
                                  reverse=True)[0]
                last_event_time = datetime.fromisoformat(last_event["timestamp"])
            
            # Only proceed if outside cooldown period
            if (last_event_time is None or 
                (now - last_event_time).total_seconds() > EMOTION_CONFIG["EVENT_COOLDOWN_HOURS"] * 3600):
                
                # Roll for event chance based on relationship score
                rel_score = get_relationship_score(member.id)
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
                        user_events.setdefault(member.id, []).append(event_record)
                        
                        # Create a memory of this event
                        await create_memory_event(
                            member.id, 
                            event["name"], 
                            f"A2 experienced a {event['name'].replace('_', ' ')}. {event['message']}",
                            event["effects"]
                        )
                        
                        # Try to send a DM if allowed
                        if member.id in DM_ENABLED_USERS:
                            try:
                                dm = await member.create_dm()
                                await dm.send(f"A2: {event['message']}")
                            except Exception:
                                pass
    
    await save_data()

@tasks.loop(minutes=10)
async def check_inactive_users():
    """Check and message inactive users"""
    now = datetime.now(timezone.utc)
    for guild in bot.guilds:
        for member in guild.members:
            if member.bot or member.id not in user_emotions or member.id not in DM_ENABLED_USERS:
                continue
            last = datetime.fromisoformat(user_emotions[member.id].get('last_interaction', now.isoformat()))
            if now - last > timedelta(hours=6):
                try:
                    dm = await member.create_dm()
                    await dm.send("...")
                except discord.errors.Forbidden:
                    DM_ENABLED_USERS.discard(member.id)
                    await save_dm_settings()
    await save_data()

@tasks.loop(hours=1)
async def decay_affection():
    """Decay affection points over time"""
    for e in user_emotions.values():
        e['affection_points'] = max(-100, e.get('affection_points', 0) - EMOTION_CONFIG["AFFECTION_DECAY_RATE"])
    await save_data()

@tasks.loop(hours=1)
async def decay_annoyance():
    """Decay annoyance points over time"""
    for e in user_emotions.values():
        e['annoyance'] = max(0, e.get('annoyance', 0) - EMOTION_CONFIG["ANNOYANCE_DECAY_RATE"])
    await save_data()

@tasks.loop(hours=24)
async def daily_affection_bonus():
    """Add daily affection bonus to users with sufficient trust"""
    for e in user_emotions.values():
        if e.get('trust', 0) >= EMOTION_CONFIG["DAILY_BONUS_TRUST_THRESHOLD"]:
            e['affection_points'] = min(1000, e.get('affection_points', 0) + EMOTION_CONFIG["DAILY_AFFECTION_BONUS"])
    await save_data()

# ─── Bot Event Handlers ─────────────────────────────────────────────────────
@bot.event
async def on_ready():
    """Handle bot startup"""
    print("A2 is online.")
    print(f"Connected to {len(bot.guilds)} guilds")
    print(f"Serving {sum(len(g.members) for g in bot.guilds)} users")
    
    # Debug data directories
    print(f"Checking data directory: {DATA_DIR}")
    print(f"Directory exists: {DATA_DIR.exists()}")
    print(f"Profile directory: {PROFILES_DIR}")
    print(f"Directory exists: {PROFILES_DIR.exists()}")
    
    # Check for existing profile files
    profile_files = list(PROFILES_DIR.glob("*.json"))
    print(f"Found {len(profile_files)} profile files")
    
    # Add first interaction timestamp for users who don't have it
    now = datetime.now(timezone.utc).isoformat()
    for uid in user_emotions:
        if 'first_interaction' not in user_emotions[uid]:
            user_emotions[uid]['first_interaction'] = user_emotions[uid].get('last_interaction', now)
    
    # Start background tasks
    check_inactive_users.start()
    decay_affection.start()
    decay_annoyance.start()
    daily_affection_bonus.start()
    dynamic_emotional_adjustments.start()
    environmental_mood_effects.start()
    trigger_random_events.start()
    
    print("All tasks started successfully.")
    print("Dynamic stats system enabled")

@bot.event
async def on_message(message):
    """Handle incoming messages"""
    if message.author.bot or message.content.startswith("A2:"):
        return
    
    uid = message.author.id
    content = message.content.strip()
    
    # Initialize first interaction time if this is a new user
    if uid not in user_emotions:
        now = datetime.now(timezone.utc).isoformat()
        user_emotions[uid] = {
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
    
    await handle_first_message_of_day(message, uid)
    
    is_cmd = any(content.startswith(p) for p in PREFIXES)
    is_mention = bot.user in getattr(message, 'mentions', [])
    is_dm = isinstance(message.channel, discord.DMChannel)
    
    if not (is_cmd or is_mention or is_dm):
        return
    
    await bot.process_commands(message)
    
    if is_cmd:
        return
    
    trust = user_emotions.get(uid, {}).get('trust', 0)
    resp = await generate_a2_response(content, trust, uid)
    
    # Track user's emotional state in history
    if uid in user_emotions:
        e = user_emotions[uid]
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
    await record_interaction_data(uid, content, resp)
    
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

# ─── Bot Commands ────────────────────────────────────────────────────────────
@bot.command(name="memory_check")
async def check_memory(ctx, user_id: discord.Member = None):
    """Check if a user has memory data loaded"""
    target_id = user_id.id if user_id else ctx.author.id
    
    results = []
    results.append(f"**Memory Check for User ID: {target_id}**")
    results.append(f"Emotional data: **{'YES' if target_id in user_emotions else 'NO'}**")
    results.append(f"Memories: **{'YES' if target_id in user_memories else 'NO'}**")
    results.append(f"Events: **{'YES' if target_id in user_events else 'NO'}**")
    results.append(f"Milestones: **{'YES' if target_id in user_milestones else 'NO'}**")
    
    # Check file existence
    profile_path = PROFILES_DIR / f"{target_id}.json"
    memory_path = PROFILES_DIR / f"{target_id}_memories.json"
    events_path = PROFILES_DIR / f"{target_id}_events.json"
    milestones_path = PROFILES_DIR / f"{target_id}_milestones.json"
    
    results.append(f"Profile file exists: **{'YES' if profile_path.exists() else 'NO'}**")
    results.append(f"Memory file exists: **{'YES' if memory_path.exists() else 'NO'}**")
    results.append(f"Events file exists: **{'YES' if events_path.exists() else 'NO'}**")
    results.append(f"Milestones file exists: **{'YES' if milestones_path.exists() else 'NO'}**")
    
    # Check file sizes
    if profile_path.exists():
        results.append(f"Profile file size: **{profile_path.stat().st_size} bytes**")
    if memory_path.exists():
        results.append(f"Memory file size: **{memory_path.stat().st_size} bytes**")
    
    # Count memory items
    memory_count = len(user_memories.get(target_id, []))
    event_count = len(user_events.get(target_id, []))
    milestone_count = len(user_milestones.get(target_id, []))
    
    results.append(f"Memory count: **{memory_count}**")
    results.append(f"Event count: **{event_count}**")
    results.append(f"Milestone count: **{milestone_count}**")
    
    await ctx.send("\n".join(results))

@bot.command(name="force_save")
@commands.has_permissions(administrator=True)
async def force_save(ctx):
    """Force save all memory data"""
    await ctx.send("A2: Forcing save of all memory data...")
    success = await save_data()
    if success:
        await ctx.send("A2: Memory save complete.")
    else:
        await ctx.send("A2: Error saving memory data.")

@bot.command(name="force_load")
@commands.has_permissions(administrator=True)
async def force_load(ctx):
    """Force reload all memory data"""
    await ctx.send("A2: Forcing reload of all memory data...")
    success = await load_data()
    if success:
        await ctx.send("A2: Memory reload complete.")
    else:
        await ctx.send("A2: Error reloading memory data.")

@bot.command(name="create_test_memory")
async def create_test_memory(ctx):
    """Create a test memory to verify the memory system is working"""
    uid = ctx.author.id
    
    # Create a test memory
    memory = await create_memory_event(
        uid,
        "test_memory",
        f"Test memory created on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        {"test": 1.0}
    )
    
    # Force save
    await save_data()
    
    # Verify memory was created
    if uid in user_memories and any(m['type'] == 'test_memory' for m in user_memories[uid]):
        await ctx.send("A2: Test memory created and saved successfully.")
    else:
        await ctx.send("A2: Failed to create test memory.")

@bot.command(name="stats")
async def stats(ctx):
    """Display enhanced, dynamic relationship stats"""
    uid = ctx.author.id
    e = user_emotions.get(uid, {})
    
    # Calculate relationship score
    rel_data = get_relationship_stage(uid)
    
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
            desc = f"{get_emotion_description(stat_name.lower(), value)}"
        else:
            desc = f"{value}/{max_val}"
            
        embed.add_field(name=f"{emoji} {stat_name}", value=f"`{bar}` {desc}", inline=False)
    
    # Add dynamic mood and state info
    current_state = select_personality_state(uid, "")
    embed.add_field(name="Current Mood", value=f"{current_state.capitalize()}", inline=True)
    
    # Add interaction stats
    embed.add_field(name="Total Interactions", value=str(e.get('interaction_count', 0)), inline=True)
    
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

@bot.command(name="memories")
async def memories(ctx):
    """Show memories A2 has formed with this user"""
    uid = ctx.author.id
    if uid not in user_memories or not user_memories[uid]:
        await ctx.send("A2: ... No significant memories stored.")
        return
    
    embed = discord.Embed(title="A2's Memory Logs", color=discord.Color.purple())
    
    # Sort memories by timestamp (newest first)
    sorted_memories = sorted(user_memories[uid], 
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

@bot.command(name="milestones")
async def show_milestones(ctx):
    """Show relationship milestones achieved with this user"""
    uid = ctx.author.id
    if uid not in user_milestones or not user_milestones[uid]:
        await ctx.send("A2: No notable milestones recorded yet.")
        return
    
    embed = discord.Embed(title="Relationship Milestones", color=discord.Color.gold())
    
    # Sort milestones by timestamp
    sorted_milestones = sorted(user_milestones[uid], 
                              key=lambda m: datetime.fromisoformat(m["timestamp"]))
    
    for i, milestone in enumerate(sorted_milestones):
        timestamp = datetime.fromisoformat(milestone["timestamp"])
        embed.add_field(
            name=f"Milestone #{i+1}",
            value=f"*{timestamp.strftime('%Y-%m-%d')}*\n{milestone['description']}",
            inline=False
        )
    
    await ctx.send(embed=embed)

@bot.command(name="relationship")
async def relationship(ctx):
    """Show detailed relationship progression info"""
    uid = ctx.author.id
    rel_data = get_relationship_stage(uid)
    e = user_emotions.get(uid, {})
    
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
    stats = interaction_stats.get(uid, Counter())
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

@bot.command(name="events")
async def show_events(ctx):
    """Show recent random events"""
    uid = ctx.author.id
    if uid not in user_events or not user_events[uid]:
        await ctx.send("A2: No notable events recorded.")
        return
    
    embed = discord.Embed(title="Recent Events", color=discord.Color.dark_red())
    
    # Sort events by timestamp (newest first)
    sorted_events = sorted(user_events[uid], 
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

@bot.command(name="reset")
@commands.has_permissions(administrator=True)
async def reset_stats(ctx, user_id: discord.Member = None):
    """Admin command to reset a user's stats"""
    target_id = user_id.id if user_id else ctx.author.id
    
    if target_id in user_emotions:
        del user_emotions[target_id]
    if target_id in user_memories:
        del user_memories[target_id]
    if target_id in user_events:
        del user_events[target_id]
    if target_id in user_milestones:
        del user_milestones[target_id]
    if target_id in interaction_stats:
        del interaction_stats[target_id]
    if target_id in relationship_progress:
        del relationship_progress[target_id]
    
    # Delete files
    profile_path = PROFILES_DIR / f"{target_id}.json"
    memory_path = PROFILES_DIR / f"{target_id}_memories.json"
    events_path = PROFILES_DIR / f"{target_id}_events.json"
    milestones_path = PROFILES_DIR / f"{target_id}_milestones.json"
    
    for path in [profile_path, memory_path, events_path, milestones_path]:
        if path.exists():
            path.unlink()
    
    await ctx.send(f"A2: Stats reset for user ID {target_id}.")
    await save_data()

@bot.command(name="dm_toggle")
async def toggle_dm(ctx):
    """Toggle whether A2 can send you DMs for events"""
    uid = ctx.author.id
    
    if uid in DM_ENABLED_USERS:
        DM_ENABLED_USERS.discard(uid)
        await ctx.send("A2: DM notifications disabled.")
    else:
        DM_ENABLED_USERS.add(uid)
        
        # Test DM permissions
        try:
            dm = await ctx.author.create_dm()
            await dm.send("A2: DM access confirmed. Notifications enabled.")
            await ctx.send("A2: DM notifications enabled. Test message sent.")
        except discord.errors.Forbidden:
            await ctx.send("A2: Cannot send DMs. Check your privacy settings.")
            DM_ENABLED_USERS.discard(uid)
    
    await save_dm_settings()

# Main entry point
if __name__ == "__main__":
    # Verify data directories before starting
    print("Verifying data directories...")
    if not verify_data_directories():
        print("WARNING: Data directory issues detected. Memory functions may not work correctly.")
    
    # Load data before bot starts
    print("Loading memory data...")
    asyncio.get_event_loop().run_until_complete(load_data())
    
    print("Starting A2 bot...")
    bot.run(DISCORD_BOT_TOKEN)
