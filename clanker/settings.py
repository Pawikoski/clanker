import os
from typing import Literal

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TARGET_GROUP_ID = int(os.getenv("TARGET_GROUP_ID"))
ALLOWED_ADMIN_IDS = [
    int(admin_id) for admin_id in os.getenv("ALLOWED_ADMIN_IDS").split(",")
]
MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", 20))
TOPIC_TIMEOUT_MINUTES = int(os.getenv("TOPIC_TIMEOUT_MINUTES", 30))

# AI Provider Configuration
AI_PROVIDER: Literal["gpt", "grok"] = os.getenv("AI_PROVIDER", "grok").lower()

ANSWER_ON_MENTIONS_ONLY = os.getenv("ANSWER_ON_MENTIONS_ONLY", "true").lower() == "true"
