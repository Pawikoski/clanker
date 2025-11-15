import asyncio
import os
import random
from collections import defaultdict, deque
from datetime import datetime

from loguru import logger
from openai import AsyncOpenAI
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from clanker import settings
from clanker.prompt_manager import prompt_manager


class Clanker:
    def __init__(self):
        self.target_group_id = settings.TARGET_GROUP_ID
        self.allowed_admins = set(settings.ALLOWED_ADMIN_IDS)

        # Topic-based message storage
        # Structure: {topic_id: deque(messages)}
        self.topics = defaultdict(lambda: deque(maxlen=settings.MAX_CONTEXT_MESSAGES))
        self.current_topic = None
        self.last_message_time = None

        # Conversation state tracking (per topic)
        self.active_conversations = {}  # {topic_id: bool}
        self.last_bot_response_times = {}  # {topic_id: datetime}
        self.conversation_participants = {}  # {topic_id: set}

        # AI Client setup (GPT or Grok)
        self.ai_client = self._setup_ai_client()

    def _setup_ai_client(self):
        """Setup AI client based on the provider configuration"""
        if settings.AI_PROVIDER == "grok":
            # Grok (xAI) configuration
            api_key = os.getenv("XAI_API_KEY")
            if api_key:
                try:
                    client = AsyncOpenAI(
                        api_key=api_key, base_url="https://api.x.ai/v1"
                    )
                    logger.info("Initialized Grok (xAI) client successfully")
                    return client
                except Exception as e:
                    logger.error(f"Failed to initialize Grok client: {e}")
                    return None
            else:
                logger.warning("XAI_API_KEY not found for Grok provider")
                return None
        else:
            # OpenAI GPT configuration (default)
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    client = AsyncOpenAI(api_key=api_key)
                    logger.info("Initialized OpenAI GPT client successfully")
                    return client
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI client: {e}")
                    return None
            else:
                logger.warning("OPENAI_API_KEY not found for GPT provider")
                return None

    def _get_ai_model(self, fast_mode=False):
        """Get the appropriate model based on the AI provider"""
        if settings.AI_PROVIDER == "grok":
            # Use faster/simpler model for quick responses
            return "grok-4-fast-non-reasoning"
            if fast_mode:
                return "grok-4-fast-non-reasoning"  # Faster model for quick decisions
            else:
                return (
                    "grok-4-fast-reasoning"  # Full reasoning model for main responses
                )
        else:
            return "gpt-5.1"

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Start command handler"""
        user_id = update.effective_user.id

        if user_id in self.allowed_admins:
            await update.message.reply_text(
                "🤖 Bot2 is ready!\n\n"
                "Available commands:\n"
                "/send <message> - Send message to the target group\n"
                "/reply <message> - Reply to the last message in group\n"
                "/status - Check bot status\n"
                "/reload_prompts - Reload prompt configuration\n"
                "/help - Show this help message"
            )
        else:
            await update.message.reply_text(
                "❌ You are not authorized to use this bot."
            )

    async def help_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Help command handler"""
        user_id = update.effective_user.id

        if user_id not in self.allowed_admins:
            await update.message.reply_text(
                "❌ You are not authorized to use this bot."
            )
            return

        help_text = """
🤖 **Bot2 Commands**

**Admin Commands:**
/send <message> - Send a message to the target group
/reply <message> - Reply to the last message in the group
/status - Check current bot status and group info
/reload_prompts - Reload prompt configuration from files
/help - Show this help message

**Group Management:**
The bot automatically monitors the target group and logs all messages.
You can interact with the group through admin commands.

**Prompt Configuration:**
Edit `prompts/system_prompt.yaml` to customize bot personality and behavior.
Use `/reload_prompts` to apply changes without restarting.
        """

        await update.message.reply_text(help_text, parse_mode="Markdown")

    async def reload_prompts(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Reload prompt configuration"""
        user_id = update.effective_user.id

        if user_id not in self.allowed_admins:
            await update.message.reply_text(
                "❌ You are not authorized to use this bot."
            )
            return

        try:
            prompt_manager.reload_config()
            await update.message.reply_text(
                "✅ Prompt configuration reloaded successfully!\n"
                "Bot personality and behavior updated from `prompts/system_prompt.yaml`"
            )
            logger.info(f"Admin {user_id} reloaded prompt configuration")
        except Exception as e:
            logger.error(f"Error reloading prompts: {e}")
            await update.message.reply_text(f"❌ Error reloading prompts: {str(e)}")

    def _store_message(
        self,
        message_text: str,
        user_name: str,
        timestamp: datetime,
        telegram_topic_id: int = None,
    ):
        """Store message in appropriate Telegram topic"""
        # Use Telegram topic ID if available, otherwise use "general" topic
        topic_id = f"topic_{telegram_topic_id}" if telegram_topic_id else "general"

        # Store message in the specific Telegram topic
        message_data = {
            "text": message_text,
            "user": user_name,
            "timestamp": timestamp,
            "telegram_topic_id": telegram_topic_id,
        }

        self.topics[topic_id].append(message_data)

        # Update current topic to this Telegram topic
        self.current_topic = topic_id
        self.last_message_time = timestamp

        logger.info(
            f"Stored message in Telegram topic {topic_id}: {len(self.topics[topic_id])} messages"
        )

    def _get_context_messages(self, topic_id: str = None) -> list:
        """Get recent messages from specific Telegram topic for context"""
        if not topic_id:
            topic_id = self.current_topic

        if not topic_id or topic_id not in self.topics:
            return []

        messages = list(self.topics[topic_id])
        # Return last N messages for context
        return messages[-settings.MAX_CONTEXT_MESSAGES :]

    async def _should_respond_to_continuation(
        self, message_text: str, user_name: str, context_messages: list, topic_id: str
    ) -> bool:
        """Determine if bot should respond as conversation continuation"""
        if settings.ANSWER_ON_MENTIONS_ONLY:
            logger.info(
                "Skipping continuation response due to ANSWER_ON_MENTIONS_ONLY setting"
            )
            return False
        if not self.ai_client or not self.active_conversations.get(topic_id, False):
            return False

        # Don't respond if bot just responded in this topic (avoid spam)
        if topic_id in self.last_bot_response_times:
            time_since_bot_response = (
                datetime.now() - self.last_bot_response_times[topic_id]
            )
            if time_since_bot_response.total_seconds() < 10:  # Wait at least 10 seconds
                return False

        try:
            # Build recent context
            recent_context = "\n".join(
                [f"{msg['user']}: {msg['text']}" for msg in context_messages[-5:]]
            )

            prompt = f"""Jesteś Mariusz w czacie grupowym. Określ czy powinieneś odpowiedzieć na tę wiadomość jako kontynuację rozmowy.
            
Ostatnie wiadomości:
{recent_context}

Nowa wiadomość od {user_name}: {message_text}

Czy ta wiadomość:
1. Odnosi się do Ciebie lub Twojej poprzedniej wypowiedzi?
2. Jest pytaniem które możesz zodpowiedzieć?
3. Jest częścią rozmowy w której uczestniczysz?
4. Wymaga Twojej reakcji?

Odpowiedz TYLKO: TAK lub NIE"""

            try:
                response = await asyncio.wait_for(
                    self.ai_client.chat.completions.create(
                        model=self._get_ai_model(
                            fast_mode=True
                        ),  # Use fast mode for quick decisions
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=10,
                        temperature=0.1,
                    ),
                    timeout=20,  # Shorter timeout for decision making
                )

                decision = response.choices[0].message.content.strip().upper()
                logger.info(
                    f"Continuation decision for '{message_text[:50]}...': {decision}"
                )

                return "TAK" in decision

            except asyncio.TimeoutError:
                logger.warning("Timeout in continuation detection, defaulting to False")
                return False

        except Exception as e:
            logger.error(f"Error in continuation detection: {e}")
            return False

    async def _should_join_conversation(
        self, message_text: str, user_name: str, context_messages: list, topic_id: str
    ) -> bool:
        """Determine if bot should join conversation organically"""
        if settings.ANSWER_ON_MENTIONS_ONLY:
            logger.info(
                "Skipping organic join detection due to ANSWER_ON_MENTIONS_ONLY setting"
            )
            return False
        if not self.ai_client or self.active_conversations.get(topic_id, False):
            return False

        # Don't join too frequently in this topic
        if topic_id in self.last_bot_response_times:
            time_since_bot_response = (
                datetime.now() - self.last_bot_response_times[topic_id]
            )
            if time_since_bot_response.total_seconds() < 120:  # Wait at least 2 minutes
                return False

        # Need at least 3 recent messages for context
        if len(context_messages) < 3:
            return False

        try:
            recent_context = "\n".join(
                [f"{msg['user']}: {msg['text']}" for msg in context_messages[-5:]]
            )

            prompt = f"""Jesteś Mariusz w czacie grupowym. Oceń czy powinieneś naturalnie dołączyć do tej rozmowy.
            
Ostatnie wiadomości w rozmowie:
{recent_context}

Czy powinieneś się włączyć jeśli:
1. Rozmowa dotyczy tematu który znasz?
2. Możesz dodać wartościowy komentarz?
3. To dobry moment żeby się odezwać?
4. Ludzie mogliby skorzystać z Twojej pomocy?
5. To nie jest bardzo prywatna/osobista rozmowa?

NIE włączaj się do:
- Bardzo osobistych rozmów
- Krótkich wymian (1-2 wiadomości)
- Rozmów gdzie nikt nie potrzebuje pomocy

Odpowiedz TYLKO: TAK lub NIE"""

            try:
                response = await asyncio.wait_for(
                    self.ai_client.chat.completions.create(
                        model=self._get_ai_model(
                            fast_mode=True
                        ),  # Use fast mode for quick decisions
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=10,
                        temperature=0.1,
                    ),
                    timeout=10,  # Shorter timeout for decision making
                )

                decision = response.choices[0].message.content.strip().upper()
                logger.info(f"Join conversation decision: {decision}")

                return "TAK" in decision

            except asyncio.TimeoutError:
                logger.warning(
                    "Timeout in join conversation detection, defaulting to False"
                )
                return False

        except Exception as e:
            logger.error(f"Error in join conversation detection: {e}")
            return False

    def _update_conversation_state(self, topic_id: str, is_bot_response: bool = False):
        """Update conversation state tracking for specific topic"""
        if is_bot_response:
            self.active_conversations[topic_id] = True
            self.last_bot_response_times[topic_id] = datetime.now()
        else:
            # Check if conversation should remain active in this topic
            if topic_id in self.last_bot_response_times:
                time_since_bot = datetime.now() - self.last_bot_response_times[topic_id]
                if time_since_bot.total_seconds() > 300:  # 5 minutes
                    self.active_conversations[topic_id] = False

    async def _send_message_with_typing(self, message, response_text: str, context_bot):
        """Send message with typing indicator and realistic delay"""
        try:
            chat_id = message.chat_id

            # Send typing action
            await context_bot.send_chat_action(
                chat_id=chat_id, action=ChatAction.TYPING
            )

            # Calculate delay based on message length and AI provider
            base_delay = 1.0  # Base delay in seconds
            length_factor = (
                len(response_text) / 100.0
            )  # Additional delay per 100 characters
            random_factor = random.uniform(0.5, 2.0)  # Random variation

            # Adjust timing for different AI providers
            if settings.AI_PROVIDER == "grok":
                # Grok tends to be slower, so shorter simulated delay
                total_delay = min(base_delay + length_factor * 0.5, 3.0)
            else:
                total_delay = min(base_delay + length_factor * random_factor, 5.0)

            logger.info(
                f"Typing delay: {total_delay:.1f}s for message length: {len(response_text)} ({settings.AI_PROVIDER.upper()})"
            )

            # Wait while "typing"
            await asyncio.sleep(total_delay)

            # Send the actual message
            await message.reply_text(response_text)

        except Exception as e:
            logger.error(f"Error in typing simulation: {e}")
            # Fallback to direct message send
            await message.reply_text(response_text)

    async def _get_ai_response(
        self, user_message: str, context_messages: list, current_user: str
    ) -> str:
        """Get response from AI provider (GPT or Grok) with context"""
        if not self.ai_client:
            api_key_name = (
                "XAI_API_KEY" if settings.AI_PROVIDER == "grok" else "OPENAI_API_KEY"
            )
            return f"Sorry, I don't have access to AI right now. Please configure {api_key_name} for {settings.AI_PROVIDER.upper()}. 🤔"

        try:
            # Build context string with user analysis
            context_text = "\n".join(
                [
                    f"{msg['user']}: {msg['text']}"
                    for msg in context_messages[-8:]  # Last 8 messages for context
                ]
            )

            # Analyze users in conversation for personalization
            users_in_context = set(msg["user"] for msg in context_messages[-8:])
            users_summary = (
                f"Current users in the conversation: {', '.join(users_in_context)}"
            )

            # Create comprehensive system prompt from configuration
            system_prompt = prompt_manager.get_system_prompt()

            # Create contextual prompt
            user_prompt = f"""Last conversation context:
{context_text}

{users_summary}

Last message to you from {current_user}: {user_message}

When replying, match {current_user}’s style and the context of the conversation. ANSWER IN FIRST PERSON WITHOUT ANY INTRODUCTION."""

            # Make AI API call with provider-specific parameters and timeout
            api_params = {
                "model": self._get_ai_model(),
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_completion_tokens": 1500,
                "temperature": 0.8,  # Higher temperature for more personality
            }

            # Add OpenAI-specific parameters only for GPT
            # if settings.AI_PROVIDER != "grok":
            #     api_params["presence_penalty"] = 0.1  # Encourage variety
            #     api_params["frequency_penalty"] = 0.1  # Reduce repetition

            timeout_seconds = 60

            try:
                # Use asyncio.wait_for to add timeout
                response = await asyncio.wait_for(
                    self.ai_client.chat.completions.create(**api_params),
                    timeout=timeout_seconds,
                )
                return response.choices[0].message.content.strip()

            except asyncio.TimeoutError:
                provider_name = "Grok" if settings.AI_PROVIDER == "grok" else "OpenAI"
                logger.warning(
                    f"{provider_name} API call timed out after {timeout_seconds}s"
                )

                # Return a quick fallback response
                fallback_responses = [
                    "Hmm, I need a moment to think... 🤔",
                ]
                return random.choice(fallback_responses)

        except Exception as e:
            provider_name = "Grok" if settings.AI_PROVIDER == "grok" else "OpenAI"
            logger.error(f"Error getting {provider_name} response: {e}")
            return "Sorry, I'm having trouble thinking right now. 🤔"

    async def send_to_group(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Send a message to the target group"""
        user_id = update.effective_user.id

        if user_id not in self.allowed_admins:
            await update.message.reply_text(
                "❌ You are not authorized to use this bot."
            )
            return

        if not self.target_group_id:
            await update.message.reply_text("❌ Target group ID is not configured.")
            return

        if not context.args:
            await update.message.reply_text(
                "❌ Please provide a message to send.\nUsage: /send <your message>"
            )
            return

        message_text = " ".join(context.args)

        try:
            sent_message = await context.bot.send_message(
                chat_id=self.target_group_id, text=message_text
            )

            await update.message.reply_text(
                f"✅ Message sent to group successfully!\n"
                f"Message ID: {sent_message.message_id}"
            )

            logger.info(
                f"Admin {user_id} sent message to group {self.target_group_id}: {message_text}"
            )

        except Exception as e:
            logger.error(f"Error sending message to group: {e}")
            await update.message.reply_text(f"❌ Error sending message: {str(e)}")

    async def reply_to_group(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Reply to the last message in the target group"""
        user_id = update.effective_user.id

        if user_id not in self.allowed_admins:
            await update.message.reply_text(
                "❌ You are not authorized to use this bot."
            )
            return

        if not self.target_group_id:
            await update.message.reply_text("❌ Target group ID is not configured.")
            return

        if not context.args:
            await update.message.reply_text(
                "❌ Please provide a reply message.\nUsage: /reply <your reply>"
            )
            return

        reply_text = " ".join(context.args)

        try:
            # Get the latest message from the group
            # Note: This is a simplified approach. In a real implementation,
            # you might want to store the last message ID or implement a more sophisticated system

            await context.bot.send_message(
                chat_id=self.target_group_id, text=f"💬 Reply: {reply_text}"
            )

            await update.message.reply_text("✅ Reply sent to group successfully!")

            logger.info(
                f"Admin {user_id} replied to group {self.target_group_id}: {reply_text}"
            )

        except Exception as e:
            logger.error(f"Error sending reply to group: {e}")
            await update.message.reply_text(f"❌ Error sending reply: {str(e)}")

    async def status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show bot status"""
        user_id = update.effective_user.id

        if user_id not in self.allowed_admins:
            await update.message.reply_text(
                "❌ You are not authorized to use this bot."
            )
            return

        try:
            # Get bot info
            bot_info = await context.bot.get_me()

            # Try to get group info if group ID is set
            group_info = None
            if self.target_group_id:
                try:
                    group_chat = await context.bot.get_chat(self.target_group_id)
                    group_info = f"Group: {group_chat.title}\nType: {group_chat.type}\nID: {self.target_group_id}"
                except Exception as e:
                    group_info = f"Group ID: {self.target_group_id}\nStatus: ❌ Cannot access group - {str(e)}"
            else:
                group_info = "Group: ❌ Not configured"

            # Calculate topic statistics
            total_topics = len(self.topics)
            active_conversations = len(
                [t for t in self.active_conversations.values() if t]
            )
            current_topic_messages = (
                len(self.topics.get(self.current_topic, []))
                if self.current_topic
                else 0
            )

            # Get prompt configuration info
            bot_personality = prompt_manager._config.get("personality", {}).get(
                "name", "Unknown"
            )
            trigger_words = ", ".join(prompt_manager.get_trigger_words())

            status_text = f"""🤖 Clanker Bot Status

Bot Info:
• Name: {bot_info.first_name}
• Username: @{bot_info.username}
• ID: {bot_info.id}

Target Group:
{group_info}

Conversation Tracking:
• Telegram Topics: {total_topics} topics tracked
• Active Conversations: {active_conversations} topics
• Current Topic: {self.current_topic or 'None'}
• Messages in Current Topic: {current_topic_messages}

Configuration:
• Admins: {len(self.allowed_admins)} configured
• AI Provider: {settings.AI_PROVIDER.upper()} ({'✅ Configured' if self.ai_client else '❌ Not configured'})
• Bot Personality: {bot_personality}
• Trigger Words: {trigger_words}
• Status: ✅ Running"""

            await update.message.reply_text(status_text)

        except Exception as e:
            logger.error(f"Error getting status: {e}")
            await update.message.reply_text(f"❌ Error getting status: {str(e)}")

    async def handle_group_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle messages from the target group"""
        chat_id = update.effective_chat.id

        # Only process messages from the target group
        if chat_id != self.target_group_id:
            return

        message = update.message
        user = update.effective_user

        # Skip if no text message
        if not message or not message.text:
            return

        message_text = message.text
        user_name = user.first_name or user.username or f"User_{user.id}"
        timestamp = datetime.now()

        # Get Telegram topic ID (None if not in a topic)
        telegram_topic_id = message.message_thread_id
        topic_id = f"topic_{telegram_topic_id}" if telegram_topic_id else "general"

        # Log the message with topic info
        logger.info(
            f"Group message from {user_name} ({user.id}) in topic {topic_id}: {message_text}"
        )

        # Store message in topic-based storage
        self._store_message(message_text, user_name, timestamp, telegram_topic_id)

        # Update conversation state for this topic
        self._update_conversation_state(topic_id, is_bot_response=False)

        # Get context for decision making from this specific topic
        context_messages = self._get_context_messages(topic_id)

        should_respond = False
        response_type = None

        # Check if any trigger words are mentioned (highest priority)
        trigger_words = prompt_manager.get_trigger_words()
        if any(trigger.lower() in message_text.lower() for trigger in trigger_words):
            should_respond = True
            response_type = "mention"

        # Check if should respond as conversation continuation in this topic
        elif await self._should_respond_to_continuation(
            message_text, user_name, context_messages, topic_id
        ):
            should_respond = True
            response_type = "continuation"

        # Check if should organically join conversation in this topic
        elif await self._should_join_conversation(
            message_text, user_name, context_messages, topic_id
        ):
            should_respond = True
            response_type = "organic_join"

        if should_respond:
            try:
                # Get AI response with user personalization and topic context
                ai_response = await self._get_ai_response(
                    message_text, context_messages, user_name
                )

                # Send message with typing indicator and delay
                await self._send_message_with_typing(message, ai_response, context.bot)

                # Store bot's response in the same topic
                self._store_message(
                    ai_response, "Mariusz (Bot)", datetime.now(), telegram_topic_id
                )

                # Update conversation state for this topic
                self._update_conversation_state(topic_id, is_bot_response=True)

                logger.info(
                    f"Responded ({response_type}) to {user_name} in {topic_id} with: {ai_response[:100]}..."
                )

            except Exception as e:
                logger.error(f"Error responding to message: {e}")
                await message.reply_text("Przepraszam, mam problemy techniczne. 😅")

    async def handle_private_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle private messages to the bot"""
        user_id = update.effective_user.id

        if user_id not in self.allowed_admins:
            await update.message.reply_text(
                "❌ You are not authorized to use this bot.\n"
                "Please contact an administrator if you need access."
            )
            return

        # If it's an admin but not a command, show help
        await update.message.reply_text(
            "👋 Hello admin! Use /help to see available commands."
        )


def main():
    """Run the bot."""
    # Get the token from environment
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set!")
        return

    # Initialize bot
    bot = Clanker()

    # Create the Application
    application = Application.builder().token(settings.TELEGRAM_BOT_TOKEN).build()

    # ===== COMMAND HANDLERS =====
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("help", bot.help_command))
    application.add_handler(CommandHandler("send", bot.send_to_group))
    application.add_handler(CommandHandler("reply", bot.reply_to_group))
    application.add_handler(CommandHandler("status", bot.status))
    application.add_handler(CommandHandler("reload_prompts", bot.reload_prompts))

    # ===== MESSAGE HANDLERS =====
    # Handle group messages
    application.add_handler(
        MessageHandler(
            filters.Chat(chat_id=settings.TARGET_GROUP_ID) & filters.TEXT,
            bot.handle_group_message,
        )
    )

    # Handle private messages
    application.add_handler(
        MessageHandler(
            filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND,
            bot.handle_private_message,
        )
    )

    # Start the Bot
    logger.info("Starting Bot2...")

    try:
        application.run_polling(
            stop_signals=None,  # Don't use signals on Windows
            allowed_updates=None,
            drop_pending_updates=True,
        )
    except KeyboardInterrupt:
        logger.info("Bot2 stopped by user")
    except Exception as e:
        logger.error(f"Unhandled exception in Bot2: {e}")


if __name__ == "__main__":
    # Configuration setup
    print("🤖 Bot2 Configuration Setup")
    print("=" * 40)

    # Check AI provider and API keys
    print(f"AI Provider: {settings.AI_PROVIDER.upper()}")

    if settings.AI_PROVIDER == "grok":
        if not os.getenv("XAI_API_KEY"):
            print("⚠️  XAI_API_KEY is not configured!")
            print("Please add your xAI API key to your .env file:")
            print("   XAI_API_KEY=your_xai_api_key_here")
            print()
    else:
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  OPENAI_API_KEY is not configured!")
            print("Please add your OpenAI API key to your .env file:")
            print("   OPENAI_API_KEY=your_openai_api_key_here")
            print()

    print("To switch AI providers, set AI_PROVIDER in your .env file:")
    print("   AI_PROVIDER=gpt   # for OpenAI GPT")
    print("   AI_PROVIDER=grok  # for xAI Grok")
    print()

    # Check if group ID is configured
    if settings.TARGET_GROUP_ID is None:
        print("⚠️  TARGET_GROUP_ID is not configured!")
        print("Please set your group ID in the bot2.py file:")
        print("   TARGET_GROUP_ID = -1001234567890  # Your group ID")
        print()
        print("To get your group ID:")
        print("1. Add @userinfobot to your group")
        print("2. Send a message in the group")
        print("3. The bot will show the group ID")
        print()

    # Check if admin IDs are configured
    if not settings.ALLOWED_ADMIN_IDS:
        print("⚠️  ALLOWED_ADMIN_IDS is empty!")
        print("Please add your user ID to the ALLOWED_ADMIN_IDS list:")
        print("   ALLOWED_ADMIN_IDS = [123456789]  # Your user ID")
        print()
        print("To get your user ID:")
        print("1. Send a message to @userinfobot")
        print("2. The bot will reply with your user ID")
        print()

    if settings.TARGET_GROUP_ID is None or not settings.ALLOWED_ADMIN_IDS:
        print("❌ Bot cannot start without proper configuration.")
        print("Please configure the required settings above and restart the bot.")
        exit(1)

    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot2 stopped by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
