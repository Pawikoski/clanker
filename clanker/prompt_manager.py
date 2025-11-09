from pathlib import Path
from typing import Any, Dict, List

import yaml
from loguru import logger


class PromptManager:
    """Manages system prompts and user configurations"""

    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            # Default to prompts directory in project root
            self.prompts_dir = Path(__file__).parent.parent / "prompts"
        else:
            self.prompts_dir = Path(prompts_dir)

        self.system_prompt_file = self.prompts_dir / "system_prompt.yaml"
        self.bot_name = "Clanker"
        self._config = None
        self._load_config()

    def _load_config(self):
        """Load prompt configuration from YAML file"""
        try:
            # If system_prompt.yaml doesn't exist, copy from example
            if not self.system_prompt_file.exists():
                example_file = self.prompts_dir / "system_prompt.example.yaml"
                if example_file.exists():
                    # Copy example to main config file
                    import shutil

                    shutil.copy2(example_file, self.system_prompt_file)
                    logger.info(f"Created {self.system_prompt_file} from example file")
                else:
                    # Create default config file if example doesn't exist either
                    self._create_default_config_file()
                    logger.info(
                        f"Created default configuration file: {self.system_prompt_file}"
                    )

            # Always load from the YAML file
            with open(self.system_prompt_file, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
            logger.info(f"Loaded prompt configuration from {self.system_prompt_file}")

        except Exception as e:
            logger.error(f"Error loading prompt config: {e}")
            # Fallback to in-memory defaults only if file operations fail
            self._config = self._get_default_config()

    def _create_default_config_file(self):
        """Create a default configuration file"""
        # Ensure prompts directory exists
        self.prompts_dir.mkdir(exist_ok=True)

        default_config = self._get_default_config()

        with open(self.system_prompt_file, "w", encoding="utf-8") as f:
            yaml.dump(
                default_config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                indent=2,
            )

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file doesn't exist"""
        return {
            "personality": {
                "name": "Clanker",
                "description": "a helpful and friendly chatbot",
            },
            "communication_rules": [
                "Be helpful and friendly",
                "Respond in same language as the user",
            ],
            "user_mappings": {},
            "trigger_words": ["clanker"],
            "politically_incorrect": {"enabled": False},
            "boundaries": [],
            "special_users": {},
        }

    def get_system_prompt(self) -> str:
        """Generate the complete system prompt from configuration"""
        config = self._config

        # Build the system prompt dynamically
        prompt_parts = []

        # Personality introduction
        personality = config.get("personality", {})
        name = personality.get("name", "Assistant")
        description = personality.get("description", "helpful assistant")
        prompt_parts.append(f"You are {name} - {description}.")

        # Communication rules
        rules = config.get("communication_rules", [])
        if rules:
            prompt_parts.append("\nZASADY KOMUNIKACJI:")
            for i, rule in enumerate(rules, 1):
                prompt_parts.append(f"{i}. {rule}")

        # Personalization
        personalization = config.get("personalization", [])
        if personalization:
            prompt_parts.append("\nPERSONALIZACJA:")
            for rule in personalization:
                prompt_parts.append(f"- {rule}")

        # Response style
        response_style = config.get("response_style", [])
        if response_style:
            prompt_parts.append("\nSTYL ODPOWIEDZI:")
            for rule in response_style:
                prompt_parts.append(f"- {rule}")

        # Reactions
        reactions = config.get("reactions", {})
        if reactions:
            prompt_parts.append("\nREAKCJE NA RÓŻNE SYTUACJE:")
            for situation, response in reactions.items():
                prompt_parts.append(
                    f"- {situation.replace('_', ' ').title()}: {response}"
                )

        # Special instructions
        special_instructions = config.get("special_instructions", [])
        if special_instructions:
            prompt_parts.append("\nSPECJALNE INSTRUKCJE:")
            for instruction in special_instructions:
                prompt_parts.append(f"- {instruction}")

        # Politically incorrect behavior
        pc_settings = config.get("politically_incorrect", {})
        if pc_settings.get("enabled", False):
            guidelines = pc_settings.get("guidelines", [])
            if guidelines:
                prompt_parts.append("\nBĄDŹ POLITICALLY INCORRECT:")
                for guideline in guidelines:
                    prompt_parts.append(f"- {guideline}")

        # Boundaries
        boundaries = config.get("boundaries", [])
        if boundaries:
            prompt_parts.append("\nZASADY GRANIC:")
            for boundary in boundaries:
                prompt_parts.append(f"- {boundary}")

        # Special users
        special_users = config.get("special_users", {})
        if special_users:
            prompt_parts.append("\nSPECJALNE ZACHOWANIA:")
            for user, settings in special_users.items():
                attitude = settings.get("attitude", "normal")
                description = settings.get("description", "")
                if description:
                    prompt_parts.append(f"- {user}: {description}")

        # User mappings
        user_mappings = config.get("user_mappings", {})
        if user_mappings:
            prompt_parts.append("\nNie używaj nicków, tylko imion:")
            for nick, name in user_mappings.items():
                prompt_parts.append(f"- {nick} - {name}")

        return "\n".join(prompt_parts)

    def get_trigger_words(self) -> List[str]:
        """Get list of trigger words that make bot respond"""
        return self._config.get("trigger_words", ["clanker"])

    def get_user_real_name(self, nickname: str) -> str:
        """Get real name for a nickname"""
        mappings = self._config.get("user_mappings", {})
        return mappings.get(nickname, nickname)

    def is_politically_incorrect_enabled(self) -> bool:
        """Check if politically incorrect mode is enabled"""
        pc_settings = self._config.get("politically_incorrect", {})
        return pc_settings.get("enabled", False)

    def reload_config(self):
        """Reload configuration from file"""
        self._load_config()
        logger.info("Prompt configuration reloaded")


# Global instance
prompt_manager = PromptManager()
