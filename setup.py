from setuptools import setup, find_packages

setup(
    name="clanker",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "python-telegram-bot",
        "openai",
        "loguru",
        "python-dotenv",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "clanker=clanker.bot:main",
        ],
    },
)