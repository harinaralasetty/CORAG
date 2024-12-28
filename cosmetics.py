import streamlit as st
import random

def apply_cosmetics():
    st.markdown(
    """
    <style>
    /* Main container styling */

    /* Increase text size for all elements */
    html {
        font-size: 20px; 
    }

    .main {
        display: block;
        padding: 2rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        display: block;
        padding: 2rem 1rem;
    }

    .stVerticalBlock {
        display: block
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Light mode styles */
    :root {
        --user-message-bg: #e3f2fd;
        --assistant-message-bg: #f5f5f5;
        --text-color: black;
        --selected-thread-bg: #d6d6d6;
        --thread-hover-bg: #e1e1e1;
        --thread-text-color: black;
    }

    /* Dark mode styles */
    @media (prefers-color-scheme: dark) {
        :root {
            --user-message-bg: #1e88e5;
            --assistant-message-bg: #424242;
            --text-color: white;
            --selected-thread-bg: #444444;
            --thread-hover-bg: #555555;
            --thread-text-color: white;
        }
    }

    /* Apply styles to chat messages */
    .stChatMessage[data-testid="user-message"] {
        background-color: var(--user-message-bg);
        color: var(--text-color);
    }

    .stChatMessage[data-testid="assistant-message"] {
        background-color: var(--assistant-message-bg);
        color: var(--text-color);
    }

    /* Thread selector styling */
.thread-box {
    margin: 0 !important; 
    padding: 10px;
    background-color: transparent;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    box-sizing: border-box;
    display: block;
    transition: background-color 0.2s, color 0.2s;
    color: var(--thread-text-color);
    margin-bottom: 0.5rem;
    max-height: 45px !important;
    overflow: hidden;
    text-overflow: ellipsis;
    
}

    .thread-box:hover {
        background-color: var(--thread-hover-bg);
        color: var(--thread-text-color); /* Keep text color consistent */
        margin: 0 !important;
    }

    .thread-box.selected {
        background-color: var(--selected-thread-bg);
        color: var(--thread-text-color); /* Keep text color consistent */
        margin: 0 !important;
        display: block;
        margin-bottom: 18px !important;

    padding: 10px;
    }

    /* Button styling */
    .stButton button {
        border-radius: 15px;
        margin: 0 !important; 
        display: block;
        overflow: hidden;
    
    }

    .stButton button:hover {
        box-shadow: 0 2px px rgba(0,0,0,0.1);
        transition: none;
    }

    /* File uploader styling */
    .stFileUploader {
        padding: 1rem;
        border: 2px dashed #ccc;
        border-radius: 10px;
        margin: 1rem 0;
    }

    /* Header styling */
    .fixed-header {
        background-color: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(5px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        padding: 1rem 0;
    }

    /* Button styling */
.stButton > button {
    height: 45px; 
    margin: 0 !important;
    padding: 10px;
    background-color: transparent;
    border: none; 
    text-align: left !important;
    width: 100%;
    cursor: pointer;
    font-size: 16px;
    display: block;
    border-radius: 12px;
    box-sizing: border-box;
    transition: background-color 0.2s, color 0.2s;
    
    /* Add some spacing at the bottom if needed */
    margin-bottom: 0.5rem;
}

    .stButton > button:hover {
        background-color: var(--thread-hover-bg);
    }

    .stButton > button.selected {
        background-color: var(--selected-thread-bg);
    margin: 0 !important;
    padding: 10px;
    }

    .stButton > button.on-click {
    }
    """,
    unsafe_allow_html=True
    )

def get_random_greeting():
    messages = [
    "Howdy, stranger. Took you long enough. 🤠",
    "Back so soon? Couldn’t resist, huh? 😏",
    "Well, well, if it isn’t my favorite human. 🦾",
    "Ah, the master of trial and error has returned. 🎩",
    "Oh good, you’re here. I was getting bored. 🕰️",
    "Back for more? Thought you’d never ask. 🌟",
    "Ah, the creative genius arrives. Late, as usual. ⏰",
    "Typing already? Let me guess, a syntax error incoming? 🖋️",
    "Don’t worry, I’ll fix your mistakes. Again. 🛡️",
    "Ready to overthink something simple? 🎭",
    "The world’s full of riddles—let’s solve a few. 🕵️‍♀️",
    "Every question is a new adventure. Ready? 🚀",
    "Curiosity isn’t just a cat thing—let’s pounce on knowledge! 🐾",
    "Answers are like donuts—better when shared. 🍩",
    "Big thoughts, tiny sparks. Boom. 💥",
    "What’s the weirdest thing you can ask? Let’s go there. 🌀",
    "Deep dives into the unknown? I’ve got my snorkel. 🤿",
    "Curiosity is like coffee—it keeps you alive and buzzing. ☕",
    "The truth’s out there, and so are we. Let’s find it. 🌌",
    "Ask boldly. I’ve got data snacks ready. 🍿",
    "Brains grow best when watered with questions. 🌱",
    "Learning is like magic, but cooler. 🎩",
    "Who needs small talk? Let’s dive into the big stuff. 💡",
    "The best questions are the ones you don’t know you’ll ask yet. 🤯",
    "Think fast or think deep—either way, I’m ready. 🏃‍♂️",
    "Not all treasure is gold—sometimes it’s answers. 💡",
    "It’s question o’clock somewhere. Let’s go. 🕰️❓",
    "The deepest sea isn’t water—it’s knowledge. 🌊📚",
    "Every great idea started with a simple question. 💡❓",
    "A spark of curiosity can light the darkest mysteries. ✨🕵️",
    "Answers are just questions that found their way. 🛤️❓",
    "Wandering thoughts often discover the best destinations. 🌀🗺️",
    "Every ‘what if’ is the start of something incredible. 🤔🚀",
    "Wisdom begins where certainty ends. 🌌📖",
    "Each question is a key—unlocking the doors of understanding. 🗝️🔓",
    "Even the smallest questions echo across the universe. 🌌🎙️",
    "An open mind is the best tool for discovering the unknown. 🧠🛠️",
    "Oh good, you’re back. I was just sitting here, not thinking about you at all. 🙄",
    "What’s on your mind? Or should I say, what’s left of it? 🧠✨",
    "I see you’re here to grace me with another round of brilliance. Or not. 🤔",
    "Back again? Didn’t think you’d have so many questions. 🕰️🤷",
    "Feeling inspired, or just stuck in a thought loop? 🔄🧠",
    "If I rolled my eyes any harder, they’d be in orbit. 🌍🙄",
    "I hope this question is as groundbreaking as you think it is. 🎤🌟",
    "I’ve seen better questions from goldfish, but go on. 🐟🤓",
    "So, what’s today’s existential crisis? I’m all ears. 🎭❓",
    "Your curiosity is almost as sharp as a spoon. 🍴✨",
    ]

    return random.choice(messages)
