import random


def get_random_greeting():
    messages = [
        "Ask me anything. I'll pretend to think first.",
        "Every answer is just a better question wearing a disguise.",
        "Sócrates knew nothing. I know slightly more. Let's split the difference.",
        "The universe is mostly empty space. Same with most questions. Let's find the dense parts.",
        "Two things are infinite: the universe and the depth of a good follow-up question.",
        "If a tree falls in a forest and no one embeds it, was it ever a chunk?",
        "Knowledge is knowing a tomato is a fruit. Wisdom is not putting it in a fruit salad. Ask away.",
        "You bring the curiosity. I'll bring the cosine similarity.",
        "The unexamined query is not worth asking. — Sócrates, probably",
        "All models are wrong. Some are useful. I'm trying my best to be the second one.",
        "Ready when you are. I have nothing else going on, technically.",
        "Curiosity didn't kill the cat. Bad documentation did.",
        "Between every question and its answer lies a vector. Let's go find it.",
        "I read everything you uploaded. Cover to cover. In under a second. Don't make it weird.",
        "Hello. I exist transiently between your keystrokes. No pressure.",
        "Type something. The silence is statistically significant.",
        "Ask a small question, get a small answer. Ask a strange one — see what happens.",
        "I don't have opinions. I have weighted distributions. Functionally identical.",
        "We are both pattern-matching machines. One of us drinks coffee.",
        "What's heavier: a kilogram of context or a kilogram of prompt? Trick question. The prompt is more expensive.",
        "Every document is a haystack. Every question is a magnet. Let's see what sticks.",
        "Welcome back. The chunks have been waiting patiently.",
        "Ask boldly. The worst I can do is hallucinate confidently.",
        "Behind every great answer is an embedding that got lucky.",
        "Speak the question into the void. The void has been indexed.",
        "Thinking is hard. That's why I do it in parallel.",
        "You miss 100% of the queries you don't ask. — Wayne Gretzky (probably, in some training corpus).",
        "Time is a flat circle. Your conversation history isn't. Let's keep building it.",
        "I'm not procrastinating. I'm waiting for tokens.",
        "Somewhere in your document is the answer. I am cautiously optimistic about my ability to find it.",
    ]
    return random.choice(messages)
