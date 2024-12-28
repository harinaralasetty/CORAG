
def format_exchange(message):
    return f"User: {message['user']} \nAssistant: {message['answer']}\n"

def get_chat_history(chat_history):
    return [
        format_exchange(message)
        for message in chat_history['messages']
    ]

def get_clean_chat_history(chat_history):
    return "\n".join(
        get_chat_history(chat_history)
    )

# def export_chat_history(chat_text):
#     # Save chat history to a text file
#     with open("chat_history.txt", "w+") as f:
#         f.write(f"Chat History:\n{chat_text}")
#     print("Chat history exported to chat_history.txt")
