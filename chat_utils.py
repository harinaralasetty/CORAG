import gradio as gr 

def format_exchange(message):
    return f"User: {message['user']} \nGemini: {message['answer']}\n"

def get_chat_history(chat_history):
    return [
        format_exchange(message)
        for message in chat_history['messages']
    ]

def get_clean_chat_history(chat_history):
    return "\n".join(
        get_chat_history(chat_history)
    )

def export_chat_history(chat_history):
    # Save chat history to a text file
    with open("chat_history.txt", "w+") as f:
        f.write("Chat History:\n")
        for message in chat_history['messages']:
            f.write(f"{message['user']}: {message['answer']}\n")
    print("Chat history exported to chat_history.txt")
