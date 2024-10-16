import gradio as gr 
import shutil

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

def export_chat_history(chat_text):
    # Save chat history to a text file
    with open("chat_history.txt", "w+") as f:
        f.write(f"Chat History:\n{chat_text}")
    print("Chat history exported to chat_history.txt")

# def export_chat_history(chat_history):
#     print('RECEIEVED 18', chat_history)
#     # Use a temporary file for writing
#     temp_file_path = "temp_chat_history.txt"
#     with open(temp_file_path, "w+") as f:
#         f.write("Chat History:\n")
#         for message in chat_history['messages']:
#             f.write(f"{message['user']}: {message['answer']}\n")
#     print("Chat history exported to temporary file.")

#     # Rename the temporary file to the desired filename
#     shutil.move(temp_file_path, "chat_history.txt")
#     print("Temporary file renamed to chat_history.txt.")
