import datalabs_chatbot as chatbot

while True:
    message = input("")
    print(chatbot.get_chat_response(message))