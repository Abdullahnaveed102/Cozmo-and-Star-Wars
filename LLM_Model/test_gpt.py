import openai

openai.api_key = 'sk-'
messages = [{"role": "system", "content": "You are a kind helpful assistant."}]

while True:
    message = input("User : ")
    if message:
        messages.append(
            {"role": "user", "content": message},
        )
        # Set the stop parameter to limit the response to one sentence
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages,
        )
    
    reply = chat.choices[0].message.content
    print(f"ChatGPT: {reply}")
    messages.append({"role": "assistant", "content": reply})










































#==========================================================================================> Another Method
# import requests
# URL = "https://api.openai.com/v1/chat/completions"

# payload = {
# "model": "gpt-3.5-turbo",
# "messages": [{"role": "user", "content": f"What is the first computer in the world?"}],
# "temperature" : 1.0,
# "top_p":1.0,
# "n" : 1,
# "stream": False,
# "presence_penalty":0,
# "frequency_penalty":0,
# }

# headers = {
# "Content-Type": "application/json",
# "Authorization": f"Bearer {openai.api_key}"
# }

# response = requests.post(URL, headers=headers, json=payload, stream=False)
# print(response.content)
#==========================================================================================>