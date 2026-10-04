import ollama

response = ollama.chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "user",
            "content": "Write Python code to find the largest number in a list."
        }
    ]
)

print(response["message"]["content"])