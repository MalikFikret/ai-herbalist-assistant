from langchain_community.llms import Ollama

print("⏳ Connecting to Llama 3 local model...")

# Set up the model to connect to Ollama running on your Windows host machine.
# We use 'host.docker.internal' so the Docker container can reach outside to your laptop's localhost.
llm = Ollama(
    model="llama3",
    base_url="http://host.docker.internal:11434" 
)

# Ask a simple question to our AI Herbalist Assistant
question = "What are the top 3 benefits of drinking Chamomile tea? Keep it short."
print(f"❓ Question: {question}")

# Fetch the generated response from the local Llama 3 model
answer = llm.invoke(question)

print("\n✅ AI Response:")
print(answer)