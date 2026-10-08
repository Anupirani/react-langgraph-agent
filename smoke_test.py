from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
model = ChatGroq(model="openai/gpt-oss-20b")
print(model.invoke("Say hello in one sentence.").content)