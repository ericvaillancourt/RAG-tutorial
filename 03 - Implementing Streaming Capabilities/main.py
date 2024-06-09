import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessageChunk
#import logging

from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Load, chunk and index the contents of the blog.
bs_strainer = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs_strainer},
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
# we need to add the streaming=True
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

app = FastAPI()

# Allow CORS for all origins (for testing purposes; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return FileResponse("static/index.html")

def serialize_aimessagechunk(chunk):
    """
    Custom serializer for AIMessageChunk objects.
    Convert the AIMessageChunk object to a serializable format.
    """
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialization"
        )

async def generate_chat_events(message):
    try:
        
        async for event in rag_chain_with_source.astream_events(message, version="v1"):
            
            if event["event"] == "on_chat_model_stream":
                chunk_content = serialize_aimessagechunk(event["data"]["chunk"])
                if len(chunk_content) != 0:
                    data_dict = {"data": chunk_content}
                    data_json = json.dumps(data_dict)
                    yield f"data: {data_json}\n\n"
            if event["event"] == "on_retriever_end":
                documents = event['data']['output']['documents']
                # Créer une nouvelle liste qui va contenir les documents formatés
                formatted_documents = []
                # Parcourir chaque document dans la liste originale
                for doc in documents:
                    
                    # Créer un nouveau dictionnaire pour chaque document avec le format requis
                    formatted_doc = {
                        'page_content': doc.page_content,
                        'metadata': {
                            'source': doc.metadata['source'],
                        },
                        'type': 'Document'
                    }
                    # Ajouter le document formaté à la liste finale
                    formatted_documents.append(formatted_doc)

                # Créer le dictionnaire final avec la clé "context"
                final_output = {'context': formatted_documents}

                # Convertir le dictionnaire en chaîne JSON
                data_json = json.dumps(final_output)
                yield f"data: {data_json}\n\n"
            if event["event"] == "on_chat_model_end":
                print("Chat model has completed its response.")

    except Exception as e:
        print('error'+ str(e))

@app.get("/chat_stream/{message}")
async def chat_stream_events(message: str):
    return StreamingResponse(generate_chat_events(message), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
