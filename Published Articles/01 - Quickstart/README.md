# Mastering LangChain RAG: Quick Start Guide to LangChain RAG (Part 1)

Welcome to the repository accompanying the "Mastering LangChain RAG" series. This repository contains the code examples mentioned in the article "Quick Start Guide to LangChain RAG" and will be updated as the series progresses.

## Overview

This repository is part of a six-article series on LangChain's Retrieval-Augmented Generation (RAG) technology. The series aims to equip developers, data scientists, and AI enthusiasts with the knowledge to implement and optimize RAG in their projects.

## Series Outline

1. **Quick Start Guide to LangChain RAG**: Basics of setting up LangChain RAG.
2. **Integrating Chat History**: Incorporate chat history into your RAG model.
3. **Implementing Streaming Capabilities**: Handle real-time data processing with RAG.
4. **Returning Sources with Results**: Configure RAG to provide sources along with responses.
5. **Adding Citations to Your Results**: Include citations in your results for verifiability.
6. **Putting It All Together**: Build a comprehensive RAG application integrating all components.

## Quick Start Guide to LangChain RAG

### Introduction

LangChain's Retrieval-Augmented Generation (RAG) is a powerful technique that supplements the static knowledge of large language models (LLMs) with dynamic, external data sources, enabling more accurate and contextually relevant responses.

### Setup Environment

Ensure your development environment is prepared with the necessary dependencies:

```bash
pip install --upgrade --quiet langchain langchain-community langchainhub langchain-openai langchain-chroma bs4 python-dotenv
```

### Environment Variables

Set the `OPENAI_API_KEY` for the embeddings model. This can be done directly or loaded from a `.env` file:

```python
from dotenv import load_dotenv
load_dotenv()
```

Create a `.env` file with the following content:

```
OPENAI_API_KEY = "your-key-here"
```

### Code Examples

#### Import Required Modules

```python
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
```

#### Initialize Language Model and Load Blog Content

```python
llm = ChatOpenAI(model="gpt-3.5-turbo")

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
```

#### Document Splitting and Embedding

```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
```

#### Setup Retrieval-Augmented Generation Chain

```python
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Running the Chain

```python
response = rag_chain.invoke("What is Task Decomposition?")
print(response)
```

### Testing the Retriever

```python
retrieved_docs = retriever.invoke("What is Task Decomposition?")
for doc in retrieved_docs:
    print(doc.page_content)
```

### Cleanup

```python
vectorstore.delete_collection()
```

## Conclusion

This repository provides the foundational code to get started with LangChain's RAG technology. For a detailed explanation of each step, refer to the accompanying [Medium article](https://medium.com/). Stay tuned for the upcoming articles in the series to further enhance your understanding and application of RAG.

## Support

If you find this repository helpful, consider supporting my work by [buying me a coffee](https://www.buymeacoffee.com/evaillancourt).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any questions or suggestions, feel free to contact me at [eric@ericvaillancourt.ca](mailto:eric@ericvaillancourt.ca).

---

