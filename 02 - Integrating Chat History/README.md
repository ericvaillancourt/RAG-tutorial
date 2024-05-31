# Mastering LangChain RAG: Integrating Chat History (Part 2)

Welcome to the second part of our in-depth series on LangChain's Retrieval-Augmented Generation (RAG) technology. This repository contains the code examples and explanations for integrating chat history into a RAG-based application. By following this tutorial, you will learn how to maintain context in conversations and improve the interaction quality of your Q&A applications.

## Overview

In this tutorial, we focus on incorporating chat history into our RAG model to maintain context and improve the quality of interactions in chat-like conversations. Additionally, we cover how to save chat history to an SQL database using SQLAlchemy, ensuring robust and scalable storage.

## Introduction

In many Q&A applications, facilitating a dynamic, back-and-forth conversation between the user and the system is essential. This requires the application to maintain a "memory" of past interactions, allowing it to reference and integrate previous exchanges into its current processing.

## Getting Started

All code examples mentioned in this tutorial can be found in the `02 - Integrating Chat History` folder. To get started, clone this repository and navigate to the relevant folder:

```bash
git clone https://github.com/ericvaillancourt/RAG-tutorial.git
cd RAG-tutorial/02 - Integrating Chat History
```

## Environment Setup

Ensure your development environment is prepared with the necessary dependencies. You can install the required packages using pip:

```bash
pip install --upgrade --quiet langchain langchain-community langchainhub langchain-openai langchain-chroma bs4 python-dotenv sqlalchemy
```

You also need to set the `OPENAI_API_KEY` environment variable for the embeddings model. This can be done directly or loaded from a `.env` file. Create a `.env` file with the following content:

```
OPENAI_API_KEY=your-key-here
```

Load the environment variable in your code:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Code Explanation

### Updating the Prompt

We modify the application’s prompt to include historical messages as input. This change ensures that the system can access prior interactions and use them to understand and respond to new inquiries more effectively.

### Contextualizing Questions

We introduce a sub-chain that utilizes both historical messages and the latest user question. This sub-chain is designed to reformulate a question whenever it references past discussions.

### Setting Up the Database

To persist chat histories, we use SQLAlchemy to set up an SQLite database. We define models for sessions and messages, and create utility functions to save and load chat history.

### Creating the History-Aware Retriever

We create a history-aware retriever that integrates chat history for context-aware processing.

### Building the Q&A Chain

We build a comprehensive Q&A chain that handles inputs and produces outputs that include not just the query and its context, but also a well-integrated response, keeping track of the entire conversation history.

### Managing Chat History

We manage chat history using a dictionary structure and ensure that chat histories are saved and retrieved efficiently.

### Saving and Loading Messages

We define functions to save and load individual messages to and from the database, ensuring persistent storage.

## Usage

Invoke the chain and save the chat history. Here is an example of how to use the modified function to interact with the chain and persist the conversation:

```python
result = invoke_and_save("abc123", "What is Task Decomposition?")
print(result)

result = invoke_and_save("abc123", "What are common ways of doing it?")
print(result)
```

## Conclusion

In this tutorial, we explored how to enhance the functionality of Q&A applications by integrating historical interactions into the application logic and ensuring persistent storage with SQLAlchemy. By automating the management of chat history, we improve the application’s ability to engage users in a meaningful dialogue.


## Support

If you found this tutorial helpful, please consider supporting my work by buying me a coffee or two!

[![Buy Me a Coffee](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/evaillancourt)

Thank you for following along, and I look forward to continuing this journey with you in the next parts of our series.

---

Eric Vaillancourt

GitHub: [ericvaillancourt](https://github.com/ericvaillancourt)