# GenAI
This repository contains projects for Generative AI tasks.

## Projects

**1. Retrieval Augmented Generation (RAG) based Question Answering System**

   Problem: Design and implement a Retrieval-Augmented Generation (RAG) pipeline with conversational memory using generative AI tools.

   Solution:
   * Built a RAG pipeline that combines generative capabilities with real-time access to domain-specific documents
   * Implemented chunking and created embeddings for each chunk
   * Used vector store FAISS and executed semantic search
   * Integrated vector store wth GPT-4 language model and added memory capability

   Outcome:
   Enterprose-ready AI system using conversational memory to enhance context-awareness and continuity. 
   Filename: rag_pipeline.ipynb

**2. Agentic AI**

   **2.1 Weather Information Agent**

   Problem: Build a function-calling agent that retrieves and processes weather information, demonstrating the practical implementation of agent capabilities. 

   Solution:
   * Interated external API
   * Agent correctly returns a formatted weather summary (e.g., location, temperature, condition).
   * Function calling via OpenAI API is successfully set up and tested with sample queries.

   Outcome:
   Leverage GenAI agents for task planning, execution and autonomous problem-solving
   Filename: weather_information_agent.ipynb

   **2.2 SQL Agent**
    Problem: Build an agent that converts natural language queries into SQL, executes them, and returns formatted results. 

   Solution:
   * Successfully use OpenAI API to convert natural language queries into valid, syntactically correct SQL.
   * Execute generated SQL safely and returns correct, formatted results from the database.
   * Implement proper validation (e.g., blocks dangerous statements like DROP, DELETE, etc.).

   Outcome:
   Use agents to interact with database systems and execute user queries
   Filename: sql_agent.ipynb
   
