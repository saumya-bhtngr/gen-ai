# GenAI
This repository contains projects for Generative AI tasks.

## Repository Structure
```
├── agentic-ai
│   ├── multi_agent_recommendation_system.ipynb
│   ├── sql_agent.ipynb
│   └── weather_information_agent.ipynb
├── ai_assistant.ipynb
├── langchain_agent_tool_calling_prototype.ipynb
├── langchain_ollama.ipynb
├── rag
│   ├── data
│   │   └── company_policy.pdf
│   └── rag_pipeline.ipynb
├── README.md
├── transformers
│   └── transformer_based_sentiment_classification.ipynb
└── vision-models
    ├── vae_mnist.ipynb
    └── vit_image_classification.ipynb
```
## Projects Summary

| Project                                            | Problem                                                                       | Solution                                                                                                                                                                   | Outcome / Impact                                                                                      | Artifacts                         |
| -------------------------------------------------- | ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------- |
| **Retrieval-Augmented Generation (RAG) QA System** | LLMs lack grounding and struggle with context continuity across conversations | Built an end-to-end RAG pipeline with document chunking, embeddings, FAISS-based semantic search, and GPT-4 integration with conversational memory                         | Enterprise-ready, context-aware QA system that improves factual grounding and multi-turn coherence    | `rag_pipeline.ipynb`              |
| **Weather Information Agent**                      | Demonstrate practical agent behavior using function calling                   | Integrated external Weather API via OpenAI function calling; agent retrieves, processes, and formats real-time weather data                                                | Demonstrates task planning, tool use, and autonomous execution via GenAI agents                       | `weather_information_agent.ipynb` |
| **SQL Agent**                                      | Enable safe interaction with databases using natural language queries         | Converted NL queries to validated SQL using OpenAI API; executed queries securely with safeguards against destructive operations                                           | Production-oriented agent for querying databases safely and effectively                               | `sql_agent.ipynb`                 |
| **Event Recommendation Agent**                     | Combine multiple data sources to generate personalized recommendations        | Built events database, weather agent, and GPT-based recommendation logic; coordinated agents to generate context-aware suggestions                                         | Practical multi-agent system demonstrating reasoning across heterogeneous data sources                | —                                 |
| **Transformer-Based Sentiment Classification**     | Apply Transformer architectures to real-world NLP classification tasks        | Implemented and compared basic vs advanced Transformers with custom positional encoding; trained on IMDB dataset; evaluated using Accuracy, Precision, Recall, F1, AUC-ROC | Strong hands-on understanding of Transformer internals and evaluation for customer sentiment analysis | —                                 |


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

   **2.3 Event Recommendation Agent**
   Problem: Build a recommendation system that combines weather data and local events to suggest activities, demonstrating practical multi-agent coordination.

   Solution:
   * Create events database and insert sample erecords using Python
   * Create a weather agent to fetch the current weather using Weather API 
   * Implement recommendation system that generates clear, context-aware recommendations using GPT
   * incorporate weather data and available events data for recommendation.
   
**3. Transformer-Based Sentiment Classification**

   Problem: Develop and evaluate basic and advanced Transformer models for binary text classification to understand the application of attention mechanisms and positional encoding in NLP.
   Dataset Used: IMDB dataset
   Solution:
   * Implement, train, and evaluate two neural network architectures (basic and advanced Transformer models)
   * Loading, pre-processing the IMDB dataset along with EDA
   * Constructing and training basic and advanced transformer models (with custom positional encoding layer)
   * Display Model architecture and visualize training progress
   * Evaluate both models using Accuracy, Precision, Recall, F1-score and AUC-ROC
   * Compare their performance on sentiment classification using appropriate metrics.

   Outcome:
   Hands-on experience with modern Transformer architectures applied to customer sentiment analysis using the IMDB dataset.

   **Tech Stack**

   * Languages: Python
   * Frameworks: LangChain, custom orchestration
   * LLM APIs: OpenAI compatible models, open-source LLMs from HuggingFace hub
   * Vector Stores: FAISS
   * Libraries: openai, transformers, Pandas, NumPy, sqlite3,  etc.
   * Experimentation: Jupyter Notebooks, Google Colab
   
