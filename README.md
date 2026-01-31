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
| **Event Recommendation Agent**                     | Combine multiple data sources to generate personalized recommendations        | Built events database, weather agent, and GPT-based recommendation logic; coordinated agents to generate context-aware suggestions                                         | Practical multi-agent system demonstrating reasoning across heterogeneous data sources                | `multi_agent_recommendation_system.ipynb`                                 |
| **Transformer-Based Sentiment Classification**     | Apply Transformer architectures to real-world NLP classification tasks        | Implemented and compared basic vs advanced Transformers with custom positional encoding; trained on IMDB dataset; evaluated using Accuracy, Precision, Recall, F1, AUC-ROC | Strong hands-on understanding of Transformer internals and evaluation for customer sentiment analysis | `transformer_based_sentiment_classification.ipynb`                                 |


## Tech Stack

   * Languages: Python
   * Frameworks: LangChain, custom orchestration
   * LLM APIs: OpenAI compatible models, open-source LLMs from HuggingFace hub
   * Vector Stores: FAISS
   * Libraries: openai, transformers, Pandas, NumPy, sqlite3,  etc.
   * Experimentation: Jupyter Notebooks, Google Colab
   
