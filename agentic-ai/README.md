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
