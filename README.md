📊 Reddit Relevance Analyzer
This Streamlit application combines social media data scraping (Reddit via PRAW), Natural Language Processing (NLTK) for keyword extraction, and advanced Language Models (Azure Mistral via LangChain) to quickly search, retrieve, and score Reddit posts based on their relevance to a specific research question.

The core goal is to filter noise and identify highly relevant, actionable, or method-specific posts on a large scale.

✨ Features
Targeted Reddit Search: Searches specific subreddits using keywords extracted from a user-defined research question.

Intelligent Keyword Extraction: Uses NLTK's Part-of-Speech tagging to identify strong keywords (Nouns, Adjective-Noun pairs) from the query for more precise searching.

LLM-Powered Relevance Scoring: Submits batches of posts (title and body/link) to a Mistral AI model (hosted on Azure) for relevance scoring (1-5 scale) against the research question.

Asynchronous Processing: Uses asyncio and semaphores to concurrently manage API calls to the LLM, dramatically speeding up the analysis of multiple posts.

Robust Retries: Implements exponential backoff and retry logic using tenacity to handle transient API issues and improve stability.

Real-time Results: Displays posts ranked by their calculated relevance score in the Streamlit UI.

⚙️ Setup and Installation
Prerequisites
Python 3.8+

Reddit API Credentials: You must have a Reddit developer account to obtain a Client ID, Client Secret, and your Reddit username/password.

Azure Mistral Credentials: Access to an Azure OpenAI or Mistral Endpoint URL and API Key for running the LLM evaluations.

Installation
Clone the repository (or save the main.py file):

git clone [Your Repository URL]
cd reddit-relevance-analyzer

Create and activate a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install the required dependencies:

pip install streamlit praw nltk tenacity langchain-mistralai requests

Configuration (API Keys)
This application requires sensitive credentials to be entered directly into the Streamlit interface.

You will need to input the following when you run the app:

Credential

Used For

Source

Reddit Client ID

Connecting PRAW to Reddit

Reddit Developer Console

Reddit Client Secret

Connecting PRAW to Reddit

Reddit Developer Console

Reddit Username

Authentication

Your Reddit Account

Reddit Password

Authentication

Your Reddit Account

Azure Endpoint URL

Initializing ChatMistralAI

Azure AI Studio/Portal

LLM API Key

Authorizing LLM requests

Azure AI Studio/Portal

🚀 How to Run
Execute the Streamlit application from your terminal:

streamlit run main.py

The application will open in your default web browser (usually at http://localhost:8501).

🖥️ Usage
Input Credentials: Enter your Reddit and LLM API credentials in sections 1 and 2.

Set Parameters:

Research Question: Enter the detailed query you want posts scored against (e.g., "What are the most effective psychological methods for managing chronic pain in young adults?").

Subreddit: Enter the subreddit name to search (e.g., ChronicPain).

Limit: Specify the number of posts to retrieve (e.g., 20).

Run Analysis: Click the "Search and Analyze Reddit Posts" button.

View Results: The application will display the retrieved posts, sorted in descending order by the Relevance Score assigned by the LLM.
