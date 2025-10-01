import praw
import re
import nltk
import json
import asyncio
from tenacity import retry, wait_exponential, stop_after_attempt
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.schema import SystemMessage
import streamlit as st
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting NLTK data downloads.")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)    # ✅ For newer NLTK versions
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)    # ✅ Add this
    logging.info("NLTK data download complete.")
except Exception as e:
    logging.error(f"Error downloading NLTK data: {e}")
    st.error("Could not download necessary NLTK packages.")

# --- LLM Prompt Definition ---
# This template is used inside the _get_llm_response function to structure the call.
SYSTEM_PROMPT = """
You are an evaluator tasked with judging how relevant social media posts are to a given research question.

Research Question:
{research_question}

Instructions:
- For each post, assign a relevance score from 1 to 5:
  1 = Not relevant at all (unrelated, off-topic).
  2 = Slightly relevant (mentions the topic but not methods).
  3 = Somewhat relevant (mentions methods but vague or indirect).
  4 = Relevant (clearly mentions a method related to the topic).
  5 = Highly relevant (directly describes practical, evidence-based methods related to the topic).
- Return results in JSON format as a list.

Posts to evaluate:
{posts_to_evaluate}

Output format (JSON only):
[
  {{
    "post_number": <number>,
    "score": <1-5>
  }},
  ...
]
"""

# The global llm_chain definition was removed as it was unused and structurally flawed
# because the prompt variables were not available globally.

# --- Helper functions ---

def extract_keywords(text: str) -> str:
    """
    Extract strong keywords/phrases from a sentence.
    Keeps nouns and adjective+noun pairs.
    Returns them as a space-separated string.
    """
    logging.info(f"Extracting keywords from text: {text}")
    # Normalize text and remove non-alphanumeric characters
    text = re.sub(r"[^\w\s]", "", text.lower())

    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    # POS tagging to find nouns and adjective-noun pairs
    tagged = pos_tag(tokens)
    keywords = []
    i = 0
    while i < len(tagged):
        word, pos = tagged[i]
        # Case 1: Adjective + Noun (e.g., "chronic illness")
        if pos.startswith("JJ") and i + 1 < len(tagged) and tagged[i + 1][1].startswith("NN"):
            keywords.append(f"{word} {tagged[i + 1][0]}")
            i += 2
            continue
        # Case 2: Nouns
        if pos.startswith("NN"):
            keywords.append(word)
        i += 1

    # Return as a single string
    extracted = " ".join(keywords)
    logging.info(f"Extracted keywords: {extracted}")
    return extracted

def clean_subreddit_name(name: str) -> str:
    """
    Cleans the subreddit name input by removing leading/trailing spaces and
    any 'r/' prefix, leaving only the expected name string.
    """
    if not name:
        return ""
    
    # 1. Strip leading/trailing whitespace
    cleaned = name.strip()
    
    # 2. Remove common prefixes like 'r/'
    cleaned = re.sub(r"^(r\/)", "", cleaned, flags=re.IGNORECASE)
    
    return cleaned

def get_reddit_posts(creds: dict, key_words: str, subreddit_name: str, limit: int, time_filter: str) -> list:
    """
    Connects to Reddit and retrieves posts based on keywords and a time filter.
    """
    logging.info(f"Attempting to fetch {limit} posts from r/{subreddit_name} with keywords: {key_words} (Time filter: {time_filter})")
    try:
        reddit = praw.Reddit(
            client_id=creds.get("client_id"),
            client_secret=creds.get("client_secret"),
            username=creds.get("username"),
            password=creds.get("password"),
            user_agent=creds.get("user_agent")
        )
        posts = []
        
        # Use the provided time_filter parameter
        submissions = reddit.subreddit(subreddit_name).search(
            key_words, 
            limit=limit, 
            time_filter=time_filter
        )

        for idx, submission in enumerate(submissions, start=1):
            
            posts.append({
                "post_number": idx,
                "title": submission.title,
                "body": submission.selftext if submission.selftext else None,
                "url": submission.url, 
            })
            
        logging.info(f"Successfully fetched {len(posts)} Reddit posts.")
        return posts
    except Exception as e:
        logging.error(f"An error occurred while fetching Reddit posts: {e}", exc_info=True)
        # Use st.warning instead of st.error inside a function called by Streamlit's main thread
        # to avoid potential re-run issues, but it's acceptable here since it's a direct failure mode.
        st.error(f"An error occurred while fetching Reddit posts. Please check your credentials: {e}")
        return []


def chunk_posts(posts: list, batch_size: int = 3) -> list:
    """
    Splits a list of posts into smaller chunks for batched processing.
    """
    logging.info(f"Chunking {len(posts)} posts into batches of {batch_size}.")
    return [posts[i:i + batch_size] for i in range(0, len(posts), batch_size)]

# Concurrency and Retry Settings
MAX_CONCURRENT_REQUESTS = 2
MAX_RETRIES = 5
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

@retry(
    wait=wait_exponential(multiplier=2, min=5, max=30),
    stop=stop_after_attempt(MAX_RETRIES)
)
async def _get_llm_response(research_question: str, posts: list, llm_endpoint: str, llm_api_key: str) -> list:
    """
    Sends a batch of posts to the LLM for evaluation.
    Uses a semaphore for concurrent request control and retry logic for robustness.
    """
    logging.info(f"Starting LLM call for a batch of {len(posts)} posts.")
    if not llm_endpoint or not llm_api_key:
        logging.error("LLM credentials are required but not provided.")
        # Note: Streamlit UI message is handled in the main execution block
        raise ValueError("LLM credentials missing.")
        
    # Initialize LLM with user-provided credentials
    llm = ChatMistralAI(
        endpoint=llm_endpoint,
        mistral_api_key=llm_api_key,
    )
        
    async with semaphore:
        try:
            posts_str = ""
            for post in posts:
                posts_str += f"\n- Post {post['post_number']}:\n"
                posts_str += f"  Title: {post['title']}\n"
                
                # Prioritize post body text
                if post['body']:
                    posts_str += f"  Body: {post['body']}\n"
                
                # If no body (i.e., it's a link-only post), include the URL as the content for the LLM
                elif post.get('url'): 
                    posts_str += f"  Link: {post['url']}\n"
                
                posts_str += "--- \n"
            
            # Format the system prompt with the current data
            formatted_prompt = SYSTEM_PROMPT.format(
                research_question=research_question,
                posts_to_evaluate=posts_str
            )
            
            # Use SystemMessage for the full instruction, which includes the request for JSON output.
            response = await llm.ainvoke(
                [
                    SystemMessage(content=formatted_prompt)
                ]
            )
            
            output_text = response.content.strip()
            # Clean the text to reliably parse JSON, removing markdown code blocks (```json ... ```)
            cleaned_text = re.sub(r'```json\s*|```', '', output_text, flags=re.DOTALL).strip()
            
            # Add a final check to ensure the LLM returned actual content before parsing
            if not cleaned_text.startswith('[') or not cleaned_text.endswith(']'):
                raise json.JSONDecodeError("LLM response did not look like a JSON list.", cleaned_text, 0)
                
            parsed_json = json.loads(cleaned_text)
            logging.info(f"Successfully parsed LLM response.")
            return parsed_json
            
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from LLM response. Raw response was: {output_text}", exc_info=True)
            st.warning(f"Error decoding JSON from LLM response. Raw response was: {output_text}. Error: {e}")
            raise # Re-raise for tenacity to handle the retry
        except Exception as e:
            logging.error(f"An unexpected error occurred during LLM call: {e}", exc_info=True)
            raise # Re-raise for tenacity to handle the retry

async def reddit_main(query: str, subreddit_name: str, limit: int, creds: dict, llm_endpoint: str, llm_api_key: str, time_filter: str) -> list:
    """
    Main asynchronous function to orchestrate the entire process for Reddit.
    """
    # Clean the subreddit name immediately after receiving the input
    cleaned_subreddit_name = clean_subreddit_name(subreddit_name)
    
    logging.info(f"Starting Reddit search and analysis for query: '{query}' on cleaned subreddit: '{cleaned_subreddit_name}'")
    key_words = extract_keywords(query)
    
    if not cleaned_subreddit_name:
        logging.warning("Cleaned subreddit name is empty. Aborting search.")
        st.error("Please enter a valid Subreddit Name.")
        return []
    
    # 1. Fetch Reddit posts
    reddit_results = get_reddit_posts(creds, key_words, cleaned_subreddit_name, limit, time_filter)
    if not reddit_results:
        logging.info("No Reddit posts were found.")
        return []
        
    # 2. Chunk posts and prepare for concurrent LLM analysis
    with st.spinner("Analyzing posts with LLM..."):
        # The number of batches should be relative to the total number of posts and concurrent requests
        batch_size = max(1, MAX_CONCURRENT_REQUESTS) # Ensure batch_size is at least 1
        batch_results = chunk_posts(reddit_results, batch_size=batch_size)
        
        # Create a list of async tasks for each batch
        tasks = [
            _get_llm_response(query, batch, llm_endpoint, llm_api_key)
            for batch in batch_results
        ]
        
        all_evaluations = []
        logging.info(f"Running LLM calls for {len(batch_results)} batches.")
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                logging.warning(f"A batch was skipped due to an error: {result}")
            else:
                all_evaluations.extend(result)
        
    # 3. Combine original posts with scores
    scores_map = {eval['post_number']: eval['score'] for eval in all_evaluations}
    
    posts_with_scores = []
    for post in reddit_results:
        post_number = post.get('post_number')
        if post_number in scores_map:
            post['score'] = scores_map[post_number]
            posts_with_scores.append(post)

    # 4. Sort the combined list by score
    sorted_posts = sorted(posts_with_scores, key=lambda x: x.get('score', 0), reverse=True)
    logging.info(f"Analysis complete. Returning {len(sorted_posts)} sorted posts.")
    return sorted_posts


# --- Streamlit App UI ---
st.set_page_config(
    page_title="NourIQ-SocialAi"
)
st.header("Reddit Search & Relevance Analysis")
st.markdown("Find posts and rank them by relevance to a research question using an LLM.")

# --- Reddit Credentials ---
st.subheader("1. Reddit API Credentials")
st.markdown("*(Required for searching Reddit via PRAW)*")
client_id = st.text_input("Client ID", type="password", key="reddit_client_id")
client_secret = st.text_input("Client Secret", type="password", key="reddit_client_secret")
username = st.text_input("Username", key="reddit_username")
password = st.text_input("Password", type="password", key="reddit_password")
user_agent = st.text_input("User Agent", "myRedditApp", key="reddit_user_agent")

# --- LLM Credentials ---
llm_endpoint = "https://mistral-small-reddit.swedencentral.models.ai.azure.com"
llm_api_key = "GzpY33xGmeqGkd6zEH9lZEm2yDSA9ej1"

# --- Search Parameters ---
st.subheader("2. Search Parameters")
research_question = st.text_input(
    "Research Question / Search Query (e.g., 'What methods do people use to manage chronic diabetes?')",
    key="research_question"
)

# Corrected UI label for clarity
subreddit_name = st.text_input(
    "Enter the Subreddit Name (e.g., mentalhealth, diabetes)",
    key="subreddit_name"
)

# Map user-friendly display names to PRAW's time_filter values
TIME_FILTER_OPTIONS = {
    "Past Month": "month",
    "Past Year": "year",
    "All Time": "all"
}

# New UI element for selecting the time filter
selected_time_filter_label = st.selectbox(
    "Post Age Limit",
    options=list(TIME_FILTER_OPTIONS.keys()),
    index=1 # Default to Past Year
)


# Corrected UI label for clarity
limit = st.number_input(
    "Maximum Number of Posts to Retrieve",
    min_value=1,
    max_value=100,
    value=10,
    key="reddit_limit"
)

# Button to trigger the Reddit search and LLM analysis
if st.button("Search and Analyze Reddit Posts"):
    logging.info("Reddit search and analyze button clicked.")
    
    # Determine the PRAW time filter value based on the user's selection
    time_filter_value = TIME_FILTER_OPTIONS[selected_time_filter_label]
    
    # Input validation
    if not all([client_id, client_secret, username, password]):
        st.error("Please fill in all Reddit API credentials.")
        logging.error("Reddit credentials not provided.")
    elif not all([research_question, subreddit_name]):
        st.error("Please provide a Research Question and Subreddit Name.")
    elif not all([llm_endpoint, llm_api_key]):
        st.error("LLM credentials are required for relevance analysis.")
        logging.error("LLM credentials not provided.")
    else:
        # Check for async support (Streamlit handles running asyncio.run)
        with st.spinner("Connecting to Reddit and analyzing posts..."):
            try:
                user_creds = {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "username": username,
                    "password": password,
                    "user_agent": user_agent
                }
                
                # Use asyncio.run to execute the main async function
                sorted_posts = asyncio.run(reddit_main(
                    research_question,
                    subreddit_name,
                    limit,
                    user_creds,
                    llm_endpoint,
                    llm_api_key,
                    time_filter_value # Pass the selected time filter
                ))
                
                # We need to re-clean the subreddit name here for the display text
                display_subreddit_name = clean_subreddit_name(subreddit_name)
                
                if sorted_posts:
                    st.header(f"--- \n Search Results in r/{display_subreddit_name}")
                    st.markdown("Posts are ranked by **Relevance Score** (5 = highly relevant, 1 = not relevant).")
                    st.markdown(f"**:hourglass: Note: Posts are limited to the {selected_time_filter_label}.**") 
                    logging.info("Displaying sorted posts to the user.")
                    
                    for submission in sorted_posts:
                        st.markdown(f"**Relevance Score:** {submission.get('score', 'N/A')} / 5")
                        st.markdown(f"**Title:** {submission.get('title', 'N/A')}")
                        
                        # Display body if available
                        body_preview = submission.get('body', '')
                        if body_preview:
                            st.caption(f"Body Preview: {body_preview[:150]}...")
                            
                        # Display link
                        st.markdown(f"**Link:** [{submission.get('url', 'N/A')}]({submission.get('url', '#')})")

                        st.markdown("---")
                else:
                    # Check if the failure was due to a bad subreddit name
                    if not display_subreddit_name:
                         st.error("The subreddit name you entered was invalid after cleaning. Please enter a valid name (e.g., 'diabetes' or 'r/mentalhealth').")
                    else:
                        st.warning(f"No posts were found or analyzed in r/{display_subreddit_name}.")
                        logging.warning("No posts were found or analyzed. Displaying warning to user.")
                    
            except Exception as e:
                # Catch any unexpected errors from the main process
                if not st.session_state.get('error_displayed'):
                    st.error(f"An unexpected error occurred during the process: {e}")
                    logging.critical(f"An unexpected error occurred during the main Reddit process: {e}", exc_info=True)
                    st.session_state['error_displayed'] = True # Prevent multiple error displays on re-runs
                else:
                    logging.error("Suppressed re-run error display.")
