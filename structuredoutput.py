from anthropic import Anthropic
import requests
from bs4 import BeautifulSoup
import json
import os
from dotenv import load_dotenv
load_dotenv()

apikey = os.getenv('ANTHROPIC_API_KEY')

client = Anthropic(api_key=apikey)
MODEL_NAME = "claude-3-haiku-20240307"

# 1. Article Summary Tool
tools = [
    {
        "name": "print_summary",
        "description": "Prints a summary of the article.",
        "input_schema": {
            "type": "object",
            "properties": {
                "author": {"type": "string", "description": "Name of the article author"},
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": 'Array of topics, e.g. ["tech", "politics"]. Should be as specific as possible, and can overlap.'
                },
                "summary": {"type": "string", "description": "Summary of the article. One or two paragraphs max."},
                "coherence": {"type": "integer", "description": "Coherence of the article's key points, 0-100 (inclusive)"},
                "persuasion": {"type": "number", "description": "Article's persuasion score, 0.0-1.0 (inclusive)"}
            },
            "required": ['author', 'topics', 'summary', 'coherence', 'persuasion', 'counterpoint']
        }
    }
]

url = "https://www.anthropic.com/news/third-party-testing" #Add any URL here
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")
article = " ".join([p.text for p in soup.find_all("p")])

query = f"""
<article>
{article}
</article>

Use the `print_summary` tool.
"""

response = client.beta.tools.messages.create(
    model=MODEL_NAME,
    max_tokens=4096,
    tools=tools,
    messages=[{"role": "user", "content": query}]
)
json_summary = None
for content in response.content:
    if content.type == "tool_use" and content.name == "print_summary":
        json_summary = content.input
        break

if json_summary:
    print("JSON Summary:")
    print(json.dumps(json_summary, indent=2))
else:
    print("No JSON summary found in the response.")

# 2. Entity Extraction
tools = [
    {
        "name": "print_entities",
        "description": "Prints extract named entities.",
        "input_schema": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The extracted entity name."},
                            "type": {"type": "string", "description": "The entity type (e.g., PERSON, ORGANIZATION, LOCATION)."},
                            "context": {"type": "string", "description": "The context in which the entity appears in the text."}
                        },
                        "required": ["name", "type", "context"]
                    }
                }
            },
            "required": ["entities"]
        }
    }
]

text = "John works at Google in New York. He met with Sarah, the CEO of Acme Inc., last week in San Francisco."

query = f"""
<document>
{text}
</document>

Use the print_entities tool.
"""

response = client.beta.tools.messages.create(
    model=MODEL_NAME,
    max_tokens=4096,
    tools=tools,
    messages=[{"role": "user", "content": query}]
)

json_entities = None
for content in response.content:
    if content.type == "tool_use" and content.name == "print_entities":
        json_entities = content.input
        break

if json_entities:
    print("Extracted Entities (JSON):")
    print(json_entities)
else:
    print("No entities found in the response.")

# 3. Sentiment Analysis
tools = [
    {
        "name": "print_sentiment_scores",
        "description": "Prints the sentiment scores of a given text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "positive_score": {"type": "number", "description": "The positive sentiment score, ranging from 0.0 to 1.0."},
                "negative_score": {"type": "number", "description": "The negative sentiment score, ranging from 0.0 to 1.0."},
                "neutral_score": {"type": "number", "description": "The neutral sentiment score, ranging from 0.0 to 1.0."}
            },
            "required": ["positive_score", "negative_score", "neutral_score"]
        }
    }
]

text = "The product was okay, but the customer service was terrible. I probably won't buy from them again."

query = f"""
<text>
{text}
</text>

Use the print_sentiment_scores tool.
"""

response = client.beta.tools.messages.create(
    model=MODEL_NAME,
    max_tokens=4096,
    tools=tools,
    messages=[{"role": "user", "content": query}]
)

json_sentiment = None
for content in response.content:
    if content.type == "tool_use" and content.name == "print_sentiment_scores":
        json_sentiment = content.input
        break

if json_sentiment:
    print("Sentiment Analysis (JSON):")
    print(json.dumps(json_sentiment, indent=2))
else:
    print("No sentiment analysis found in the response.")

# 4. Text Classification
tools = [
    {
        "name": "print_classification",
        "description": "Prints the classification results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "The category name."},
                            "score": {"type": "number", "description": "The classification score for the category, ranging from 0.0 to 1.0."}
                        },
                        "required": ["name", "score"]
                    }
                }
            },
            "required": ["categories"]
        }
    }
]

text = "The new quantum computing breakthrough could revolutionize the tech industry."

query = f"""
<document>
{text}
</document>

Use the print_classification tool. The categories can be Politics, Sports, Technology, Entertainment, Business.
"""

response = client.beta.tools.messages.create(
    model=MODEL_NAME,
    max_tokens=4096,
    tools=tools,
    messages=[{"role": "user", "content": query}]
)

json_classification = None
for content in response.content:
    if content.type == "tool_use" and content.name == "print_classification":
        json_classification = content.input
        break

if json_classification:
    print("Text Classification (JSON):")
    print(json.dumps(json_classification, indent=2))
else:
    print("No text classification found in the response.")
