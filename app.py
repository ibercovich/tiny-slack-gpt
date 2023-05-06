"""
Single file app to run a GPT slackbot
Includes basic Text2SQL capabilities

1-) create application on slack
2-) enable socket mode
3-) create OAuth tokens
4-) add slash commands for /query and /summarize
5-) figure out your bot user ID
6-) enable the right scopes:

Bot Token Scopes    |    User Token Scopes
=============================================      
app_mentions:read   |    channels:history
channels:history    |    chat:write
channels:read       |    groups:history
chat:write          |    im:history
commands            |    im:read
groups:history      |    im:write
groups:read         |    mpim:history
im:history          |    mpim:read
mpim:history        |    mpim:write
users:read          |

7-) run from terminal, cron, or as a service
"""
from __future__ import print_function

# Python built-in modules
import os
import re
import time
import socket
import sqlite3

# Third-party modules
import openai
import tiktoken
from tabulate import tabulate
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


#############################
# load local only credentials
##############################
# best to add credentials to .env file
# on same folder and keep it out of git
# you can use pip import dotenv
SLACK_BOT_TOKEN = "xoxb-..."
SLACK_APP_TOKEN = "xapp-..."
SLACK_BOT_USER = "U..."
OPENAI_API_KEY = "sk-..."
OPENAI_ORG = "org-..."
OPENAI_MODEL = "gpt-3.5-turbo" #or "gpt-4"
TOKEN_LIMIT = 8000 if OPENAI_MODEL == "gpt-4" else 4000
# other params
DATA_DB = '...' #sqlite file name _.db
DATA_TABLE = '...' #table names


#########################################
# check there is internet connection or
# program fails when connecting to Slack
#########################################
def is_connected():
    """
    Check if there is an internet connection.
    Returns True if connected, False otherwise.
    """
    try:
        conn = socket.create_connection(("www.google.com", 80))
        if conn is not None:
            conn.close()
        return True
    except OSError:
        pass
    return False


while not is_connected():
    time.sleep(5)
print("is connected!")


########################
# initialize APIs
########################
app = App(token=SLACK_BOT_TOKEN)
web_client = WebClient(token=SLACK_BOT_TOKEN)
openai.organization = OPENAI_ORG
openai.api_key = OPENAI_API_KEY
encoding = tiktoken.encoding_for_model(OPENAI_MODEL)


########################
# Slack event listeners
########################


@app.event("app_mention")
def mention_handler(body, say):
    """
    Gets called when the bot is mentioned
    Message is passed to GPT, with context
    """
    channel_id = body["event"]["channel"]
    channel_name = get_channel_name(channel_id)
    history = get_message_history(channel_id)
    response = chat_gpt(history, channel_name)
    say({"text": response, "response_type": "in_channel"})


# pylint: disable=unused-argument
@app.command("/summarize")
def call_gpt_summarize(ack, say, command):
    """
    Summarizes conversation in existing channel, in thread
    """
    ack()
    channel = command["channel_id"]
    prompt = f"<@{command['user_id']}> asked to summarize conversation"
    m_ts = web_client.chat_postMessage(channel=channel, text=prompt)["ts"]
    history = get_message_history(command["channel_id"])
    text = summarize_conversation(history)
    web_client.chat_postMessage(channel=channel, text=text, thread_ts=m_ts)


@app.command("/query")
def data_query_command(ack, say, command):
    """
    Sends natural language question to DB
    """
    ack()
    question = command["text"].lower()
    query = text2sql(command["text"])

    if query:
        # post user question
        prompt = f"<@{command['user_id']}> asked: {question}"
        say({"text": prompt, "response_type": "in_channel"})
        # post query
        say({"text": f"```\n{query}\n```\n", "response_type": "in_channel"})
        data = fetch(query)
        # display raw data as table
        split_markdown(data, say)


# pylint: disable=unused-argument
@app.event("message")
def handle_message_events(body, logger):
    """
    catch-all for other events or it throws errors
    """


#############################
# Slack helper functions
#############################


def get_slack_users():
    """
    gets the list of users so they can be referred by name.
    otherwise, all we know are their IDs
    """
    result = web_client.users_list()
    result = {usr["id"]: usr["name"] for usr in result["members"]}
    return result


def get_channel_name(channel_id):
    """
    get channel name
    """
    # Get the channel information using the conversations.info API method
    response = web_client.conversations_info(channel=channel_id)

    # Extract the channel name from the response
    return response["channel"]["name"]


USERS = get_slack_users()


def get_message_history(channel_id):
    """
    gets message history, including threads, and removes unnecessary components
    """
    messages = get_channel_messages(channel_id)
    history = get_threads(channel_id, messages)
    history = [
        {"text": replace_user_id(conv["text"]), "user": USERS[conv["user"]]}
        for conv in history
    ]
    history = list(reversed(history))
    return history


def get_channel_messages(channel_id):
    """
    retrieves messages from a channel
    """
    try:
        result = web_client.conversations_history(channel=channel_id)
        messages = result["messages"]
        return messages
    except SlackApiError as error:
        print(f"Error: {error}")
        return []


def get_threads(channel_id, messages):
    """
    retrieves threads for messages that contain them
    flattens the whole structure into a single thread
    """
    all_messages = []
    for message in messages:
        all_messages.append(message)
        if "thread_ts" in message:
            try:
                result = web_client.conversations_replies(
                    channel=channel_id, ts=message["ts"]
                )
                # for some reason threads are ordered opposite to messages
                thread_messages = list(reversed(result["messages"]))
                all_messages.extend(thread_messages)
            except SlackApiError as error:
                print(f"Error: {error}")
    return all_messages


def split_markdown(code, say):
    """
    slack is bad at printing large comment blocks so I need this work around
    this will split blocks of text intended as markdown into multiple blocks
    """
    if isinstance(code, str):
        lines = code.splitlines()
        sections = ["\n".join(lines[i : i + 10]) for i in range(0, len(lines), 10)]
        for section in sections:
            table = "```\n" + section + "\n```"
            say({"text": table, "response_type": "in_channel"})


def replace_user_id(message):
    """
    Function to replace user IDs with user names
    Also remove duplicate mentions to users
    """

    def helper(match):
        """
        Function to replace user IDs with user names
        """
        user_id = match.group(1)
        return USERS.get(user_id, f"<@{user_id}>")

    pattern = r"<@([A-Z0-9]+)>"
    # remove name: at the start of the message
    message = re.sub(r"^\w+:\s*", "", message)
    # replace mention/ids with names
    return re.sub(pattern, helper, message)


def get_root_dir():
    """
    Returns root dir for the main file, in this case app.py
    """
    # pylint: disable=no-member
    main_script_path = os.path.abspath(__file__)
    main_script_dir = os.path.dirname(os.path.realpath(main_script_path))
    return main_script_dir


#############################
# SQLite functions
#############################


def text2sql(text):
    """
    takes a text prompt and returns sql
    """
    schema = get_json_schema(DATA_TABLE)
    rows, cols = data_query(f"SELECT * FROM {DATA_TABLE};")
    markdown_table = get_markdown_table(rows, cols, pretty=False)

    query = call_gpt(
        get_prompt_template(
            "query",
            params={
                "table": DATA_TABLE,
                "schema": schema,
                "data": markdown_table,
                "question": text,
            },
        )
    ).lower()
    # remove code blocks which appear ocassionally
    query = query.replace("sql", "")
    query = query.replace("```", "")
    return query.strip()


def fetch(query):
    """
    takes a query and returns data and explanation

        call_gpt(
            get_prompt_template(
                "explain_query",
                params={"question": question, "query": query, "response": rows},
            )
        )
    """
    rows, cols = data_query(query)
    data = get_markdown_table(rows, cols)
    return data


def get_markdown_table(rows, cols, pretty=True):
    """
    returns markdown table for one or more items
    """
    # make headers multi-line
    if pretty:
        cols = [col.replace("_", "\n") for col in cols]
    return tabulate(rows, cols, floatfmt=".2f", intfmt=",")


def data_query(query):
    """
    runs the actual query
    """
    rows = []
    cols = []
    if query:
        conn, cursor = get_db_conn()
        cursor.execute(query)
        cols = [col[0] for col in cursor.description]
        rows = cursor.fetchall()
        conn.close()
    return rows, cols


def get_json_schema(table):
    """
    get's the schema for the table
    """
    # Retrieve the schema of the table and convert it to a JSON object
    rows, _ = data_query(f"PRAGMA table_info({table})")
    schema = [f'column: "{row[1]}", type: "{row[2]}"' for row in rows]
    schema = "\n".join(schema)
    return schema


def get_db_conn(mode="ro", db_path=None):
    """
    Parameters:
    mode (string): default 'ro' (read only), alternatively 'rw' (read write).
    db_path (path): path where the db is located, alternatively it gets resolved from .env

    Returns:
    conn (sqlite connection):
    cursor (sqlite cursor):
    """
    if db_path is None:
        db_path = os.path.join(get_root_dir(), DATA_DB)
    # Connect to the database, read only
    conn = sqlite3.connect(f"file:{db_path}?mode={mode}", uri=True)
    # Create a cursor object
    cursor = conn.cursor()
    return conn, cursor


#############################
# GPT functions
#############################


def trim_list_by_tokens(str_list, prompt_header):
    """
    Takes a list and trims it based on the number of tokens allowed
    This approach is useful to avoid cutting messages in half
    Removes the token legth of the prompt header
    """
    count = 0
    ret = []
    if isinstance(str_list, list):
        prompt_token_len = len(encoding.encode(prompt_header)) + 25  # add a buffer
        # discount tokens for separator
        prompt_token_len += len(encoding.encode("\n\n###\n\n")) * len(str_list)
        # we reverse it so we only end up cropping the oldest messages
        for item in reversed(str_list):
            tokens = encoding.encode(item)
            count += len(tokens)
            if count > TOKEN_LIMIT - prompt_token_len:
                break
            ret.append(item)
        # we reverse the list again so it's in chronological order
        ret = list(reversed(ret))
        ret = "\n\n###\n\n".join(ret)
    return ret


def summarize_conversation(history):
    """
    uses GPT to summarize a conversation between various participants
    """
    if len(history) > 0:
        history = [f'{conv["user"]}: {conv["text"]}' for conv in history]
        history = trim_list_by_tokens(history, get_prompt_template("summarize"))
        return call_gpt(get_prompt_template("summarize", {"history": history}))
    return None


def chat_gpt(history, channel):
    """
    calls GPT with context, so it behaves like chat
    """
    history = [f'{conv["user"]}: {conv["text"]}' for conv in history]
    history = trim_list_by_tokens(history, get_prompt_template("bot_mention"))
    prompt = get_prompt_template(
        "bot_mention", {"history": history, "channel": channel}
    )

    response = call_gpt(prompt)
    return response


def call_gpt(prompt, model=OPENAI_MODEL):
    """
    calls GPT
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": get_prompt_template("system")},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def get_prompt_template(template, params=None):
    """
    Params need to be a dict with keys that match template
    """
    content = PROMPTS[template]
    if isinstance(params, dict):
        content = content.format(**params)
    return content


PROMPTS = {
    "system":       """You are a helpful assistant named scopy that works for a venture capital company named ScOp VC.""",

    "bot_mention":  """You are a helpful assistant that is part of a venture capital firm, named ScOp. Your name is scopy.
                    Below you will see a conversation in a particular slack channel among several members of this veture 
                    capital firm named ScOp. The next time someone asks a question, please consider the existing conversation 
                    context and provide a helpful answer. Sometimes, the answer will be unrelated to the conversation, 
                    in which case you should come up with your own. The name of the channel is {channel}. 
                    The channel name might be related to one of the companies in ScOp's portfolio. Always be nice and respecful, 
                    but if the person asking the question is named Kevin, answer accurately but in a funny tone, as if you are 
                    exasperated because Kevin is always asking silly questions. The last paragraph is the user's most recent 
                    question, which is intended for you scopy. \n\n {history} \n\n###\n\n """,

    "query":        """Please regard the following table, named "{table}" with the following schema: \n###\n ```\n {schema} \n``` \n###\n
                    Here's what the data looks like: \n###\n ```\n {data} \n``` \n###\n
                    Write a SQL query, without any explanation of any kind, just the SQL query, 
                    to answer the following question: {question}""",

    "summarize":    """Please write a thorough summary the following conversation. There are many participants.
                    Each time a participant speaks, a paragraph with start with the person's name.
                    The conversation follows below: \n\n###\n\n {history}"""
}


if __name__ == "__main__":
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
