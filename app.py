import os
import re

import streamlit as st
from google.cloud import dlp_v2
from google.cloud.dlp_v2 import types
from langchain.agents import create_sql_agent
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.messages import AIMessage
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    PromptTemplate,
    FewShotPromptTemplate,
)
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.memory import ConversationBufferMemory
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.sql_database import SQLDatabase
from langchain.tools import Tool


# Setup
service_account_file = f"{os.getcwd()}/xxxx.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file

model = ChatOpenAI(model="gpt-4o", temperature=0)
project = "xxxxxx"
dataset = "customer_profiles"
sqlalchemy_url = (
    f"bigquery://{project}/{dataset}?credentials_path={service_account_file}"
)
db = SQLDatabase.from_uri(sqlalchemy_url)

# Example Queries
sql_examples = [
    {
        "input": "Count of Customers by Source System",
        "query": f"""
            SELECT
                source_system_name,
                COUNT(*) AS customer_count
            FROM
                `{project}.{dataset}.customer`
            GROUP BY
                source_system_name
            ORDER BY
                customer_count DESC;
        """,
    },
    {
        "input": "Average Age of Customers by Gender",
        "query": f"""
            SELECT
                gender,
                AVG(EXTRACT(YEAR FROM CURRENT_DATE()) - EXTRACT(YEAR FROM dob)) AS average_age
            FROM
                `{project}.{dataset}.customer`
            GROUP BY
                gender;
        """,
    },
    {
        "input": "Count of Customers with Email and/or Phone",
        "query": f"""
            SELECT
                c.customer_key,
                c.first_name,
                c.last_name,
                SUM(CASE WHEN ct.type = 'email' THEN 1 ELSE 0 END) AS email_count,
                SUM(CASE WHEN ct.type = 'phone' THEN 1 ELSE 0 END) AS phone_count
            FROM
                `{project}.{dataset}.customer` c
            LEFT JOIN
                `{project}.{dataset}.contact` ct
            ON
                c.customer_key = ct.customer_key
            GROUP BY
                c.customer_key, c.first_name, c.last_name
            ORDER BY
                email_count DESC, phone_count DESC;
        """,
    },
    {
        "input": "List of Customers with Addresses",
        "query": f"""
            SELECT
                c.customer_key,
                c.first_name,
                c.last_name,
                a.full_address,
                a.state,
                a.country
            FROM
                `{project}.{dataset}.customer` c
            JOIN
                `{project}.{dataset}.customer_address` ca
            ON
                c.customer_key = ca.customer_key
            JOIN
                `{project}.{dataset}.address` a
            ON
                ca.address_key = a.address_key;
        """,
    },
    {
        "input": "Job States Summary",
        "query": f"""
            SELECT
                batch_id,
                status,
                record_count,
                load_timestamp,
                JSON_EXTRACT_SCALAR(job_summary, '$.SYS1') AS sys1_count,
                JSON_EXTRACT_SCALAR(job_summary, '$.SYS2') AS sys2_count,
                JSON_EXTRACT_SCALAR(job_summary, '$.SYS3') AS sys3_count,
                JSON_EXTRACT_SCALAR(job_summary, '$.SYS4') AS sys4_count,
                JSON_EXTRACT_SCALAR(job_summary, '$.SYS5') AS sys5_count
            FROM
                `{project}.{dataset}.job_states`
            ORDER BY
                load_timestamp DESC;
        """,
    },
    {
        "input": "Top 5 Most Populated States",
        "query": f"""
            SELECT
                state,
                COUNT(*) AS address_count
            FROM
                `{project}.{dataset}.address`
            GROUP BY
                state
            ORDER BY
                address_count DESC
            LIMIT 5;
        """,
    },
    {
        "input": "Total Contacts (Emails and Phones) by Source System",
        "query": f"""
            SELECT
                c.source_system_name,
                SUM(CASE WHEN ct.type = 'email' THEN 1 ELSE 0 END) AS total_emails,
                SUM(CASE WHEN ct.type = 'phone' THEN 1 ELSE 0 END) AS total_phones
            FROM
                `{project}.{dataset}.customer` c
            JOIN
                `{project}.{dataset}.contact` ct
            ON
                c.customer_key = ct.customer_key
            GROUP BY
                c.source_system_name;
        """,
    },
    {
        "input": "Distribution of Customers by Age Groups",
        "query": f"""
            SELECT
                CASE
                    WHEN age < 20 THEN 'Under 20'
                    WHEN age BETWEEN 20 AND 29 THEN '20-29'
                    WHEN age BETWEEN 30 AND 39 THEN '30-39'
                    WHEN age BETWEEN 40 AND 49 THEN '40-49'
                    WHEN age BETWEEN 50 AND 59 THEN '50-59'
                    ELSE '60 and above'
                END AS age_group,
                COUNT(*) AS customer_count
            FROM
                (SELECT
                    EXTRACT(YEAR FROM CURRENT_DATE()) - EXTRACT(YEAR FROM dob) AS age
                FROM
                    `{project}.{dataset}.customer`)
            GROUP BY
                age_group
            ORDER BY
                customer_count DESC;
        """,
    },
    {
        "input": "Customers with Multiple Source Systems",
        "query": f"""
            SELECT
                first_name,
                last_name,
                COUNT(DISTINCT source_system_name) AS source_system_count
            FROM
                `{project}.{dataset}.customer`
            GROUP BY
                first_name, last_name
            HAVING
                source_system_count > 1;
        """,
    },
    {
        "input": "Recent Job Runs with Their Status",
        "query": f"""
            SELECT
                batch_id,
                status,
                record_count,
                load_timestamp
            FROM
                `{project}.{dataset}.job_states`
            ORDER BY
                load_timestamp DESC
            LIMIT 10;
        """,
    },
]


PREFIX = """
You are a SQL expert. You have access to a BigQuery database.
Identify which tables can be used to answer the user's question and write and execute a SQL query accordingly.
Given an input question, create a syntactically correct SQL query to run against the dataset customer_profiles, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table; only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the information returned by these tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

If the user asks for a visualization of the results, use the python_agent tool to create and display the visualization.

After obtaining the results, you must use the mask_pii_data tool to mask the results before providing the final answer.
"""

SUFFIX = """Begin!

{chat_history}

Question: {input}
Thought: I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables.
{agent_scratchpad}"""


def mask_pii_data(text):
    dlp = dlp_v2.DlpServiceClient()

    project_id = project
    parent = f"projects/{project_id}"

    info_types = [
        {"name": "EMAIL_ADDRESS"},
        {"name": "PHONE_NUMBER"},
        {"name": "DATE_OF_BIRTH"},
        {"name": "LAST_NAME"},
        {"name": "STREET_ADDRESS"},
        {"name": "LOCATION"},
    ]

    deidentify_config = types.DeidentifyConfig(
        info_type_transformations=types.InfoTypeTransformations(
            transformations=[
                types.InfoTypeTransformations.InfoTypeTransformation(
                    primitive_transformation=types.PrimitiveTransformation(
                        character_mask_config=types.CharacterMaskConfig(
                            masking_character="*", number_to_mask=0, reverse_order=False
                        )
                    )
                )
            ]
        )
    )

    item = {"value": text}
    inspect_config = {"info_types": info_types}
    request = {
        "parent": parent,
        "inspect_config": inspect_config,
        "deidentify_config": deidentify_config,
        "item": item,
    }

    response = dlp.deidentify_content(request=request)

    return response.item.value


python_repl = PythonREPL()


def sql_agent_tools():
    tools = [
        Tool.from_function(
            func=mask_pii_data,
            name="mask_pii_data",
            description="Masks PII data in the input text using Google Cloud DLP.",
        ),
        Tool(
            name="python_repl",
            description=f"A Python shell. Use this to execute python commands. \
              Input should be a valid python command. \
              If you want to see the output of a value, \
              you should print it out with `print(...)`.",
            func=python_repl.run,
        ),
    ]
    return tools


example_selector = SemanticSimilarityExampleSelector.from_examples(
    sql_examples,
    OpenAIEmbeddings(),
    FAISS,
    k=2,
    input_keys=["input"],
)

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    prefix=PREFIX,
    suffix="",
    input_variables=["input", "top_k"],
    example_separator="\n\n",
)

messages = [
    SystemMessagePromptTemplate(prompt=few_shot_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
    AIMessage(content=SUFFIX),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
]
prompt = ChatPromptTemplate.from_messages(messages)
extra_tools = sql_agent_tools()

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, input_key="input"
)

# Create the agent executor
agent_executor = create_sql_agent(
    llm=model,
    db=db,
    verbose=True,
    top_k=10,
    prompt=prompt,
    extra_tools=extra_tools,
    input_variables=["input", "agent_scratchpad", "chat_history"],
    agent_type="openai-tools",
    agent_executor_kwargs={"handle_parsing_errors": True, "memory": memory},
)

st.title("Data Analytics Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask your question:")

if st.button("Run Query"):
    if user_input:
        with st.spinner("Processing..."):
            st.session_state.history.append(f"User: {user_input}")

            response = agent_executor.run(input=user_input)

            if "sandbox:" in response:
                response = response.replace(f"sandbox:", "")

            match = re.search(r"\((.+\.png)\)", response)
            if match:
                image_file_path = match.group(1)
                if os.path.isfile(image_file_path):
                    st.session_state.history.append({"image": image_file_path})
                else:
                    st.error("The specified image file does not exist.")
            else:
                st.session_state.history.append(f"Agent: {response}")

            st.experimental_rerun()
    else:
        st.error("Please enter a question.")

for message in st.session_state.history:
    if isinstance(message, str):
        st.write(message)
    elif isinstance(message, dict) and "image" in message:
        st.image(message["image"])
