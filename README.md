# Data Analytics Assistant with Langchain and This project demonstrates how to use Langchain and OpenAI to automate data analysis tasks, making it easier to derive insights from large datasets. The application connects to a BigQuery dataset, utilizes custom tools for data privacy and visualization, and provides an interactive interface through a Streamlit app.
https://yia333.medium.com/leveraging-generative-ai-for-data-analytics-with-langchain-and-openai-ce95b1fbaff2

#Create and activate a virtual environment (optional but recommended):

```
python3 -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```
Install the required libraries:
```
pip install -r requirements.txt
```

Set the environment variables:

```
export OPENAI_API_KEY="your-openai-api-key"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/service-account-file.json"

```

Update the connection settings in your script:

```
import os
from langchain import SQLDatabase, Tool, create_sql_agent
from openai import ChatOpenAI

# Set environment variables
service_account_file = "path/to/your/service-account-file.json"
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_file

# Initialize OpenAI model
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Connect to BigQuery
project = "your-gcp-project-id"
dataset = "your-bigquery-dataset"
sqlalchemy_url = f"bigquery://{project}/{dataset}?credentials_path={service_account_file}"
db = SQLDatabase.from_uri(sqlalchemy_url)

```

Running the Streamlit App

```
streamlit run app.py
```