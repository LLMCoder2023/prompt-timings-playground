"""
##### IMPORTANT NOTES #####
1. Edit setup-environment.sh as you may have to remove the "3" in python3 and pip3 depending on your system
2. Run "chmod +x setup-environment.sh" in your terminal
3. Run "source ./setup-environment.sh" in your terminal
4. Authenticate with AWS and then run "streamlit run [PYTHON-APP-FILE-NAME].py" in your terminal.  A browser window/tab will appear with the application.
#####
"""

import anthropic
import boto3
from botocore.config import Config
import json
import os
import time
import sys
import streamlit as st
from llm_claude_3 import LLM_Claude_3

LLM_CLAUDE_3 = LLM_Claude_3()

# Set Streamlit page configuration
st.set_page_config(page_title="Prompt Timings Playground", layout="wide")
st.title("🤖 Prompt Timings Playground")

st.subheader(body="Only configured for Anthropic Claude 3 models on Amazon Bedrock...")
st.divider()

if (
    "llm_result_timings" not in st.session_state
    or st.session_state.llm_result_timings == None
    or st.session_state.llm_result_timings == ""
):
    st.session_state.llm_result_timings = "placeholder"

if (
    "llm_results" not in st.session_state
    or st.session_state.llm_results == None
    or st.session_state.llm_results == ""
):
    st.session_state.llm_results = "placeholder"

if (
    "model_id" not in st.session_state
    or st.session_state.model_id == None
    or st.session_state.model_id == ""
):
    st.session_state.model_id = ""


def exec_time(start, end):
    elapsed = end - start
    seconds = int(elapsed)
    milliseconds = int(elapsed * 1000) % 1000

    execution_time_string = f" {seconds} Seconds and {milliseconds} Milliseconds"
    execution_time_number = "{0:02d}.{1:0.6f}".format(seconds, milliseconds)

    print(execution_time_number)
    print(execution_time_string)
    return execution_time_number, execution_time_string


def format_func(option):
    return bedrock_model_choices[option]


# Sidebar info
with st.sidebar:

    bedrock_model_choices = {
        "anthropic.claude-3-haiku-20240307-v1:0": "Claude 3 Haiku v1",
        "anthropic.claude-3-sonnet-20240229-v1:0": "Claude 3 Sonnet v1",
    }
    st.session_state.model_id = st.selectbox(
        label="Model Selection",
        placeholder="Choose an option...",
        options=list(bedrock_model_choices.keys()),
        format_func=format_func,
    )
    # st.session_state.model_id = "anthropic.claude-instant-v1"

    st.markdown("## Inference Parameters")
    TEMPERATURE = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.1
    )
    TOP_P = st.slider("Top-P", min_value=0.0, max_value=1.0, value=0.9, step=0.01)
    TOP_K = st.slider("Top-K", min_value=1, max_value=500, value=10, step=5)
    MAX_TOKENS = st.slider("Max Token", min_value=0, max_value=2048, value=1024, step=8)


input_container = st.container()
results_container = st.container()
prompt_container = st.container()

with input_container:
    st.write(f"Model Selected: {st.session_state.model_id}")
    query_try = st.text_input(
        label="Enter your query",
        key="query_try",
        value="Is there a significant relationship between ice cream consumption and drowning deaths at the local freshwater lake?",
        placeholder="Enter your query here...",
    )

    query_submit = st.button(label="Run test...", key="query_submit", type="primary")
    st.divider()

with prompt_container:
    st.markdown("<h4>Prompt - (Editable)</h4>", unsafe_allow_html=True)
    st.warning(
        icon="⚠️",
        body="""Don't edit what is originally line 6, "<user_query>{user_query}</user_query>" """,
    )

    my_prompt = st.code(
        language="xml",
        line_numbers=True,
        body="""<role>
    1. You are a distinguished expert at translating natural language queries into a logical operator search API format.
    2. You are an expert at expanding a researcher search query into complete sentences and offering alternative versions of the query.
    </role>

    <user_query>{user_query}</user_query>

    <example>
    <example_query_a>What is the relationship between gut bacteria and obesity?</example_query_a>
    <example_query_a_generated_clause>(("gut bacteria" OR "gut microbiota" OR "gut microflora") AND (obesity OR obese OR overweight)) OR (("gut bacteria" OR "gut microbiota" OR "gut microflora" OR microbiome OR microbiota) AND (obesity OR obese OR overweight OR "body mass index" OR BMI OR "waist circumference")) OR (("gut bacteria" OR "gut microbiota") AND (obesity OR obese))</example_query_a_generated_clause>
    </example>

    <tasks>
    1.  Take a deep breath and focus on the search query provided in the <user_query></user_query> XML tags.
    2.  You will expand the user's query, provided in the <user_query></user_query> xml tags, into complete sentences and generate alternative versions of those queries.
    3.  Given the alternates query in conjunction with the original user query provided in the <user_query></user_query> XML tags, extract keywords or phrases from the search string and keep them in mind for task 3.
    4.  Given task 2's result and the query provided in the <user_query></user_query> XML tags, write three search clauses that consist of search terms combined with AND and OR logical operators with parentheses added if nesessary
    5.  You MUST format your answer based on the XML style format provided in the <answer_format></answer_format> xml tags.
    6.  An example of a good response has been provided in the <example></example> XML tags.

    </tasks>

    <task_guidance>
    * The search clauses that you generated will be used to search a document search API.
    * The user's query is located in the <user_query></user_query> xml tags.
    * Put parentheses to form the clauses as needed.
    * If you end up with a clause formatted such as "A OR B AND C OR D", that really means A OR (B AND C) OR D and your answer should reflect logical groupings like that.
    * Ignore all meta information such as dates, publishing information.
    * Ignore common words such as "papers"
    * SKIP THE PREAMBLE, GO STRAIGHT TO THE ANSWER
    * YOU MUST NOT SHARE YOUR THOUGHT PROCESS
    </task_guidance>

    <answer_format>
    <search_clauses>
    <search_clause_1>[YOUR FIRST GENERATED SEARCH CLAUSE]</search_clause_1>
    <search_clause_2>[YOUR SECOND GENERATED SEARCH CLAUSE]</search_clause_2>
    <search_clause_3>[YOUR THIRD GENERATED SEARCH CLAUSE]</search_clause_3>
    </search_clauses>
    </answer_format>

        """,
    )

if query_submit:

    inference_configuration = {
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_tokens_to_sample": MAX_TOKENS,
        "stop_sequences": ["\n\nHuman:"],
    }

    my_prompt = my_prompt.replace("{user_query}", query_try)

    st.session_state.llm_results, st.session_state.llm_result_timings = (
        LLM_CLAUDE_3.call_llm_claude_3(
            my_prompt, inference_configuration, st.session_state.model_id
        )
    )

    with results_container:
        st.markdown("<h4>Timings</h4>", unsafe_allow_html=True)
        st.success(f"Timings: {st.session_state.llm_result_timings}")

        st.divider()

        results_container = st.container()
        with results_container:
            st.markdown("<h4>Results</h4>", unsafe_allow_html=True)
            st.code(
                language="xml", body=st.session_state.llm_results, line_numbers=True
            )

        st.divider()
