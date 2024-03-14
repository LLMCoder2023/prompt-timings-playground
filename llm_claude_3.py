import boto3
import base64
import json
import os
from anthropic import Anthropic
from botocore.config import Config
from botocore.exceptions import ClientError
import streamlit as st
import time

config = Config(read_timeout=600, retries=dict(max_attempts=5))  ## Handle retries

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime", region_name="us-west-2", config=config
)

"""
Invokes Anthropic Claude 3 Sonnet to run an inference using the input
provided in the request body.

:param prompt: The prompt that you want Claude 3 to complete.
:return: Inference response from the model.
"""


def exec_time(start, end):
    elapsed = end - start
    seconds = int(elapsed)
    milliseconds = int(elapsed * 1000) % 1000

    execution_time_string = f" {seconds} Seconds and {milliseconds} Milliseconds"
    execution_time_number = "{0:02d}.{1:0.6f}".format(seconds, milliseconds)

    print(execution_time_number)
    print(execution_time_string)
    return execution_time_number, execution_time_string


class LLM_Claude_3:
    claude_inference_configuration = {
        "temperature": 0.1,
        "top_p": 0.999,
        "top_k": 250,
        "max_tokens_to_sample": 1000,
    }

    def setup_bedrock_runtime(self):
        session = boto3.Session()
        config = Config(read_timeout=2000)
        bedrock_runtime = session.client(
            service_name="bedrock-runtime",
            config=config,
        )
        return bedrock_runtime

    def setup_langchain_bedrock_claude_3(
        self, bedrock_model_id, inference_configuration
    ):

        langchain_bedrock_claude_3 = Bedrock(
            model_id=bedrock_model_id,
            client=self.setup_bedrock_runtime(),
            verbose=True,
            model_kwargs=inference_configuration,
            cache=False,
        )

        return langchain_bedrock_claude_3

    def call_llm_claude_3(self, prompt, type, model_id):
        print(f"calling claude 3 model: {model_id}")
        # Invoke Claude 3 with the text prompt
        # model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        # model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        model_id = model_id
        max_tokens = 2000

        try:
            start = time.time()
            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "temperature": self.claude_inference_configuration[
                            "temperature"
                        ],
                        "top_p": self.claude_inference_configuration["top_p"],
                        "top_k": self.claude_inference_configuration["top_k"],
                        "max_tokens": max_tokens,
                        "messages": [
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": prompt}],
                            }
                        ],
                    }
                ),
            )
            end = time.time()
            llm_duration = exec_time(start, end)
            print(f"LLM Inference Time: f{llm_duration[1]}")

            # Process and print the response
            result = json.loads(response.get("body").read())
            input_tokens = result["usage"]["input_tokens"]
            output_tokens = result["usage"]["output_tokens"]
            output_list = result.get("content", [])
            for output in output_list:
                print(output["text"])
                result = output["text"]
                print(result)

            st.session_state.llm_result_timings = llm_duration[1]
            st.session_state.llm_results = result

            return result, llm_duration

        except ClientError as err:
            print(err.response["Error"]["Code"])
            print(err.response["Error"]["Message"])
            raise
