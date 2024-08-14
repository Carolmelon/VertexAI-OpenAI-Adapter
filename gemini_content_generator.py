import json
import os
import requests
import sseclient
import subprocess
from parse_trunc import all_trunc_parser


REGION = "asia-northeast1"
# REGION = "asia-east1"
# REGION = "asia-east2"
PROJECT_ID = "micro-citadel-425019-a2"
MODEL = "gemini-1.5-pro"
GCLOUD_PATH = "/Users/zebralee/Downloads/google-cloud-sdk/bin/gcloud"


class GeminiContentGenerator:
    def __init__(
        self,
        region=REGION,
        project_id=PROJECT_ID,
        model=MODEL,
        gcloud_path=GCLOUD_PATH,
        publisher="google",
        model_version=None,  # For Mistral models
    ):
        self.region = region
        self.project_id = project_id
        self.model = model
        self.gcloud_path = gcloud_path
        self.publisher = publisher
        self.model_version = model_version
        self.headers = {
            "charset": "utf-8",
            "content-type": "application/json",
            "Authorization": f"Bearer {self.get_gcloud_access_token()}",
        }

    def get_gcloud_access_token(self):
        try:
            result = subprocess.run(
                [self.gcloud_path, "auth", "print-access-token"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            print(f"Error occurred: {e.stderr}")
            return None

    def reload_token(self):
        # token = self.get_gcloud_access_token()
        self.headers["Authorization"] = f"Bearer {self.get_gcloud_access_token()}"

    def generate_content(self, request_messages):
        if isinstance(request_messages, str):
            # If request_messages is a string, assume it's a file path
            try:
                with open(request_messages, "r", encoding="utf-8") as file:
                    request_messages = json.load(file)
            except FileNotFoundError:
                print(f"File not found: {request_messages}")
                return
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {request_messages}")
                return
        elif not isinstance(request_messages, dict):
            print(
                "Invalid input: request_messages should be a file path or a dictionary"
            )
            return

        # url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.model}:streamGenerateContent?alt=sse"

        # Construct URL based on the publisher and model
        if self.publisher == "google":
            url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.model}:streamGenerateContent?alt=sse"
        elif self.publisher == "anthropic":
            url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/anthropic/models/{self.model}:streamRawPredict"
        elif self.publisher == "mistralai":
            if not self.model_version:
                raise ValueError("Model version is required for Mistral models.")
            url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/mistralai/models/{self.model}@{self.model_version}:streamRawPredict"
        elif self.publisher == "openapi":
            url = f"https://{self.region}-aiplatform.googleapis.com/v1beta1/projects/{self.project_id}/locations/{self.region}/endpoints/openapi/chat/completions"
        else:
            raise ValueError(
                f"Unsupported publisher: {self.publisher}. Choose from 'google', 'anthropic', 'mistralai', or 'openapi'."
            )

        response = requests.post(
            url, json=request_messages, headers=self.headers, stream=True
        )
        response.raise_for_status()

        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data == "[DONE]":
                break

            partial_result = json.loads(event.data)
            token = partial_result["candidates"][0]["content"]["parts"][0]["text"]
            print(token, end="", flush=True)

    def generate_content_stream(self, request_messages):
        if isinstance(request_messages, str):
            # If request_messages is a string, assume it's a file path
            try:
                with open(request_messages, "r", encoding="utf-8") as file:
                    request_messages = json.load(file)
            except FileNotFoundError:
                print(f"File not found: {request_messages}")
                return
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {request_messages}")
                return
        elif not isinstance(request_messages, dict):
            print(
                "Invalid input: request_messages should be a file path or a dictionary"
            )
            return

        # url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.model}:streamGenerateContent?alt=sse"
        
        # Construct URL based on the publisher and model
        if self.publisher == "google":
            url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.model}:streamGenerateContent?alt=sse"
        elif self.publisher == "anthropic":
            url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/anthropic/models/{self.model}:streamRawPredict"
        elif self.publisher == "mistralai":
            if not self.model_version:
                raise ValueError("Model version is required for Mistral models.")
            url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/mistralai/models/{self.model}@{self.model_version}:rawPredict"
        elif self.publisher == "openapi":
            url = f"https://{self.region}-aiplatform.googleapis.com/v1beta1/projects/{self.project_id}/locations/{self.region}/endpoints/openapi/chat/completions"
        else:
            raise ValueError(
                f"Unsupported publisher: {self.publisher}. Choose from 'google', 'anthropic', 'mistralai', or 'openapi'."
            )
        
        response = requests.post(
            url, json=request_messages, headers=self.headers, stream=True
        )
        response.raise_for_status()

        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data == "[DONE]":
                break

            partial_result = json.loads(event.data)
            token = partial_result["candidates"][0]["content"]["parts"][0]["text"] # this only suitable for gemini-1.5-pro
            yield token

    def generate_content_stream_raw_format(self, request_messages, return_pure_text=True):
        if isinstance(request_messages, str):
            # If request_messages is a string, assume it's a file path
            try:
                with open(request_messages, "r", encoding="utf-8") as file:
                    request_messages = json.load(file)
            except FileNotFoundError:
                print(f"File not found: {request_messages}")
                return
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file: {request_messages}")
                return
        elif not isinstance(request_messages, dict):
            print(
                "Invalid input: request_messages should be a file path or a dictionary"
            )
            return

        # url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.model}:streamGenerateContent?alt=sse"
        
        # Construct URL based on the publisher and model
        if self.publisher == "google":
            url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.model}:streamGenerateContent?alt=sse"
        elif self.publisher == "anthropic":
            url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/anthropic/models/{self.model}:streamRawPredict"
        elif self.publisher == "mistralai":
            if not self.model_version:
                raise ValueError("Model version is required for Mistral models.")
            SELECTED_MODEL_VERSION = "" if self.model_version == "latest" else f"@{self.model_version}"
            url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/mistralai/models/{self.model}{SELECTED_MODEL_VERSION}:streamRawPredict"
        elif self.publisher == "openapi":
            url = f"https://{self.region}-aiplatform.googleapis.com/v1beta1/projects/{self.project_id}/locations/{self.region}/endpoints/openapi/chat/completions"
        else:
            raise ValueError(
                f"Unsupported publisher: {self.publisher}. Choose from 'google', 'anthropic', 'mistralai', or 'openapi'."
            )
        
        response = requests.post(
            url, json=request_messages, headers=self.headers, stream=True
        )
        response.raise_for_status()

        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data == "[DONE]":
                break

            if return_pure_text:
                yield event.data
                continue
            partial_result = json.loads(event.data)
            # token = partial_result["candidates"][0]["content"]["parts"][0]["text"]
            yield partial_result


if __name__ == "__main__":
    print(all_trunc_parser)
    publishers = ["google", "anthropic", "mistralai", "openapi"]
    
    for publisher in publishers:
    
        print("\n", publisher, ":")
        
        parser = all_trunc_parser[publisher]

        if publisher == "google":
            # Example usage
            REQUEST_MESSAGES_PATH = "./gemini-message-example.json"
            REQUEST_MESSAGES_DICT = {
                "contents": [
                    {"role": "USER", "parts": {"text": "你是谁"}},
                    {
                        "role": "MODEL",
                        "parts": {"text": "我是OpenAI开发的gpt-4o-2024-05-13"},
                    },
                    {"role": "USER", "parts": {"text": "给我一份香蕉面包的食谱"}},
                ],
                "generation_config": {
                    "temperature": 0.2,
                    "topP": 0.8,
                    "topK": 40,
                    "maxOutputTokens": 100,
                },
                "systemInstruction": {
                    "role": "system",
                    "parts": [{"text": "回答尽可能简明扼要"}],
                },
            }

            # generator = GeminiContentGenerator(REGION, PROJECT_ID, MODEL, GCLOUD_PATH)
            generator = GeminiContentGenerator()
            # generator.generate_content(REQUEST_MESSAGES_PATH)  # Using file path
            # generator.generate_content(REQUEST_MESSAGES_DICT)  # Using dictionary

            # Using the generator function
            # for token in generator.generate_content_stream(REQUEST_MESSAGES_DICT):
            #     print(token, end="", flush=True)
            
            for partial_result in generator.generate_content_stream_raw_format(REQUEST_MESSAGES_DICT):
                # print(partial_result)
                print(parser.parse_trunc(partial_result), end="", flush=True)
        
        elif publisher == "anthropic":
            MODEL = "claude-3-5-sonnet@20240620"  # @param ["claude-3-5-sonnet@20240620", "claude-3-opus@20240229", "claude-3-haiku@20240307", "claude-3-sonnet@20240229" ]
            if MODEL == "claude-3-5-sonnet@20240620":
                available_regions = ["us-east5", "europe-west1"]
            elif MODEL == "claude-3-opus@20240229":
                available_regions = ["us-east5"]
            elif MODEL == "claude-3-haiku@20240307":
                available_regions = ["us-east5", "europe-west1"]
            elif MODEL == "claude-3-sonnet@20240229":
                available_regions = ["us-east5"]
                
            PAYLOAD = {
                "anthropic_version": "vertex-2023-10-16",
                "messages": [{"role": "user", "content": "给我一份香蕉面包的食谱"}],
                "max_tokens": 100,
                "stream": True,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 10,
            }
            
            generator = GeminiContentGenerator(
                model=MODEL,
                region=available_regions[0],
                publisher="anthropic",
                model_version=None,
            )
            for partial_result in generator.generate_content_stream_raw_format(PAYLOAD):
                # print(partial_result)
                print(parser.parse_trunc(partial_result), end="", flush=True)
                    
        
        elif publisher == "mistralai":

            MODEL = "mistral-large"  # @param ["mistral-large", "mistral-nemo", "codestral"]
            if MODEL == "mistral-large":
                available_regions = ["europe-west4", "us-central1"]
                available_versions = ["latest", "2407"]
            elif MODEL == "mistral-nemo":
                available_regions = ["europe-west4", "us-central1"]
                available_versions = ["latest", "2407"]
            elif MODEL == "codestral":
                available_regions = ["europe-west4", "us-central1"]
                available_versions = ["latest", "2405"]
                
            PAYLOAD = {
                "model": MODEL,
                "messages": [{"role": "user", "content": "给我一份香蕉面包的食谱"}],
                "max_tokens": 100,
                "top_p": 1.0,
                # "top_k": 10, # mistral不支持top_k参数
                "temperature": 1.0,
                "stream": True,
            }
            
            generator = GeminiContentGenerator(
                model=MODEL,
                region=available_regions[0],
                publisher="mistralai",
                model_version=available_versions[-1],
            )
            for partial_result in generator.generate_content_stream_raw_format(PAYLOAD):
                # print(partial_result)
                print(parser.parse_trunc(partial_result), end="", flush=True)
        
        elif publisher == "openapi": # or another name
            MODEL_ID = "meta/llama3-405b-instruct-maas"
            MODEL_LOCATION = "us-central1"
            PAYLOAD = {
                "model": MODEL_ID,
                "messages": [{"role": "user", "content": "给我一份香蕉面包的食谱"}],
                "max_tokens": 100,
                "top_p": 1.0,
                "top_k": 10,
                "temperature": 1.0,
                "stream": True,
            }
            
            generator = GeminiContentGenerator(
                model=MODEL,
                region=MODEL_LOCATION,
                publisher="openapi",
                model_version=None,
            )
            for partial_result in generator.generate_content_stream_raw_format(PAYLOAD):
                # print(partial_result)
                print(parser.parse_trunc(partial_result), end="", flush=True)
            