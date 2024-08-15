import json
import os
import requests
import sseclient
import subprocess
import threading
import time
from parse_trunc import all_trunc_parser

# REGION = "asia-northeast1"
REGION = "asia-east1"
# REGION = "asia-east2"
PROJECT_ID = "micro-citadel-425019-a2"
MODEL = "gemini-1.5-pro"
GCLOUD_PATH = "/Users/zebralee/Downloads/google-cloud-sdk/bin/gcloud"

class GeminiContentGenerator:
    """A class to generate content using various models and publishers."""

    def __init__(
        self,
        region=REGION,
        project_id=PROJECT_ID,
        model=MODEL,
        gcloud_path=GCLOUD_PATH,
        publisher="google",
        model_version=None,  # For Mistral models
    ):
        """
        Initialize the GeminiContentGenerator.

        Args:
            region (str): The region where the model is deployed.
            project_id (str): The project ID.
            model (str): The model to use.
            gcloud_path (str): The path to the gcloud executable.
            publisher (str): The publisher of the model.
            model_version (str, optional): The version of the model. Defaults to None.
        """
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
        self.parser = all_trunc_parser[publisher]  # This parser converts the model's output trunc to plain text
        # Construct URL based on the publisher and model
        if self.publisher == "google":
            url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/google/models/{self.model}:streamGenerateContent?alt=sse"
        elif self.publisher == "anthropic":
            url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/anthropic/models/{self.model}:streamRawPredict"
        elif self.publisher == "mistralai":
            if not self.model_version:
                raise ValueError("Model version is required for Mistral models.")
            selected_model_version = "" if self.model_version == "latest" else f"@{self.model_version}"
            url = f"https://{self.region}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.region}/publishers/mistralai/models/{self.model}{selected_model_version}:streamRawPredict"
        elif self.publisher == "openapi":
            url = f"https://{self.region}-aiplatform.googleapis.com/v1beta1/projects/{self.project_id}/locations/{self.region}/endpoints/openapi/chat/completions"
        else:
            raise ValueError(
                f"Unsupported publisher: {self.publisher}. Choose from 'google', 'anthropic', 'mistralai', or 'openapi'."
            )
        self.url = url

        # Start the token refresh thread
        self.start_token_refresh_thread()

        # 从环境变量中读取代理设置
        http_proxy = os.getenv('http_proxy')
        https_proxy = os.getenv('https_proxy')
        all_proxy = os.getenv('all_proxy')
        # proxy
        self.proxies = {
            'http': http_proxy,
            'https': https_proxy,
            'all': all_proxy
        }

    def get_gcloud_access_token(self):
        """
        Get the gcloud access token.

        Returns:
            str: The access token.
        """
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
        """Reload the gcloud access token."""
        self.headers["Authorization"] = f"Bearer {self.get_gcloud_access_token()}"

    def start_token_refresh_thread(self):
        """Start a thread to periodically refresh the gcloud access token."""
        def refresh_token_periodically():
            while True:
                time.sleep(1800)  # Sleep for 30 minutes
                self.reload_token()
                print("Token reloaded")

        thread = threading.Thread(target=refresh_token_periodically, daemon=True)
        thread.start()

    def __str__(self):
        return (f"GeminiContentGenerator(region={self.region}, project_id={self.project_id}, "
                f"model={self.model}, gcloud_path={self.gcloud_path}, publisher={self.publisher}, "
                f"model_version={self.model_version}, headers={self.headers}, url={self.url}), "
                f"proxies={self.proxies}"
                )

    def __repr__(self):
        return self.__str__()

    def generate_content(self, request_messages):
        """
        Generate content using the specified request messages.
        这个方法还未修改，目前只适用于Google格式

        Args:
            request_messages (str or dict): The request messages, either as a file path or a dictionary.
        """
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
            token = partial_result["candidates"][0]["content"]["parts"][0]["text"]  # only suitable for google style request
            print(token, end="", flush=True)

    def generate_content_stream(self, request_messages):
        """
        Generate a stream of content using the specified request messages.
        处理流程: google/anthropic/mistralai/openapi messages -> 通过generate_content_stream_raw_format发起请求 -> google/anthropic/mistralai/openapi raw_format -> 根据publisher转换成单字yield返回

        Args:
            request_messages (str or dict): The request messages, either as a file path or a dictionary.

        Yields:
            str: The parsed truncated content.
        """
        for partial_result in self.generate_content_stream_raw_format(request_messages):
            yield self.parser.parse_trunc(partial_result)

    def generate_content_stream_raw_format(self, request_messages, return_pure_text=True):
        """
        Generate a stream of raw content using the specified request messages.
        处理流程: google/anthropic/mistralai/openapi messages -> 请求 -> google/anthropic/mistralai/openapi raw_format

        Args:
            request_messages (str or dict): The request messages, either as a file path or a dictionary.
            return_pure_text (bool): Whether to return pure text or raw format.

        Yields:
            str or dict: The raw content or pure text.
        """
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

        response = requests.post(
            self.url, json=request_messages, headers=self.headers, stream=True, proxies=self.proxies
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
            yield partial_result

# Define the all_generators dictionary at the top level
all_generators = {}

# Initialize the generators for each publisher
all_generators["google"] = GeminiContentGenerator(
    model="gemini-1.5-pro",
    region="asia-east1",
    publisher="google",
    model_version=None,
)
all_generators["anthropic"] = GeminiContentGenerator(
    model="claude-3-5-sonnet@20240620",
    region="us-east5",
    publisher="anthropic",
    model_version=None,
)
all_generators["mistralai"] = GeminiContentGenerator(
    model="mistral-large",
    region="europe-west4",
    publisher="mistralai",
    model_version="2407",
)
all_generators["openapi"] = GeminiContentGenerator(
    model="meta/llama3-405b-instruct-maas",
    region="us-central1",
    publisher="openapi",
    model_version=None,
)


if __name__ == "__main__":
    # print(all_trunc_parser)
    # publishers = ["google", "anthropic", "mistralai", "openapi"]
    publishers = []
    
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
            
    # test_return_pure_content_publisher = "openapi"
    test_return_pure_content_publisher = "anthropic"
    
    print("\n", test_return_pure_content_publisher, ":")
    
    if test_return_pure_content_publisher == "google":
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
        generator = GeminiContentGenerator(
            region=REGION,
            model=MODEL,
            publisher="google",
        )
        for word in generator.generate_content_stream(REQUEST_MESSAGES_DICT):
            print(word, flush=True, end="")
    elif test_return_pure_content_publisher == "anthropic":
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
        print(generator)
        # for word in generator.generate_content_stream(PAYLOAD):
        #     print(word, flush=True, end="")
            
    elif test_return_pure_content_publisher == "mistralai":

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
        for word in generator.generate_content_stream(PAYLOAD):
            print(word, flush=True, end="")
    elif test_return_pure_content_publisher == "openapi": # or another name
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
        for word in generator.generate_content_stream(PAYLOAD):
            print(word, flush=True, end="")