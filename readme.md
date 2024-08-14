

让Vertex-ai上serve的几个模型可以用OpenAI格式的消息请求

# Vertex-ai

### mistral

curl

```shell
# "us-central1" or "europe-west4"
GOOGLE_REGION="europe-west4" 
MODEL="mistral-large"
MODEL_VERSION="2407"
GOOGLE_PROJECT_ID="micro-citadel-425019-a2"
url="https://$GOOGLE_REGION-aiplatform.googleapis.com/v1/projects/$GOOGLE_PROJECT_ID/locations/$GOOGLE_REGION/publishers/mistralai/models/$MODEL@$MODEL_VERSION:rawPredict"

curl \
  -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  $url \
  --data '{
    "model": "'"$MODEL"'",
    "temperature": 0,
    "messages": [
      {"role": "user", "content": "What is the best French cheese?"}
    ]
}'
```

Request

```shell
MODEL="mistral-large"
MODEL_VERSION="2407"


url="https://$GOOGLE_REGION-aiplatform.googleapis.com/v1/projects/$GOOGLE_PROJECT_ID/locations/$GOOGLE_REGION/publishers/mistralai/models/$MODEL@$MODEL_VERSION:rawPredict"


curl \
  -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  $url \
  --data '{
    "model": "mistral-large",
    "temperature": 0,
    "messages": [
      {"role": "user", "content": "What is the best French cheese?"}
    ]
}'
```

response

```

```



## Jupyter发送请求的示例

### Gemini

非Jupyter，暂且放到这里

```python
import json
import os
import requests
import sseclient
import subprocess


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
            token = partial_result["candidates"][0]["content"]["parts"][0]["text"]
            yield token

    def generate_content_stream_raw_format(self, request_messages):
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
            # token = partial_result["candidates"][0]["content"]["parts"][0]["text"]
            yield partial_result


if __name__ == "__main__":
    publisher = "google"

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
                {"role": "USER", "parts": {"text": "你知道茴香豆的茴字有几种写法吗"}},
            ],
            "generation_config": {
                "temperature": 0.2,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": 500,
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
            print(partial_result)
            """
输出:
{'candidates': [{'content': {'role': 'model', 'parts': [{'text': '茴'}]}}]}
{'candidates': [{'content': {'role': 'model', 'parts': [{'text': '香豆的“茴”字只有一种写法。 😊 \n\n您'}]}, 'safetyRatings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.111328125, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.08251953}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.07470703, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.14453125}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.103515625, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.08251953}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.14160156, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.049560547}]}]}
{'candidates': [{'content': {'role': 'model', 'parts': [{'text': '是不是想起了鲁迅先生笔下的孔乙己？ \n'}]}, 'safetyRatings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.13769531, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.09814453}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.061767578, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.15625}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.10986328, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.16308594}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.16308594, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.07714844}]}]}
{'candidates': [{'content': {'role': 'model', 'parts': [{'text': ''}]}, 'finishReason': 'STOP'}], 'usageMetadata': {'promptTokenCount': 40, 'candidatesTokenCount': 31, 'totalTokenCount': 71}}
            """
    elif publisher == "mistralai":
        # TODO:
        pass
    elif publisher == "anthropic":
        # TODO:
        pass
    elif publisher == "llama3": # or another name
        # TODO:
        pass
```

#### Gemini 输出

```json
{'candidates': [{'content': {'role': 'model', 'parts': [{'text': '茴'}]}}]}
{'candidates': [{'content': {'role': 'model', 'parts': [{'text': '香豆的“茴”字只有一种写法。 😊 \n\n您'}]}, 'safetyRatings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.111328125, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.08251953}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.07470703, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.14453125}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.103515625, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.08251953}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.14160156, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.049560547}]}]}
{'candidates': [{'content': {'role': 'model', 'parts': [{'text': '是不是想起了鲁迅先生笔下的孔乙己？ \n'}]}, 'safetyRatings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.13769531, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.09814453}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.061767578, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.15625}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.10986328, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.16308594}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.16308594, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.07714844}]}]}
{'candidates': [{'content': {'role': 'model', 'parts': [{'text': ''}]}, 'finishReason': 'STOP'}], 'usageMetadata': {'promptTokenCount': 40, 'candidatesTokenCount': 31, 'totalTokenCount': 71}}
```





### anthropic

```python
! pip3 install -U -q httpx

import sys

if "google.colab" in sys.modules:

    import IPython

    app = IPython.Application.instance()
    app.kernel.do_shutdown(True)
    
import sys

if "google.colab" in sys.modules:

    from google.colab import auth

    auth.authenticate_user()
    
MODEL = "claude-3-5-sonnet@20240620"  # @param ["claude-3-5-sonnet@20240620", "claude-3-opus@20240229", "claude-3-haiku@20240307", "claude-3-sonnet@20240229" ]
if MODEL == "claude-3-5-sonnet@20240620":
    available_regions = ["us-east5", "europe-west1"]
elif MODEL == "claude-3-opus@20240229":
    available_regions = ["us-east5"]
elif MODEL == "claude-3-haiku@20240307":
    available_regions = ["us-east5", "europe-west1"]
elif MODEL == "claude-3-sonnet@20240229":
    available_regions = ["us-east5"]
    
import ipywidgets as widgets
from IPython.display import display

dropdown = widgets.Dropdown(
    options=available_regions,
    description="Select a location:",
    font_weight="bold",
    style={"description_width": "initial"},
)


def dropdown_eventhandler(change):
    global LOCATION
    if change["type"] == "change" and change["name"] == "value":
        LOCATION = change.new
        print("Selected:", change.new)


LOCATION = dropdown.value
dropdown.observe(dropdown_eventhandler, names="value")
display(dropdown)

PROJECT_ID = "micro-citadel-425019-a2"  # @param {type:"string"}
ENDPOINT = f"https://{LOCATION}-aiplatform.googleapis.com"

if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    raise ValueError("Please set your PROJECT_ID")
    
import base64
import json

import httpx
import requests
from IPython.display import Image

PAYLOAD = {
    "anthropic_version": "vertex-2023-10-16",
    "messages": [{"role": "user", "content": "Send me a recipe for banana bread."}],
    "max_tokens": 100,
    "stream": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 10,
}

request = json.dumps(PAYLOAD)
!curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" {ENDPOINT}/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/anthropic/models/{MODEL}:streamRawPredict -d '{request}'

"""
输出:
event: message_start
data: {"type":"message_start","message":{"id":"msg_vrtx_017o4feJdWfVviVPbMusHB1q","type":"message","role":"assistant","model":"claude-3-5-sonnet-20240620","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":15,"output_tokens":1}}    }

event: ping
data: {"type": "ping"}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}   }

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Here"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"'s"}       }

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" a classic banana brea"}  }

(中间省略一部分模型输出)

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" flour"}          }

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\n- Optional"}     }

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":": "}     }

event: content_block_stop
data: {"type":"content_block_stop","index":0           }

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":99}        }

event: message_stop
data: {"type":"message_stop"    }
"""
```

#### anthropic输出

```json
event: message_start
data: {"type":"message_start","message":{"id":"msg_vrtx_017o4feJdWfVviVPbMusHB1q","type":"message","role":"assistant","model":"claude-3-5-sonnet-20240620","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":15,"output_tokens":1}}    }

event: ping
data: {"type": "ping"}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}   }

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Here"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"'s"}       }

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" a classic banana brea"}  }

(中间省略一部分模型输出)

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" flour"}          }

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\n- Optional"}     }

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":": "}     }

event: content_block_stop
data: {"type":"content_block_stop","index":0           }

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":99}        }

event: message_stop
data: {"type":"message_stop"    }
```





### Mistral 

```python
import sys

if "google.colab" in sys.modules:
    from google.colab import auth

    auth.authenticate_user()
    
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
    
import ipywidgets as widgets
from IPython.display import display

dropdown_loc = widgets.Dropdown(
    options=available_regions,
    description="Select a location:",
    font_weight="bold",
    style={"description_width": "initial"},
)

dropdown_ver = widgets.Dropdown(
    options=available_versions,
    description="Select the model version (optional):",
    font_weight="bold",
    style={"description_width": "initial"},
)


def dropdown_loc_eventhandler(change):
    global LOCATION
    if change["type"] == "change" and change["name"] == "value":
        LOCATION = change.new
        print("Selected:", change.new)


def dropdown_ver_eventhandler(change):
    global MODEL_VERSION
    if change["type"] == "change" and change["name"] == "value":
        MODEL_VERSION = change.new
        print("Selected:", change.new)


LOCATION = dropdown_loc.value
dropdown_loc.observe(dropdown_loc_eventhandler, names="value")
display(dropdown_loc)

MODEL_VERSION = dropdown_ver.value
dropdown_ver.observe(dropdown_ver_eventhandler, names="value")
display(dropdown_ver)

PROJECT_ID = "micro-citadel-425019-a2"  # @param {type:"string"}
ENDPOINT = f"https://{LOCATION}-aiplatform.googleapis.com"
SELECTED_MODEL_VERSION = "" if MODEL_VERSION == "latest" else f"@{MODEL_VERSION}"

if not PROJECT_ID or PROJECT_ID == "[your-project-id]":
    raise ValueError("Please set your PROJECT_ID")
    
import json
import subprocess

import requests

PAYLOAD = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "who is the best French painter?"}],
    "max_tokens": 100,
    "top_p": 1.0,
    # "top_k": 10, # mistral不支持top_k参数
    "temperature": 1.0,
    "stream": True,
}

request = json.dumps(PAYLOAD)
!curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" {ENDPOINT}/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/mistralai/models/{MODEL}{SELECTED_MODEL_VERSION}:streamRawPredict -d '{request}'

"""
输出:
data: {"id":"033a46377c66422d9d625191286d2025","object":"chat.completion.chunk","created":1723619858,"model":"mistral-large","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null,"logprobs":null}]}

data: {"id":"033a46377c66422d9d625191286d2025","object":"chat.completion.chunk","created":1723619858,"model":"mistral-large","choices":[{"index":0,"delta":{"content":"Det"},"finish_reason":null,"logprobs":null}]}

(中间省略一部分模型输出)

data: {"id":"033a46377c66422d9d625191286d2025","object":"chat.completion.chunk","created":1723619858,"model":"mistral-large","choices":[{"index":0,"delta":{"content":"eting"},"finish_reason":null,"logprobs":null}]}

data: {"id":"033a46377c66422d9d625191286d2025","object":"chat.completion.chunk","created":1723619858,"model":"mistral-large","choices":[{"index":0,"delta":{"content":" effects"},"finish_reason":null,"logprobs":null}]}

data: {"id":"033a46377c66422d9d625191286d2025","object":"chat.completion.chunk","created":1723619858,"model":"mistral-large","choices":[{"index":0,"delta":{"content":" of"},"finish_reason":"length","logprobs":null}],"usage":{"prompt_tokens":10,"total_tokens":110,"completion_tokens":100}}

data: [DONE]
"""
```



#### mistral输出

```json
data: {"id":"033a46377c66422d9d625191286d2025","object":"chat.completion.chunk","created":1723619858,"model":"mistral-large","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null,"logprobs":null}]}

data: {"id":"033a46377c66422d9d625191286d2025","object":"chat.completion.chunk","created":1723619858,"model":"mistral-large","choices":[{"index":0,"delta":{"content":"Det"},"finish_reason":null,"logprobs":null}]}

(中间省略一部分模型输出)

data: {"id":"033a46377c66422d9d625191286d2025","object":"chat.completion.chunk","created":1723619858,"model":"mistral-large","choices":[{"index":0,"delta":{"content":"eting"},"finish_reason":null,"logprobs":null}]}

data: {"id":"033a46377c66422d9d625191286d2025","object":"chat.completion.chunk","created":1723619858,"model":"mistral-large","choices":[{"index":0,"delta":{"content":" effects"},"finish_reason":null,"logprobs":null}]}

data: {"id":"033a46377c66422d9d625191286d2025","object":"chat.completion.chunk","created":1723619858,"model":"mistral-large","choices":[{"index":0,"delta":{"content":" of"},"finish_reason":"length","logprobs":null}],"usage":{"prompt_tokens":10,"total_tokens":110,"completion_tokens":100}}

data: [DONE]
```



### llama3

```python
import json
import sys

if "google.colab" in sys.modules:

    from google.colab import auth

    auth.authenticate_user()

MODEL_LOCATION = "us-central1"
MODEL_ID = "meta/llama3-405b-instruct-maas"  # @param {type:"string"} ["meta/llama3-405b-instruct-maas"]
PROJECT_ID = "micro-citadel-425019-a2"
base_url=f"https://{MODEL_LOCATION}-aiplatform.googleapis.com/v1beta1/projects/{PROJECT_ID}/locations/{MODEL_LOCATION}/endpoints/openapi/chat/completions"

PAYLOAD = {
    "model": MODEL_ID,
    "messages": [{"role": "user", "content": "who is the best French painter?"}],
    "max_tokens": 100,
  	"top_p": 1.0,
  	"top_k": 10,
  	"temperature": 1.0,
    "stream": True,
}

request = json.dumps(PAYLOAD)

!curl -X POST -H "Authorization: Bearer $(gcloud auth print-access-token)" -H "Content-Type: application/json" {base_url} -d '{request}'

"""
输出:
data: {"choices":[{"delta":{"content":"Choosing the 'best' French painter is subjective and highly depends on personal taste and","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":" historical context. France has produced a multitude of artists who have significantly contributed to the","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":" development of art and painting. Here are a few standout artists across different periods who","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":" could be considered among the greatest, depending on your criteria:\n\n1. **Cla","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"ude Monet (1840-1926)**: A pioneer of the French","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":" Impressionist movement, Monet is famous for his landscapes, particularly his dep","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"ictions of light in","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"","role":"assistant"},"finish_reason":"stop","index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk","usage":{"completion_tokens":102,"prompt_tokens":7,"total_tokens":109}}

"""
```



#### llama3输出

```json
data: {"choices":[{"delta":{"content":"Choosing the 'best' French painter is subjective and highly depends on personal taste and","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":" historical context. France has produced a multitude of artists who have significantly contributed to the","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":" development of art and painting. Here are a few standout artists across different periods who","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":" could be considered among the greatest, depending on your criteria:\n\n1. **Cla","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"ude Monet (1840-1926)**: A pioneer of the French","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":" Impressionist movement, Monet is famous for his landscapes, particularly his dep","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"ictions of light in","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"","role":"assistant"},"finish_reason":"stop","index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk","usage":{"completion_tokens":102,"prompt_tokens":7,"total_tokens":109}}
```



### 输出格式汇总

#### Gemini 输出

publisher: google

```json
{'candidates': [{'content': {'role': 'model', 'parts': [{'text': '茴'}]}}]}
{'candidates': [{'content': {'role': 'model', 'parts': [{'text': '香豆的“茴”字只有一种写法。 😊 \n\n您'}]}, 'safetyRatings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.111328125, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.08251953}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.07470703, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.14453125}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.103515625, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.08251953}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.14160156, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.049560547}]}]}
{'candidates': [{'content': {'role': 'model', 'parts': [{'text': '是不是想起了鲁迅先生笔下的孔乙己？ \n'}]}, 'safetyRatings': [{'category': 'HARM_CATEGORY_HATE_SPEECH', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.13769531, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.09814453}, {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.061767578, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.15625}, {'category': 'HARM_CATEGORY_HARASSMENT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.10986328, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.16308594}, {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'probability': 'NEGLIGIBLE', 'probabilityScore': 0.16308594, 'severity': 'HARM_SEVERITY_NEGLIGIBLE', 'severityScore': 0.07714844}]}]}
{'candidates': [{'content': {'role': 'model', 'parts': [{'text': ''}]}, 'finishReason': 'STOP'}], 'usageMetadata': {'promptTokenCount': 40, 'candidatesTokenCount': 31, 'totalTokenCount': 71}}
```



#### anthropic输出

publisher: anthropic

```json
event: message_start
data: {"type":"message_start","message":{"id":"msg_vrtx_017o4feJdWfVviVPbMusHB1q","type":"message","role":"assistant","model":"claude-3-5-sonnet-20240620","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":15,"output_tokens":1}}    }

event: ping
data: {"type": "ping"}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}   }

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Here"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"'s"}       }

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" a classic banana brea"}  }

(中间省略一部分模型输出)

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" flour"}          }

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\n- Optional"}     }

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":": "}     }

event: content_block_stop
data: {"type":"content_block_stop","index":0           }

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"max_tokens","stop_sequence":null},"usage":{"output_tokens":99}        }

event: message_stop
data: {"type":"message_stop"    }
```



#### mistral输出

publisher: mistralai

```json
data: {"id":"033a46377c66422d9d625191286d2025","object":"chat.completion.chunk","created":1723619858,"model":"mistral-large","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null,"logprobs":null}]}

data: {"id":"033a46377c66422d9d625191286d2025","object":"chat.completion.chunk","created":1723619858,"model":"mistral-large","choices":[{"index":0,"delta":{"content":"Det"},"finish_reason":null,"logprobs":null}]}

(中间省略一部分模型输出)

data: {"id":"033a46377c66422d9d625191286d2025","object":"chat.completion.chunk","created":1723619858,"model":"mistral-large","choices":[{"index":0,"delta":{"content":"eting"},"finish_reason":null,"logprobs":null}]}

data: {"id":"033a46377c66422d9d625191286d2025","object":"chat.completion.chunk","created":1723619858,"model":"mistral-large","choices":[{"index":0,"delta":{"content":" effects"},"finish_reason":null,"logprobs":null}]}

data: {"id":"033a46377c66422d9d625191286d2025","object":"chat.completion.chunk","created":1723619858,"model":"mistral-large","choices":[{"index":0,"delta":{"content":" of"},"finish_reason":"length","logprobs":null}],"usage":{"prompt_tokens":10,"total_tokens":110,"completion_tokens":100}}

data: [DONE]
```



#### llama3输出

publisher: llama3

```json
data: {"choices":[{"delta":{"content":"Choosing the 'best' French painter is subjective and highly depends on personal taste and","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":" historical context. France has produced a multitude of artists who have significantly contributed to the","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":" development of art and painting. Here are a few standout artists across different periods who","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":" could be considered among the greatest, depending on your criteria:\n\n1. **Cla","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"ude Monet (1840-1926)**: A pioneer of the French","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":" Impressionist movement, Monet is famous for his landscapes, particularly his dep","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"ictions of light in","role":"assistant"},"index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk"}

data: {"choices":[{"delta":{"content":"","role":"assistant"},"finish_reason":"stop","index":0}],"model":"meta/llama3-405b-instruct-maas","object":"chat.completion.chunk","usage":{"completion_tokens":102,"prompt_tokens":7,"total_tokens":109}}
```





### 输入格式汇总

#### Gemini输入

publisher == "google"

```json
{
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
```



#### anthropic输入

publisher == "anthropic"

```json
{
    "anthropic_version": "vertex-2023-10-16", #这是固定值
    "messages": [{"role": "user", "content": "给我一份香蕉面包的食谱"}],
    "max_tokens": 100,
    "stream": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 10,
}
```



#### mistralai输入

publisher == "mistralai"

```json
{
    "model": "mistral-large",
    "messages": [{"role": "user", "content": "给我一份香蕉面包的食谱"}],
    "max_tokens": 100,
    "top_p": 1.0,
    # "top_k": 10, # mistral不支持top_k参数
    "temperature": 1.0,
    "stream": True,
}
```



#### openapi输入

publisher == "openapi"

```json
{
    "model": "meta/llama3-405b-instruct-maas",
    "messages": [{"role": "user", "content": "给我一份香蕉面包的食谱"}],
    "max_tokens": 100,
    "top_p": 1.0,
    "top_k": 10,
    "temperature": 1.0,
    "stream": True,
}
```

