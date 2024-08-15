from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import json
import time
import uuid
import random
import requests

# from gemini_content_generator import GeminiContentGenerator
from gemini_content_generator import GeminiContentGenerator, all_generators
from gemini_messages_converter import convert_to_new_format

app = Flask(__name__)
CORS(app, resources={r"/v1/chat/completions": {"origins": "https://app.nextchat.dev"}})

models = json.load(open("/Users/zebralee/Desktop/codings/gemini-convert/all_models.json"))

models_dict = {
    "google": "gemini-1.5-pro",
    "anthropic": "claude-3-5-sonnet@20240620",
    "mistralai": "mistral-large",
    "openapi": "meta/llama3-405b-instruct-maas"
}

current_generator_name = 'anthropic'  # 有四种取值: models_dict.keys()

generator = all_generators[current_generator_name]

print(generator)
print(f"\033[32mcurrent_generator_name: {current_generator_name}\033[0m")

@app.route('/')
def index():
    """
    Render the index page.

    Returns:
        str: The content of the index.html file.
    """
    return open('static/index.html').read()

@app.route('/set_generator', methods=['POST'])
def set_generator():
    """
    Set the current generator based on the request.

    Returns:
        flask.Response: JSON response indicating success or failure.
    """
    global current_generator_name
    global generator
    new_generator = request.json['generator']
    if new_generator in models_dict.keys():
        current_generator_name = new_generator
        generator = all_generators[current_generator_name]
        print(f"\033[32mcurrent_generator_name: {current_generator_name}\033[0m")
        print(f"\033[36mcurrent_model_name: {models_dict[current_generator_name]}\033[0m")
        return jsonify({"success": True, "current_generator": current_generator_name})
    return jsonify({"success": False, "error": "Invalid generator name"})

@app.route('/v1/models')
def get_models():
    """
    Get the list of models.

    Returns:
        flask.Response: JSON response containing the models.
    """
    return models

@app.route('/v1/chat/completions', methods=['POST', 'OPTIONS'])
def chat_completions():
    """
    Handle chat completions.

    Returns:
        flask.Response: JSON response containing the chat completions.
    """
    print(f"\033[31mcurrent_generator_name: {current_generator_name}\033[0m")

    if request.method == 'OPTIONS':
        response = Response()
        response.headers['Access-Control-Allow-Origin'] = 'https://app.nextchat.dev'
        response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        return response

    data = request.json
    if not data.get("max_tokens"):
        data['max_tokens'] = 4000
    messages = data.get('messages', [])
    # model_name = data.get('model', 'gemini-1.5-pro')
    model_name = models_dict[current_generator_name]
    
    print(f"\033[35mmodel_name: {model_name}\033[0m")
    
    stream_tokens = data.get("stream_tokens", False) or data.get("stream", False)

    if current_generator_name != "openapi":
        messages = convert_to_new_format(data, publisher=current_generator_name)
    else:
        messages = data.copy()
        messages['model'] = "meta/llama3-405b-instruct-maas" # data['model'] == "gpt-3.5-turbo"
        
    print("="*100)
    print(json.dumps(data, ensure_ascii=False, indent=4))
    print("-" * 100)
    print(json.dumps(messages, ensure_ascii=False, indent=4))
    
    # 非流式返回
    
    print(f"stream_tokens: {stream_tokens}")
    if not stream_tokens:
        response = generator.generate_content_stream(messages)
        response_arr = ''.join(list(response))

        response_id = str(uuid.uuid4())
        created_time = int(time.time())

        return_data = {
            'id': response_id,
            'object': 'chat.completion',
            'created': created_time,
            'model': model_name,
            'prompt': [],
            'choices': [
                {
                    'finish_reason': 'stop', 'seed': int(time.time() * 1000),
                    'logprobs': None, 'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': response_arr
                    }
                }
            ],
            'usage': {'prompt_tokens': 0, 'completion_tokens': len(response_arr), 'total_tokens': 0}
        }
        return return_data
    
    # 流式返回
    
    while True:
        try:
            response_iterator = generator.generate_content_stream(messages)
        except requests.exceptions.HTTPError as e:
            print(e)
        else:
            break

    response_id = str(uuid.uuid4())
    created_time = int(time.time())

    def generate():
        total_tokens = 0
        last_content = None
        for index, content in enumerate(response_iterator):
            total_tokens += 1
            if last_content is not None:
                chunk = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": created_time,
                    "choices": [{
                        "index": 0,
                        "text": last_content,
                        "logprobs": None,
                        "finish_reason": None,
                        "seed": None,
                        "delta": {
                            "token_id": random.randint(1, 100),
                            "role": "assistant",
                            "content": last_content,
                            "tool_calls": None
                        }
                    }],
                    "model": model_name,
                    "usage": None
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            last_content = content

        # 最后一个块，包含 finish_reason 和 usage 信息
        if last_content is not None:
            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "choices": [{
                    "index": 0,
                    "text": last_content,
                    "logprobs": None,
                    "finish_reason": "length",
                    "seed": int(time.time() * 1000),
                    "delta": {
                        "token_id": random.randint(1, 100),
                        "role": "assistant",
                        "content": last_content,
                        "tool_calls": None
                    }
                }],
                "model": model_name,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": total_tokens,
                    "total_tokens": 0,
                }
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return Response(generate(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9876)
