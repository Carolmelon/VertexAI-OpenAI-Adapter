from flask import Flask, request, Response, jsonify
from flask_cors import CORS
import json
import time
import uuid
import random
import requests
import sqlite3
import hashlib


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

def get_sha512_hash(input_string):
    """
    计算输入字符串的 SHA-512 哈希值。

    参数:
    input_string (str): 要计算哈希值的字符串。

    返回:
    str: 输入字符串的 SHA-512 哈希值。
    """
    # 创建 SHA-512 哈希对象
    sha512_hash = hashlib.sha512()
    
    # 更新哈希对象
    sha512_hash.update(input_string.encode('utf-8'))
    
    # 获取哈希值的十六进制表示
    sha512_digest = sha512_hash.hexdigest()
    
    return sha512_digest

def flatten_user_message(user_messages):
    result = ""
    for uttr in user_messages:
        result += f'<role name="{uttr["role"]}">'
        result += ":\n"
        result += uttr['content']
        result += '\n\n'
    return result

def init_db():
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  session_id TEXT,
                  request_id TEXT,
                  user_message TEXT,
                  model_response TEXT,
                  model_name TEXT,
                  user_message_flatten TEXT,
                  user_message_sha512 TEXT,
                  timestamp DATETIME DEFAULT (datetime('now', '+8 hours')))''')
    conn.commit()
    conn.close()

init_db()

def insert_chat_history(user_id, session_id, request_id, user_message, model_response, model_name):
    jsonified_user_message = json.dumps(user_message, ensure_ascii=False) # 这个不只是messages,还包含了temperature/top_k/top_p/stream/max_tokens/presence_penalty/frequency_penalty/model等
    conn = sqlite3.connect('chat_history.db')
    c = conn.cursor()
    c.execute(
        "INSERT INTO chat_history (user_id, session_id, request_id, user_message, model_response, model_name, user_message_flatten, user_message_sha512) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            user_id, session_id, request_id, 
            jsonified_user_message,
            model_response, 
            model_name,
            flatten_user_message(user_message['messages']),
            get_sha512_hash(jsonified_user_message),
        )
    )
    conn.commit()
    conn.close()

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
    # nextchat不会传入这个字段, nextchat的最大tokens数量是在前端控制的
    if not data.get("max_tokens"):
        data['max_tokens'] = 4000
    messages = data.get('messages', [])
    model_name = models_dict[current_generator_name]

    print(f"\033[35mmodel_name: {model_name}\033[0m")

    stream_tokens = data.get("stream_tokens", False) or data.get("stream", False)
    # 强行把stream设为True，让所有模型都走stream=True
    if 'stream_tokens' in data:
        data['stream_tokens'] = True
    elif 'stream' in data:
        data['stream'] = True
    else:
        data['stream'] = True

    if current_generator_name != "openapi":
        messages = convert_to_new_format(data, publisher=current_generator_name)
    else:
        messages = data.copy()
        messages['model'] = "meta/llama3-405b-instruct-maas" # data['model'] == "gpt-3.5-turbo"
        
    print("="*100)
    print(json.dumps(data, ensure_ascii=False, indent=4))
    print("-" * 100)
    print(json.dumps(messages, ensure_ascii=False, indent=4))

    user_id = data.get('user_id', 'anonymous')
    session_id = data.get('session_id', str(uuid.uuid4()))
    request_id = str(uuid.uuid4())

    # 非流式返回
    print(f"stream_tokens: {stream_tokens}")
    if not stream_tokens:
        # 这里实际请求还是走的stream，不过用list收集了迭代器的所有内容
        response = generator.generate_content_stream(messages)
        response_arr = ''.join(list(response))

        # 插入聊天历史
        insert_chat_history(user_id, session_id, request_id, data, response_arr, model_name)

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
        full_response = ""
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
            full_response += content

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

        # 插入聊天历史
        insert_chat_history(user_id, session_id, request_id, data, full_response, model_name)

    return Response(generate(), content_type='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9876)
