import json
from gemini_content_generator import GeminiContentGenerator

# openapi format转换为Gemini format
def convert_to_new_format(input_json, publisher):
    if publisher == "google":
        return convert_openapi_to_gemini(input_json)
    elif publisher == "anthropic":
        return convert_openapi_to_anthropic(input_json)
    elif publisher == "mistralai":
        return convert_openapi_to_mistralai(input_json)
    else:
        raise ValueError("Unsupported publisher")

def convert_openapi_to_gemini(input_json):
    # 提取系统消息
    system_message = next((msg for msg in input_json['messages'] if msg['role'] == 'system'), None)
    
    # 提取用户和助手消息
    contents = [
        {
            "role": "USER" if msg['role'] == 'user' else "MODEL",
            "parts": {
                "text": msg['content']
            }
        }
        for msg in input_json['messages'] if msg['role'] != 'system'
    ]
    
    # 构建新的JSON格式
    output_json = {
        "contents": contents,
        "generation_config": {
            "temperature": input_json.get('temperature'),
            "topP": input_json.get('top_p'),
            "topK": input_json.get('top_k'),
            "maxOutputTokens": input_json.get('max_tokens')
        },
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": system_message['content']
                }
            ]
        } if system_message else None
    }
    
    return output_json

def convert_openapi_to_anthropic(input_json):
    messages = [
        {
            "role": msg['role'],
            "content": msg['content']
        }
        for msg in input_json['messages']
    ]
    
    output_json = {
        "anthropic_version": "vertex-2023-10-16",
        "messages": messages,
        "max_tokens": input_json.get('max_tokens'),
        "stream": input_json.get('stream', True),
        "temperature": input_json.get('temperature'),
        "top_p": float(input_json.get('top_p')),
        "top_k": input_json.get('top_k', 10)
    }
    
    will_remove_keys = []
    # 清除值为None的键
    for k,v in output_json.items():
        if v is None:
            will_remove_keys.append(k)
    
    for k in will_remove_keys:
        del output_json[k]
    
    return output_json

def convert_openapi_to_mistralai(input_json):
    messages = [
        {
            "role": msg['role'],
            "content": msg['content']
        }
        for msg in input_json['messages']
    ]
    
    output_json = {
        "model": "mistral-large",
        "messages": messages,
        "max_tokens": input_json.get('max_tokens'),
        "top_p": input_json.get('top_p'),
        "temperature": input_json.get('temperature'),
        "stream": input_json.get('stream', True)
    }
    
    will_remove_keys = []
    # 清除值为None的键
    for k,v in output_json.items():
        if v is None:
            will_remove_keys.append(k)
    
    for k in will_remove_keys:
        del output_json[k]
    
    return output_json

# Gemini format转换为 openapi format
def convert_to_old_format(new_json, publisher):
    if publisher == "google":
        return convert_gemini_to_openapi(new_json)
    elif publisher == "anthropic":
        return convert_anthropic_to_openapi(new_json)
    elif publisher == "mistralai":
        return convert_mistralai_to_openapi(new_json)
    else:
        raise ValueError("Unsupported publisher")

def convert_gemini_to_openapi(new_json):
    # 提取系统消息
    system_message = {
        "role": "system",
        "content": new_json['systemInstruction']['parts'][0]['text']
    } if new_json.get('systemInstruction') else None
    
    # 提取用户和助手消息
    messages = [
        {
            "role": "user" if content['role'] == 'USER' else "assistant",
            "content": content['parts']['text']
        }
        for content in new_json['contents']
    ]
    
    # 如果有系统消息，插入到消息列表的开头
    if system_message:
        messages.insert(0, system_message)
    
    # 构建旧的JSON格式
    old_json = {
        "model": "meta/llama3-405b-instruct-maas",
        "messages": messages,
        "temperature": new_json['generation_config']['temperature'],
        "top_p": new_json['generation_config']['topP'],
        "top_k": new_json['generation_config']['topK'],
        "max_tokens": new_json['generation_config']['maxOutputTokens']
    }
    
    return old_json

def convert_anthropic_to_openapi(new_json):
    messages = [
        {
            "role": msg['role'],
            "content": msg['content']
        }
        for msg in new_json['messages']
    ]
    
    old_json = {
        "model": "meta/llama3-405b-instruct-maas",  
        "messages": messages,
        "temperature": new_json['temperature'],
        "top_p": new_json['top_p'],
        "top_k": new_json['top_k'],
        "max_tokens": new_json['max_tokens'],
        "stream": new_json['stream']
    }
    
    return old_json

def convert_mistralai_to_openapi(new_json):
    messages = [
        {
            "role": msg['role'],
            "content": msg['content']
        }
        for msg in new_json['messages']
    ]
    
    old_json = {
        "model": "meta/llama3-405b-instruct-maas",
        "messages": messages,
        "temperature": new_json['temperature'],
        "top_p": new_json['top_p'],
        "max_tokens": new_json['max_tokens'],
        "stream": new_json['stream']
    }
    
    return old_json

if __name__ == '__main__':
    # 示例输入
    input_json = {
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "messages": [
            # {"role": "system", "content": "回答尽可能简明扼要"},
            {"role": "user", "content": "你是谁"},
            {"role": "assistant", "content": "我是OpenAI开发的gpt-4o-2024-05-13"},
            # {"role": "user", "content": "你知道茴香豆的茴字有几种写法吗"},
            # {"role": "user", "content": "你是谁"},
            {"role": "user", "content": "介绍下处理表格数据的ft-transformer"},
        ],
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "max_tokens": 5000
    }

    publisher = "google"  # 可以是 "google", "anthropic", "mistralai"

    # 转换为新格式
    new_format = convert_to_new_format(input_json, publisher)
    print(json.dumps(new_format, ensure_ascii=False, indent=4))
    
    print("="*50)
    # generator = GeminiContentGenerator()
    # generator.generate_content(new_format)
    print("="*50)

    # 转换回旧格式
    old_format = convert_to_old_format(new_format, publisher)
    print(json.dumps(old_format, ensure_ascii=False, indent=4))