import json

class TruncParser:
    def __init__(self, publisher):
        self.publisher = publisher
        self.last_incomplete_line = ""
        self.content = ""

    def parse_trunc(self, trunc):
        # 将上次的不完整行与新的trunc合并
        trunc = self.last_incomplete_line + trunc
        lines = trunc.split('\n')
        
        # 保存最后一行，可能是不完整的
        # self.last_incomplete_line = lines[-1]
        # lines = lines[:-1]

        new_content = ""
        for line in lines:
            new_content += self._process_line(line)

        self.content += new_content
        return new_content

    def _process_line(self, line):
        if self.publisher == "google":
            try:
                data = json.loads(line)
                if 'candidates' in data and data['candidates']:
                    return data['candidates'][0]['content']['parts'][0]['text']
            except json.JSONDecodeError:
                pass

        elif self.publisher == "anthropic":
            # if line.startswith("data: "):
            #     try:
            #         data = json.loads(line[6:])
            #         if data['type'] == 'content_block_delta':
            #             return data['delta']['text']
            #     except json.JSONDecodeError:
            #         pass
            try:
                data = json.loads(line)
                if data['type'] == 'content_block_delta':
                    return data['delta']['text']
            except json.JSONDecodeError:
                pass

        elif self.publisher in ["mistralai", "llama3"]:
            # if line.startswith("data: "):
            #     try:
            #         data = json.loads(line[6:])
            #         if 'choices' in data and data['choices']:
            #             delta = data['choices'][0].get('delta', {})
            #             if 'content' in delta:
            #                 return delta['content']
            #     except json.JSONDecodeError:
            #         pass
            try:
                data = json.loads(line)
                if 'choices' in data and data['choices']:
                    delta = data['choices'][0].get('delta', {})
                    if 'content' in delta:
                        return delta['content']
            except json.JSONDecodeError:
                pass

        return ""

    def get_full_content(self):
        return self.content.strip()

# 使用示例
def parse_trunc(trunc, publisher):
    parser = TruncParser(publisher)
    return parser.parse_trunc(trunc)

# 测试用例
def test_parse_trunc():
    # 测试用例尚未更新
    # 一般来说，需要解析的数据里面不包含"data: "
    # "data: "通常在SSE情况下，会被http客户端去掉，因此测试用例无需包含"data: "
    
    
    # Google 测试用例
    google_parser = TruncParser("google")
    assert google_parser.parse_trunc('{"candidates": [{"content": {"role": "model", "parts": [{"text": "茴"}]}}]}\n') == "茴"
    assert google_parser.parse_trunc('{"candidates": [{"content": {"role": "model", "parts": [{"text": "香豆的"}]}}]}\n') == "香豆的"
    # assert google_parser.parse_trunc('{"candidates": [{"content": {"role": "model", "parts": [{"text": ""茴"字"}]}}]}\n') == '"茴"字'
    print("""google_parser.parse_trunc('{"candidates": [{"content": {"role": "model", "parts": [{"text": ""茴"字"}]}}]}\n'):\n""", google_parser.parse_trunc('{"candidates": [{"content": {"role": "model", "parts": [{"text": "\\"茴\\"字"}]}}]}\n'))
    assert google_parser.get_full_content() == '茴香豆的"茴"字'

    # Anthropic 测试用例
    anthropic_parser = TruncParser("anthropic")
    assert anthropic_parser.parse_trunc('data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Here"}}\n') == "Here"
    assert anthropic_parser.parse_trunc('data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\'s"}}\n') == "\'s"
    assert anthropic_parser.parse_trunc('data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" a "}}\n') == " a "
    assert anthropic_parser.get_full_content() == "Here's a"

    # Mistral AI 测试用例
    mistral_parser = TruncParser("mistralai")
    assert mistral_parser.parse_trunc('data: {"choices":[{"delta":{"content":"Det"},"index":0}]}\n') == "Det"
    assert mistral_parser.parse_trunc('data: {"choices":[{"delta":{"content":"ecting"},"index":0}]}\n') == "ecting"
    assert mistral_parser.parse_trunc('data: {"choices":[{"delta":{"content":" effects"},"index":0}]}\n') == " effects"
    assert mistral_parser.get_full_content() == "Detecting effects"

    # LLaMA-3 测试用例
    llama3_parser = TruncParser("llama3")
    assert llama3_parser.parse_trunc('data: {"choices":[{"delta":{"content":"Choosing "},"index":0}]}\n') == "Choosing "
    assert llama3_parser.parse_trunc('data: {"choices":[{"delta":{"content":"the \'best\'"},"index":0}]}\n') == "the \'best\'"
    assert llama3_parser.parse_trunc('data: {"choices":[{"delta":{"content":" French"},"index":0}]}\n') == " French"
    assert llama3_parser.get_full_content() == "Choosing the 'best' French"

    print("All test cases passed!")

all_trunc_parser = {
    "google": TruncParser("google"),
    "anthropic": TruncParser("anthropic"),
    "mistralai": TruncParser("mistralai"),
    "openapi": TruncParser("llama3"),
}

if __name__ == '__main__':
    # 运行测试
    test_parse_trunc()