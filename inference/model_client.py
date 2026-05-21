from openai import OpenAI
import os
from typing import Optional


class ModelClient:

    PROVIDER_API_CONFIGS = {
        'openrouter': {
            'api_key_env': 'OPENROUTER_API_KEY',
            'base_url_env': 'OPENROUTER_BASE_URL',
            'base_url_default': 'https://openrouter.ai/api/v1',
        },
        'dashscope': {
            'api_key_env': 'DASHSCOPE_API_KEY',
            'base_url_env': 'DASHSCOPE_BASE_URL',
            'base_url_default': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
        },
        'openai': {
            'api_key_env': 'OPENAI_API_KEY',
            'base_url_env': 'OPENAI_API_BASE',
            'base_url_default': 'https://api.openai.com/v1',
        },
    }

    # Stage 默认 provider 映射
    STAGE_DEFAULT_PROVIDER = {
        'research': None,  
        'rephrase': None, 
        'summary': None,  
        'embedding': 'dashscope',  
    }

    def __init__(self, stage: str):
        """
        stage: research | rephrase | summary | embedding
        """
        self.stage = stage

        default_provider = self.STAGE_DEFAULT_PROVIDER.get(stage)
        if default_provider is None:
            default_provider = os.getenv('PROVIDER', 'openrouter')
        self.provider = os.getenv(f'{stage.upper()}_PROVIDER', default_provider)

        # 获取对应的 api_key 和 base_url
        config = self.PROVIDER_API_CONFIGS.get(self.provider, self.PROVIDER_API_CONFIGS['openrouter'])
        print(f"[DEBUG] Final provider for stage {stage} is {self.provider}")
        self.api_key = os.getenv(config['api_key_env'], '')
        self.base_url = os.getenv(config['base_url_env'], config['base_url_default'])
        self.model = os.getenv(f'{stage.upper()}_MODEL', '')

    def get_client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=600.0)

    def call(self, messages, **kwargs):
        client = self.get_client()
        return client.chat.completions.create(model=self.model, messages=messages, **kwargs)

    def __repr__(self):
        return f"ModelClient(stage={self.stage}, provider={self.provider}, model={self.model}, base_url={self.base_url})"
