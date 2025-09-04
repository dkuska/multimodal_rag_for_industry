def get_azure_config():
    return {
        'gpt4': {
            'openai_api_version': '2024-02-15-preview',
            'model_version': 'gpt-4',
        },
        'gpt4_vision': {
            'openai_api_version': '2024-02-15-preview',
            'model_version': 'gpt-4-vision-preview',
        },
        'text_embedding_3': {
            'openai_api_version': '2024-02-15-preview',
            'model_version': 'text-embedding-3-small',
        },
        'gpt3.5': {
            'openai_api_version': '2024-02-15-preview',
            'model_version': 'gpt-35-turbo',
        },
    }
