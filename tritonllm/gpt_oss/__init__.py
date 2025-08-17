try:
    from gpt_oss.tokenizer import get_tokenizer
except ImportError:
    try:
        # Editable install
        from tritonllm.utils import init_env
        init_env()
    except ImportError:
        pass
