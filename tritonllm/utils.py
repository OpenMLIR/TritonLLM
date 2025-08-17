import urllib.request
import os

def open_url(url):
    user_agent = 'Mozilla/5.0 (X11; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0'
    headers = {
        'User-Agent': user_agent,
    }
    request = urllib.request.Request(url, None, headers)
    # Set timeout to 300 seconds to prevent the request from hanging forever.
    return urllib.request.urlopen(request, timeout=300)


def save_file_to_triton_llm_bin(triton_llm_bin):
    os.makedirs(triton_llm_bin, exist_ok=True)
    url = "https://tritonllm.top/down/o200k_base.tiktoken"
    save_as = os.path.join(triton_llm_bin, "fb374d419588a4632f3f557e76b4b70aebbca790")
    if not os.path.exists(save_as):
        with open_url(url) as response:
            content = response.read()

        with open(save_as, "wb") as f:
            f.write(content)
