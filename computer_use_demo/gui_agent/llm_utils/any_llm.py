import os
import logging
import base64
import requests
from PIL import Image
from io import BytesIO
import mimetypes
from computer_use_demo.gui_agent.llm_utils.llm_utils import is_image_path, encode_image

def upload_image_to_gemini(image_path: str, gemini_upload_url: str) -> str:
        """Upload image to Gemini and get file URI (for Gemini platform only)"""
        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            
            # 检测MIME类型
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = 'application/octet-stream'
            
            headers = {
                "Content-Type": mime_type,
                "X-Goog-Upload-Protocol": "raw"
            }
            
            response = requests.post(gemini_upload_url, data=image_data, headers=headers, timeout=60)
            if response.status_code == 200:
                return response.json().get("fileUri")
            else:
                raise Exception(f"Gemini image upload failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Gemini image upload failed: {str(e)}")
            raise

def run_llm(
    messages: list, 
    system: str, 
    llm: str, 
    url: str, 
    max_tokens=256, 
    temperature=0.7, 
    do_sample=True,
    platform: str = "OpenAI"  # Supports "OpenAI", "Anthropic", "Gemini"
):
    """Send chat completion request to SSH remote server with platform-specific message formats"""

    try:
        # Prepare base message list
        final_messages = []
        
        # Add system message if provided
        if system:
            if platform == "Anthropic":
                # Anthropic: System message is part of the messages list with role "system"
                final_messages.append({
                    "role": "system",
                    "content": system
                })
            elif platform == "Gemini":
                # Gemini: System message is handled via the "system_instruction" field in the request
                system_instruction = system
            else:  # OpenAI (default)
                # OpenAI: System message is part of the messages list with role "system"
                final_messages.append({
                    "role": "system",
                    "content": system
                })

        # Process user messages
        if isinstance(messages, list):
            for item in messages:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", [])
                    
                    if platform == "Gemini":
                        # Gemini format: uses "parts" instead of "content", and requires file URIs for images
                        parts = []
                        for cnt in content:
                            if isinstance(cnt, str):
                                if is_image_path(cnt):
                                    # Upload image to Gemini and get file URI
                                    # Note: You need to provide the Gemini upload URL (usually different from the chat URL)
                                    gemini_upload_url = url.replace("/v1/chat/completions", "/v1/files/upload")
                                    file_uri = upload_image_to_gemini(cnt, gemini_upload_url)
                                    parts.append({
                                        "file_data": {
                                            "mime_type": mimetypes.guess_type(cnt)[0] or "image/jpeg",
                                            "file_uri": file_uri
                                        }
                                    })
                                else:
                                    parts.append({"text": cnt})
                        final_messages.append({
                            "role": role,
                            "parts": parts
                        })
                    
                    elif platform == "Anthropic":
                        # Anthropic format: content is an array of text and image objects
                        anthropic_content = []
                        for cnt in content:
                            if isinstance(cnt, str):
                                if is_image_path(cnt):
                                    base64_str, mime_type = encode_image(cnt)
                                    anthropic_content.append({
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": mime_type,
                                            "data": base64_str
                                        }
                                    })
                                else:
                                    anthropic_content.append({
                                        "type": "text",
                                        "text": cnt
                                    })
                        final_messages.append({
                            "role": role,
                            "content": anthropic_content
                        })
                    
                    else:  # OpenAI (default) and QwenVL
                        # OpenAI format: content is an array of text and image_url objects
                        openai_content = []
                        for cnt in content:
                            if isinstance(cnt, str):
                                if is_image_path(cnt):
                                    base64_str, mime_type = encode_image(cnt)
                                    openai_content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:{mime_type};base64,{base64_str}"
                                        }
                                    })
                                else:
                                    openai_content.append({
                                        "type": "text",
                                        "text": cnt
                                    })
                        final_messages.append({
                            "role": role,
                            "content": openai_content
                        })
                else:  # str
                    if platform == "Gemini":
                        final_messages.append({
                            "role": "user",
                            "parts": [{"text": item}]
                        })
                    elif platform == "Anthropic":
                        final_messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": item}]
                        })
                    else:  # OpenAI
                        final_messages.append({
                            "role": "user",
                            "content": [{"type": "text", "text": item}]
                        })
        elif isinstance(messages, str):
            if platform == "Gemini":
                final_messages.append({
                    "role": "user",
                    "parts": [{"text": messages}]
                })
            elif platform == "Anthropic":
                final_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": messages}]
                })
            else:  # OpenAI
                final_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": messages}]
                })

        # Prepare request data based on platform
        data = {
            "model": llm,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "do_sample": do_sample
        }
        
        if platform == "Gemini":
            # Gemini uses "contents" instead of "messages" and has "system_instruction"
            data["contents"] = final_messages
            if system:
                data["system_instruction"] = system
        else:
            # OpenAI and Anthropic use "messages"
            data["messages"] = final_messages
        
        print(f"[ssh] Sending chat completion request to model: {llm} (platform: {platform})")
        print(f"[ssh] Sending messages:", final_messages)
        
        # Send request
        response = requests.post(
            url,
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        result = response.json()
        
        if response.status_code == 200:
            if platform == "Gemini":
                # Gemini returns content in "candidates[0].content.parts[0].text"
                content = result['candidates'][0]['content']['parts'][0]['text']
                token_usage = int(result.get('usage', {}).get('total_tokens', 0))
            else:
                # OpenAI and Anthropic return content in "choices[0].message.content"
                content = result['choices'][0]['message']['content']
                token_usage = int(result['usage']['total_tokens'])
            
            print(f"[ssh] Generation successful: {content}")
            return content, token_usage
        else:
            print(f"[ssh] Request failed: {result}")
            raise Exception(f"API request failed: {result}")
            
    except Exception as e:
        print(f"[ssh] Chat completion request failed: {str(e)}")
        raise

# Helper function to check if a string is an image path
def is_image_path(path: str) -> bool:
    """Check if a string is a valid image path based on extension"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    return any(path.lower().endswith(ext) for ext in image_extensions)


# def run_oai_interleaved(messages: list, system: str, llm: str, api_key: str,
#                         max_tokens=256, temperature=0):

#     api_key = api_key or os.environ.get("OPENAI_API_KEY")
#     if not api_key:
#         raise ValueError("OPENAI_API_KEY is not set")

#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": f"Bearer {api_key}"
#     }

#     # OpenAI 官方 Chat Completions URL
#     oai_url = "https://api.openai.com"

#     # 调用通用 LLM 接口即可
#     return run_llm_interleaved(
#         messages=messages,
#         system=system,
#         llm=llm,
#         url=oai_url,
#         max_tokens=max_tokens,
#         temperature=temperature,
#         do_sample=False,         # OpenAI 兼容
#         headers=headers
#     )


if __name__ == "__main__":
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    
    # text, token_usage = run_oai_interleaved(
    #     messages= [{"content": [
    #                     "What is in the screenshot?",   
    #                     "./tmp/outputs/screenshot_0b04acbb783d4706bc93873d17ba8c05.png"],
    #                 "role": "user"
    #                 }],
    #     llm="gpt-4o-mini",
    #     system="You are a helpful assistant",
    #     api_key=api_key,
    #     max_tokens=256,
    #     temperature=0)
    
    # print(text, token_usage)
    # text, token_usage = run_llm_interleaved(
    #     messages= [{"content": [
    #                     "What is in the screenshot?",   
    #                     "tmp/outputs/screenshot_5a26d36c59e84272ab58c1b34493d40d.png"],
    #                 "role": "user"
    #                 }],
    #     llm="Qwen2.5-VL-7B-Instruct",
    #     ssh_host="10.245.92.68",
    #     ssh_port=9192,
    #     max_tokens=256,
    #     temperature=0.7
    # )
    # print(text, token_usage)
    # There is an introduction describing the Calyx... 36986
