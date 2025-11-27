"""
Entrypoint for Gradio, see https://gradio.app/
"""

import platform
import os
import json
from datetime import datetime
from functools import partial
from typing import cast, Dict
from typing_extensions import TypedDict
from typing import List, Any
import requests

import gradio as gr
from anthropic import APIResponse
from anthropic.types import TextBlock
from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock
from anthropic.types.tool_use_block import ToolUseBlock

from screeninfo import get_monitors
from computer_use_demo.tools.logger import logger, truncate_string
logger.info("Starting the gradio app")

screens = get_monitors()
logger.info(f"Found {len(screens)} screens")

from computer_use_demo.loop import APIProvider, sampling_loop_sync

from computer_use_demo.tools import ToolResult
from computer_use_demo.tools.computer import get_screen_details
SCREEN_NAMES, SELECTED_SCREEN_INDEX = get_screen_details()

API_CONFIG_PATH = './api_config.json'

AVAILABLE_PLATFORMS = ['OpenAI', 'Anthropic', 'Gemini', 'DashScope', 'Local']

WARNING_TEXT = "⚠️ Security Alert: Do not provide access to sensitive accounts or data, as malicious web content can hijack Agent's behavior. Keep monitor on the Agent's actions."

DEFAULT_PLANNER_PROVIDER: APIProvider = {
    "platform": "OpenAI",
    "key_name": "",
    "use_requests": True,
    "url": "",
    "model_url": "",
    "avail_models": []
}

DEFAULT_PLANNER_PROVIDER_LOCAL: APIProvider = {
    "platform": "OpenAI",
    "key_name": "",
    "use_requests": True,
    "url": "0.0.0.0:6666",
    "model_url": "",
    "avail_models": []
}

DEFAULT_ACTOR_PROVIDER: APIProvider = {
    "platform": "Local",
    "avail_models": ["ShowUI"],
    "key_name": "",
    "use_requests": True,
    "url": "",
    "model_url": "",
}

def list_available_models(url:str):
    """Query remote SSH server for available models"""
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if resp.status_code == 200:
            models = [m["id"] for m in data.get("data", [])]
            print("[ssh] Available models:", models)
            return models
        else:
            print(f"[ssh] Failed to list models: {data}")
            raise Exception(data)
    except Exception as e:
        print("[ssh] Error querying model list:", e)
        return []

def parse_api_provider(api_provider_name: str, api_provider_config: dict):
    assert 'url' in api_provider_config.keys(), f'Config of {api_provider_name} must have the api url!'
    # 👇 Initialize url (add http:// if missing)
    raw_url = api_provider_config['url']
    if not (raw_url.startswith('http://') or raw_url.startswith('https://')):
        raw_url = "http://" + raw_url
    api_provider_config['url'] = raw_url  # 👈 重要：更新原始 url！
    # Deal with API key
    if 'key' in api_provider_config:
        os.environ[api_provider_config['key_name']] = api_provider_config['key']
        api_provider_config.pop('key')
    # Set model_url
    if 'model_url' not in api_provider_config:
        api_provider_config['model_url'] = f"{api_provider_config['url']}/v1/models"
    # Check model list with model_url
    api_provider_config['avail_models'] = list_available_models(api_provider_config['model_url'])
    # Set platform
    if 'platform' in api_provider_config:
        assert api_provider_config['platform'] in AVAILABLE_PLATFORMS
    else:
        api_provider_config['platform'] = AVAILABLE_PLATFORMS[0]  # OpenAI
    if 'use_requests' not in api_provider_config:
        api_provider_config['use_requests'] = True
    else:
        api_provider_config['use_requests'] = bool(api_provider_config['use_requests'])
    provider = APIProvider(**api_provider_config)
    return provider

def parse_api_provider_all(api_config_path=API_CONFIG_PATH):
    with open(api_config_path) as f:
        data = json.load(f)
    provider_dict = {}
    for key, item in data.items():
        provider_dict[key] = parse_api_provider(key, item)
    return provider_dict

API_PROVIDERS = parse_api_provider_all(API_CONFIG_PATH)

API_NAMES = list(API_PROVIDERS.keys()); API_NAMES.append("Custom")

class State(TypedDict, total=False):
    messages: List[dict] = []
    planner_provider: APIProvider = APIProvider(platform='OpenAI')
    actor_provider: APIProvider = APIProvider(platform='Local', avail_models=['ShowUI'])
    planner_model: str
    actor_model: str
    openai_api_key: str
    anthropic_api_key: str
    qwen_api_key: str
    ui_tars_url: str
    planner_api_key: str
    planner_url: str
    available_models: List[str]
    selected_screen: int
    auth_validated: bool
    responses: Dict[str, Any]
    tools: Dict[str, Any]
    only_n_most_recent_images: int
    custom_system_prompt: str
    hide_images: bool
    chatbot_messages: List[Any]
    showui_config: str
    max_pixels: int
    awq_4bit: bool

def setup_state(state: State) -> State:
    # =============== 基础字段 ===============
    state.setdefault("messages", [])
    state.setdefault("planner_model", "gpt-4o")
    state.setdefault("actor_model", "ShowUI")

    # Provider 结构
    state.setdefault("planner_provider", DEFAULT_PLANNER_PROVIDER.copy())
    state.setdefault("actor_provider",   DEFAULT_ACTOR_PROVIDER.copy())

    # =============== 环境变量 API keys ===============
    state.setdefault("openai_api_key",    os.getenv("OPENAI_API_KEY", ""))
    state.setdefault("anthropic_api_key", os.getenv("ANTHROPIC_API_KEY", ""))
    state.setdefault("qwen_api_key",      os.getenv("QWEN_API_KEY", ""))
    state.setdefault("ui_tars_url", "")

    # =============== 其他状态字段 ===============
    state.setdefault("available_models", [])
    state.setdefault("responses", {})
    state.setdefault("tools", {})
    state.setdefault("only_n_most_recent_images", 10)
    state.setdefault("hide_images", False)
    state.setdefault("chatbot_messages", [])
    state.setdefault("showui_config", "Default")
    state.setdefault("max_pixels", 1344)
    state.setdefault("awq_4bit", False)

    # =============== system prompt (加入操作系统信息) ===============
    if "custom_system_prompt" not in state:
        os_name = platform.system()
        device_os_name = (
            "Windows" if os_name == "Windows"
            else "Mac" if os_name == "Darwin"
            else "Linux"
        )
        state["custom_system_prompt"] = f"\n\nNOTE: you are operating a {device_os_name} machine"

    # =============== screen 选择 ===============
    state.setdefault("selected_screen", SELECTED_SCREEN_INDEX if SCREEN_NAMES else 0)

    # =============== auth ===============
    state.setdefault("auth_validated", False)

    return state

async def main(state):
    """Render loop for Gradio"""
    setup_state(state)
    return "Setup completed"

def _api_response_callback(response: APIResponse[BetaMessage], response_state: dict):
    response_id = datetime.now().isoformat()
    response_state[response_id] = response

def _tool_output_callback(tool_output: ToolResult, tool_id: str, tool_state: dict):
    tool_state[tool_id] = tool_output

def chatbot_output_callback(message, chatbot_state, hide_images=False, sender="bot"):
    
    def _render_message(message: str | BetaTextBlock | BetaToolUseBlock | ToolResult, hide_images=False):
    
        logger.info(f"_render_message: {str(message)[:100]}")

        if isinstance(message, str):
            return message
        
        is_tool_result = not isinstance(message, str) and (
            isinstance(message, ToolResult)
            or message.__class__.__name__ == "ToolResult"
            or message.__class__.__name__ == "CLIResult"
        )
        if not message or (
            is_tool_result
            and hide_images
            and not hasattr(message, "error")
            and not hasattr(message, "output")
        ):  # return None if hide_images is True
            return
        # render tool result
        if is_tool_result:
            message = cast(ToolResult, message)
            if message.output:
                return message.output
            if message.error:
                return f"Error: {message.error}"
            if message.base64_image and not hide_images:
                # somehow can't display via gr.Image
                # image_data = base64.b64decode(message.base64_image)
                # return gr.Image(value=Image.open(io.BytesIO(image_data)))
                return f'<img src="data:image/png;base64,{message.base64_image}">'

        elif isinstance(message, BetaTextBlock) or isinstance(message, TextBlock):
            return message.text
        elif isinstance(message, BetaToolUseBlock) or isinstance(message, ToolUseBlock):
            return f"Tool Use: {message.name}\nInput: {message.input}"
        else:
            return message

    # processing Anthropic messages
    message = _render_message(message, hide_images)
    
    if sender == "bot":
        chatbot_state.append((None, message))
    else:
        chatbot_state.append((message, None))

    # Create a concise version of the chatbot state for logging
    concise_state = [(truncate_string(user_msg), truncate_string(bot_msg)) for user_msg, bot_msg in chatbot_state]
    logger.info(f"chatbot_output_callback chatbot_state: {concise_state} (truncated)")

def process_input(user_input, state:State):
    
    setup_state(state)

    # Append the user message to state["messages"]
    state["messages"].append(
            {
                "role": "user",
                "content": [TextBlock(type="text", text=user_input)],
            }
        )

    # Append the user's message to chatbot_messages with None for the assistant's reply
    state['chatbot_messages'].append((user_input, None))
    yield state['chatbot_messages']  # Yield to update the chatbot UI with the user's message

    # Run sampling_loop_sync with the chatbot_output_callback
    for loop_msg in sampling_loop_sync(
        system_prompt_suffix=state["custom_system_prompt"],
        planner_model=state["planner_model"],
        planner_provider=state["planner_provider"],
        actor_model=state["actor_model"],
        actor_provider=state["actor_provider"],
        messages=state["messages"],
        output_callback=partial(chatbot_output_callback, chatbot_state=state['chatbot_messages'], hide_images=state["hide_images"]),
        tool_output_callback=partial(_tool_output_callback, tool_state=state["tools"]),
        api_response_callback=partial(_api_response_callback, response_state=state["responses"]),
        api_key=state["planner_api_key"],
        only_n_most_recent_images=state["only_n_most_recent_images"],
        selected_screen=state['selected_screen'],
        showui_max_pixels=state['max_pixels'],
        showui_awq_4bit=state['awq_4bit']
    ):  
        if loop_msg is None:
            yield state['chatbot_messages']
            logger.info("End of task. Close the loop.")
            break
            
        yield state['chatbot_messages']  # Yield the updated chatbot_messages to update the chatbot UI

def update_only_n_images(only_n_images_value, state:State):
    """Update how many recent screenshots to keep in state."""
    try:
        state["only_n_most_recent_images"] = int(only_n_images_value)
    except Exception:
        state["only_n_most_recent_images"] = only_n_images_value
    logger.info(f"only_n_most_recent_images set to {state['only_n_most_recent_images']}")

def update_selected_screen(selected_screen_name, state:State):
    """
    Update selected screen index in state (called when screen_selector changes).
    Returns nothing because in your wiring the change handler had outputs=None.
    """
    global SCREEN_NAMES, SELECTED_SCREEN_INDEX
    try:
        if SCREEN_NAMES and selected_screen_name in SCREEN_NAMES:
            SELECTED_SCREEN_INDEX = SCREEN_NAMES.index(selected_screen_name)
        else:
            # fallback: keep previous or 0
            SELECTED_SCREEN_INDEX = SELECTED_SCREEN_INDEX if isinstance(SELECTED_SCREEN_INDEX, int) else 0
        state['selected_screen'] = SELECTED_SCREEN_INDEX
        logger.info(f"Selected screen updated to: {SELECTED_SCREEN_INDEX} ({selected_screen_name})")
    except Exception as e:
        logger.exception("Failed to update selected screen: %s", e)

def update_custom_info(url, state):
    model_url = f"{url}/v1/models"
    API_PROVIDERS['Custom'] = APIProvider(url=url, model_url=model_url)
    state['planner_url'] = url
    avail_models = list_available_models(model_url)
    state['available_models'] = avail_models
    print(avail_models)
    API_PROVIDERS['Custom']['avail_models'] = avail_models
    value = avail_models[0] if len(avail_models) else ''
    return gr.Dropdown(choices=avail_models, value=value, interactive=True)

def get_avail_planner_list(provider_name):
    """根据 provider 动态更新 model dropdown"""
    provider = API_PROVIDERS[provider_name]
    models = provider.get('avail_models', [])

    if not models:
        return gr.update(choices=[], value=None)

    return gr.update(choices=models, value=models[0])

def update_custom_box_status(choice):
    # 
    if choice == "Custom":
        if API_PROVIDERS[choice] is None:
            API_PROVIDERS[choice] = APIProvider()
        return gr.update(visible=True, value=API_PROVIDERS[choice].get('url'), interactive=True), gr.Dropdown(choices=API_PROVIDERS[choice].get('avail_models'), value='', interactive=True)
    else:
        return gr.update(visible=False, value=choice), gr.update()

def update_planner_model(model_selection: str, state:State):
    """
    Called when planner_model changes. Update state with the selected planner model and
    return three updates corresponding to outputs:
    (planner_api_provider, planner_api_key, actor_model)
    These are used in the Gradio wiring in your file.
    """
    if not model_selection:
        # no-op but return defaults
        # choose first provider key if available
        provider_default = next(iter(API_PROVIDERS.keys()), "")
        return (
            gr.update(choices=list(API_PROVIDERS.keys()), value=provider_default, interactive=True),
            gr.update(placeholder="Paste your planner model API key", value=state.get("planner_api_key", ""), interactive=True),
            gr.update(choices=["ShowUI", "UI-TARS"], value=state.get("actor_model", "ShowUI"), interactive=True),
        )

    # heuristics: map model name to provider
    ms = model_selection.lower()
    if ms.startswith("gpt") or ms.startswith("gpt-") or ms.startswith("gpt4") or "openai" in ms:
        provider_guess = "OpenAI"
    elif ms.startswith("claude") or "anthropic" in ms:
        provider_guess = "Anthropic"
    elif ms.startswith("qwen") or "qwen" in ms:
        provider_guess = "Qwen"
    else:
        # fallback to Local if model looks like a local model id (e.g., ShowUI)
        provider_guess = "Local"

    # try to pick a provider key from API_PROVIDERS that matches guess (case-insensitive)
    matched_provider_key = None
    for k in API_PROVIDERS.keys():
        if k.lower() == provider_guess.lower():
            matched_provider_key = k
            break
    if not matched_provider_key:
        matched_provider_key = next(iter(API_PROVIDERS.keys()), "")

    # update state
    state["planner_model"] = model_selection
    state["planner_provider"] = matched_provider_key
    logger.info(f"planner_model set to {model_selection}, planner_provider set to {matched_provider_key}")

    # Prepare gr.update objects to return to the three UI elements bound in your file
    provider_update = gr.update(choices=list(API_PROVIDERS.keys()), value=matched_provider_key, interactive=True)
    api_key_placeholder = "Paste your planner model API key"
    api_key_value = state.get("planner_api_key", "")
    api_key_update = gr.update(placeholder=api_key_placeholder, value=api_key_value, interactive=True, type="password")
    actor_model_update = gr.update(choices=["ShowUI", "UI-TARS"], value=state.get("actor_model", "ShowUI"), interactive=True)

    return provider_update, api_key_update, actor_model_update

def update_actor_model(actor_model_selection, state:State):
    """When actor model dropdown changes."""
    state["actor_model"] = actor_model_selection
    logger.info(f"Actor model updated to: {actor_model_selection}")

def update_system_prompt_suffix(system_prompt_suffix, state):
    """Update the system prompt suffix in state."""
    state["custom_system_prompt"] = system_prompt_suffix
    logger.info("custom_system_prompt updated.")

def handle_showui_config_change(showui_config_val, state:State):
    """
    Keep showui presets synchronized with max_pixels and awq_4bit.
    Returns two outputs: (max_pixels_update, awq_4bit_update)
    """
    try:
        if showui_config_val == "Default (Maximum)":
            state["max_pixels"] = 1344
            state["awq_4bit"] = False
            return (
                gr.update(value=1344, interactive=False),
                gr.update(value=False, interactive=False),
            )
        elif showui_config_val == "Medium":
            state["max_pixels"] = 1024
            state["awq_4bit"] = False
            return (
                gr.update(value=1024, interactive=False),
                gr.update(value=False, interactive=False),
            )
        elif showui_config_val == "Minimal":
            state["max_pixels"] = 1024
            state["awq_4bit"] = True
            return (
                gr.update(value=1024, interactive=False),
                gr.update(value=True, interactive=False),
            )
        elif showui_config_val == "Custom":
            # allow user control — do not overwrite values
            return (
                gr.update(interactive=True),
                gr.update(interactive=True),
            )
        else:
            # fallback: no change
            return (
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
    except Exception as e:
        logger.exception("handle_showui_config_change error: %s", e)
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

def update_api_key(api_key_value, state:State):
    """Handle changes to the planner API key textbox."""
    state["planner_api_key"] = api_key_value
    # If you keep an alias 'planner_provider' or 'api_key' elsewhere for SSH, ensure it's updated
    if state.get("planner_provider", "").lower() == "ssh":
        state["api_key"] = api_key_value
    logger.info(f"Planner API key updated. provider={state.get('planner_provider')}")

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    state0 = setup_state(State())  # Initialize the state
    state = gr.State(state0)  # Use Gradio's state management
    # Retrieve screen details
    gr.Markdown("# Computer Use OOTB")
    if not os.getenv("HIDE_WARNING", False):
        gr.Markdown(WARNING_TEXT)
    with gr.Accordion("Settings", open=True): 
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    planner_api_provider = gr.Dropdown(
                        label="API Provider",
                        choices=API_NAMES,
                        value="openai",
                        interactive=True,
                    )
                    custom_box = gr.Textbox(label="URL", visible=False, submit_btn=True)

            with gr.Column():
                # --------------------------
                # Planner
                planner_model = gr.Dropdown(
                    label="Planner Model",
                    choices=[],
                    value="",
                    interactive=True,
                )
            
            with gr.Column():
                planner_api_key = gr.Textbox(
                    label="Planner API Key",
                    type="password",
                    value=state.value.get("planner_api_key", ""),
                    placeholder="Paste your planner model API key",
                    interactive=True,
                )
            with gr.Column():
                actor_model = gr.Dropdown(
                    label="Actor Model",
                    choices=["ShowUI", "UI-TARS"],
                    value="ShowUI",
                    interactive=True,
                )
            with gr.Column():
                custom_prompt = gr.Textbox(
                    label="System Prompt Suffix",
                    value="",
                    interactive=True,
                )
            with gr.Column():
                screen_options, primary_index = get_screen_details()
                SCREEN_NAMES = screen_options
                SELECTED_SCREEN_INDEX = primary_index
                screen_selector = gr.Dropdown(
                    label="Select Screen",
                    choices=screen_options,
                    value=screen_options[primary_index] if screen_options else None,
                    interactive=True,
                )
            with gr.Column():
                only_n_images = gr.Slider(
                    label="N most recent screenshots",
                    minimum=0,
                    maximum=10,
                    step=1,
                    value=2,
                    interactive=True,
                )
    
    with gr.Accordion("ShowUI Advanced Settings", open=True):  
        
        gr.Markdown("""
                    **Note:** Adjust these settings to fine-tune the resource (**memory** and **infer time**) and performance trade-offs of ShowUI. \\
                    Quantization model requires additional download. Please refer to [Computer Use OOTB - #ShowUI Advanced Settings guide](https://github.com/showlab/computer_use_ootb?tab=readme-ov-file#showui-advanced-settings) for preparation for this feature.
                    """)
        # New configuration for ShowUI
        with gr.Row():
            with gr.Column():
                showui_config = gr.Dropdown(
                    label="ShowUI Preset Configuration",
                    choices=["Default (Maximum)", "Medium", "Minimal", "Custom"],
                    value="Default (Maximum)",
                    interactive=True,
                )
            with gr.Column():
                max_pixels = gr.Slider(
                    label="Max Visual Tokens",
                    minimum=720,
                    maximum=1344,
                    step=16,
                    value=1344,
                    interactive=False,
                )
            with gr.Column():
                awq_4bit = gr.Checkbox(
                    label="Enable AWQ-4bit Model",
                    value=False,
                    interactive=False
                )
            
    # Define the merged dictionary with task mappings
    merged_dict = json.load(open("assets/examples/ootb_examples.json", "r"))

    # Callback to update the second dropdown based on the first selection
    def update_second_menu(selected_category):
        return gr.update(choices=list(merged_dict.get(selected_category, {}).keys()))
    # Callback to update the third dropdown based on the second selection
    def update_third_menu(selected_category, selected_option):
        return gr.update(choices=list(merged_dict.get(selected_category, {}).get(selected_option, {}).keys()))
    # Callback to update the textbox based on the third selection
    def update_textbox(selected_category, selected_option, selected_task):
        task_data = merged_dict.get(selected_category, {}).get(selected_option, {}).get(selected_task, {})
        prompt = task_data.get("prompt", "")
        preview_image = task_data.get("initial_state", "")
        task_hint = "Task Hint: " + task_data.get("hint", "")
        return prompt, preview_image, task_hint
    
    with gr.Accordion("Quick Start Prompt", open=False):  # open=False 表示默认收
        # Initialize Gradio interface with the dropdowns
        with gr.Row():
            # Set initial values
            initial_category = "Game Play"
            initial_second_options = list(merged_dict[initial_category].keys())
            initial_third_options = list(merged_dict[initial_category][initial_second_options[0]].keys())
            initial_text_value = merged_dict[initial_category][initial_second_options[0]][initial_third_options[0]]
            with gr.Column(scale=2):
                # First dropdown for Task Category
                first_menu = gr.Dropdown(
                    choices=list(merged_dict.keys()), label="Task Category", interactive=True, value=initial_category
                )
                # Second dropdown for Software
                second_menu = gr.Dropdown(
                    choices=initial_second_options, label="Software", interactive=True, value=initial_second_options[0]
                )
                # Third dropdown for Task
                third_menu = gr.Dropdown(
                    choices=initial_third_options, label="Task", interactive=True, value=initial_third_options[0]
                    # choices=["Please select a task"]+initial_third_options, label="Task", interactive=True, value="Please select a task"
                )
            with gr.Column(scale=1):
                initial_image_value = "./assets/examples/init_states/honkai_star_rail_showui.png"  # default image path
                image_preview = gr.Image(value=initial_image_value, label="Reference Initial State", height=260-(318.75-280))
                hintbox = gr.Markdown("Task Hint: Selected options will appear here.")

    with gr.Row():
        # submit_button = gr.Button("Submit")  # Add submit button
        with gr.Column(scale=8):
            chat_input = gr.Textbox(show_label=False, placeholder="Type a message to send to Computer Use OOTB...", container=False)
        with gr.Column(scale=1, min_width=50):
            submit_button = gr.Button(value="Send", variant="primary")
    chatbot = gr.Chatbot(label="Chatbot History", type="tuples", autoscroll=True, height=580, group_consecutive_messages=False)
    
    # planner_api_provider.change(fn=update_api_provider, inputs=[planner_api_provider, state], outputs=[custom_box, planner_model])
    planner_api_provider.change(fn=get_avail_planner_list, inputs=planner_api_provider, outputs=planner_model)
    planner_api_provider.change(fn=update_custom_box_status, inputs=planner_api_provider, outputs=[custom_box, planner_model])
    
    custom_box.submit(fn=update_custom_info, inputs=[custom_box, state], outputs=planner_model)
    planner_model.change(fn=update_planner_model, inputs=[planner_model, state], outputs=[planner_api_provider, planner_api_key, actor_model])
    
    actor_model.change(fn=update_actor_model, inputs=[actor_model, state], outputs=None)
    screen_selector.change(fn=update_selected_screen, inputs=[screen_selector, state], outputs=None)
    only_n_images.change(fn=update_only_n_images, inputs=[only_n_images, state], outputs=None)
    
    # When showui_config changes, we update max_pixels and awq_4bit automatically.
    showui_config.change(fn=handle_showui_config_change, 
                        inputs=[showui_config, state], 
                        outputs=[max_pixels, awq_4bit])
    
    # Link callbacks to update dropdowns based on selections
    first_menu.change(fn=update_second_menu, inputs=first_menu, outputs=second_menu)
    second_menu.change(fn=update_third_menu, inputs=[first_menu, second_menu], outputs=third_menu)
    third_menu.change(fn=update_textbox, inputs=[first_menu, second_menu, third_menu], outputs=[chat_input, image_preview, hintbox])
    # chat_input.submit(process_input, [chat_input, state], chatbot)
    submit_button.click(process_input, [chat_input, state], chatbot)
    planner_api_key.change(
        fn=update_api_key,
        inputs=[planner_api_key, state],
        outputs=None
    )

if __name__ == '__main__':
    demo.launch(share=False,
        allowed_paths=["./"],
        server_port=7888)  # TODO: allowed_paths
