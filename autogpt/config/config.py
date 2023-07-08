import pinecone
import numpy as np

def main():
    pinecone.init(api_key="fa7d752e-735c-49ef-b473-6ca455369b0e")
    index_name = "vecs"
    num_vectors = 128
    vector_dimension = 300
    vectors = {str(i): np.random.rand(vector_dimension).tolist() for i in range(num_vectors)}
    pinecone.deposit(index_name, vectors)
    query_vector = np.random.rand(vector_dimension).tolist()
    top_k_results = pinecone.fetch(index_name, query_vector, top_k=5)
    print(top_k_results)
    pinecone.deinit()

if __name__ == "__main__":
    main()
import os
import re
from typing import List
import openai
import yaml
from auto_gpt_plugin_template import AutoGPTPluginTemplate
from colorama import Fore
import autogpt

class Config:
    def __init__(self) -> None:
        self.workspace_path: str = None
        self.file_logger_path: str = None
        self.debug_mode = False
        self.continuous_mode = False
        self.continuous_limit = 0
        self.speak_mode = False
        self.skip_reprompt = False
        self.allow_downloads = False
        self.skip_news = False
        self.authorise_key = os.getenv("AUTHORISE_COMMAND_KEY", "y")
        self.exit_key = os.getenv("EXIT_KEY", "n")
        self.plain_output = os.getenv("PLAIN_OUTPUT", "False") == "True"
        disabled_command_categories = os.getenv("DISABLED_COMMAND_CATEGORIES")
        self.disabled_command_categories = disabled_command_categories.split(",") if disabled_command_categories else []
        self.shell_command_control = os.getenv("SHELL_COMMAND_CONTROL", "denylist")
        shell_denylist = os.getenv("SHELL_DENYLIST", os.getenv("DENY_COMMANDS"))
        self.shell_denylist = shell_denylist.split(",") if shell_denylist else ["sudo", "su"]
        shell_allowlist = os.getenv("SHELL_ALLOWLIST", os.getenv("ALLOW_COMMANDS"))
        self.shell_allowlist = shell_allowlist.split(",") if shell_allowlist else []
        self.ai_settings_file = os.getenv("AI_SETTINGS_FILE", "ai_settings.yaml")
        self.prompt_settings_file = os.getenv("PROMPT_SETTINGS_FILE", "prompt_settings.yaml")
        self.fast_llm_model = os.getenv("FAST_LLM_MODEL", "gpt-3.5-turbo")
        self.smart_llm_model = os.getenv("SMART_LLM_MODEL", "gpt-3.5-turbo")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.browse_spacy_language_model = os.getenv("BROWSE_SPACY_LANGUAGE_MODEL", "en_core_web_sm")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_organization = os.getenv("OPENAI_ORGANIZATION")
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyCEOdc3l_M0qRPvmtpBRket3J3FEuj1ie0")
        self.temperature = float(os.getenv("TEMPERATURE", "0"))
        self.use_azure = os.getenv("USE_AZURE") == "True"
        self.execute_local_commands = os.getenv("EXECUTE_LOCAL_COMMANDS", "False") == "True"
        if self.use_azure:
            self.load_azure_config()
            openai.api_type = self.openai_api_type
            openai.api_base = self.openai_api_base
            openai.api_version = self.openai_api_version
        elif os.getenv("OPENAI_API_BASE_URL", None):
            openai.api_base = os.getenv("OPENAI_API_BASE_URL")
        if self.openai_organization is not None:
            openai.organization = self.openai_organization
        self.openai_functions = os.getenv("OPENAI_FUNCTIONS", "False") == "True"
        self.elevenlabs_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "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")
        default_tts_provider = "macos" if os.getenv("USE_MAC_OS_TTS") else "elevenlabs" if self.elevenlabs_api_key else "streamelements" if os.getenv("USE_BRIAN_TTS") else "gtts"
        self.text_to_speech_provider = os.getenv("TEXT_TO_SPEECH_PROVIDER", default_tts_provider)
        self.github_api_key = os.getenv("GITHUB_API_KEY")
        self.github_username = os.getenv("GITHUB_USERNAME")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.google_custom_search_engine_id = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID", os.getenv("CUSTOM_SEARCH_ENGINE_ID"))
        self.image_provider = os.getenv("IMAGE_PROVIDER")
        self.image_size = int(os.getenv("IMAGE_SIZE", 256))
        self.huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.huggingface_image_model = os.getenv("HUGGINGFACE_IMAGE_MODEL", "CompVis/stable-diffusion-v1-4")
        self.audio_to_text_provider = os.getenv("AUDIO_TO_TEXT_PROVIDER", "huggingface")
        self.huggingface_audio_to_text_model = os.getenv("HUGGINGFACE_AUDIO_TO_TEXT_MODEL")
        self.sd_webui_url = os.getenv("SD_WEBUI_URL", "http://localhost:7860")
        self.sd_webui_auth = os.getenv("SD_WEBUI_AUTH")
        selenium_web_browser=r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
        self.selenium_headless = os.getenv("HEADLESS_BROWSER", "True") == "True"
        self.user_agent = os.getenv("USER_AGENT", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36")
        self.memory_backend = os.getenv("MEMORY_BACKEND", "json_file")
        self.memory_index = os.getenv("MEMORY_INDEX", "auto-gpt-memory")
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = os.getenv("REDIS_PASSWORD", "")
        self.wipe_redis_on_start = os.getenv("WIPE_REDIS_ON_START", "True") == "True"
        self.plugins_dir = os.getenv("PLUGINS_DIR", "plugins")
        self.plugins: List[AutoGPTPluginTemplate] = []
        self.plugins_openai = []
        plugins_allowlist = os.getenv("ALLOWLISTED_PLUGINS")
        self.plugins_allowlist = plugins_allowlist.split(",") if plugins_allowlist else []
        plugins_denylist = os.getenv("DENYLISTED_PLUGINS")
        self.plugins_denylist = plugins_denylist.split(",") if plugins_denylist else []
        from autogpt.plugins import DEFAULT_PLUGINS_CONFIG_FILE
       
