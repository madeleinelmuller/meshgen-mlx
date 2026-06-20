from pathlib import Path

from smolagents import Model


def get_models_dir():
    """Get path to the local models directory."""
    return Path(__file__).parent / "models"


def get_available_models():
    """List all GGUF models in the local models directory."""
    models_dir = get_models_dir()
    if not models_dir.exists():
        return []
    return [f.name for f in models_dir.glob("*.gguf")]


class LlamaCppModel(Model):
    """Custom LlamaCppModel, since smolagents doesn't yet support llama_cpp_python."""

    def __init__(
        self,
        model_path,
        n_gpu_layers=-1,
        n_ctx=8192,
        max_tokens=8192,
        **kwargs,
    ):
        super().__init__(**kwargs)
        import llama_cpp

        self.model = llama_cpp.Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            max_tokens=max_tokens,
        )

    def __call__(
        self,
        messages,
        stop_sequences=None,
        grammar=None,
        tools_to_call_from=None,
        **kwargs,
    ):
        from llama_cpp import LlamaGrammar
        from smolagents.models import ChatMessage, remove_stop_sequences

        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            grammar=grammar,
            tools_to_call_from=tools_to_call_from,
            stream=False,
            **kwargs,
        )

        filtered_kwargs = {
            k: v
            for k, v in completion_kwargs.items()
            if k
            not in [
                "messages",
                "stop",
                "grammar",
                "max_tokens",
                "tools_to_call_from",
            ]
        }

        processed_messages = []
        for msg in completion_kwargs["messages"]:
            if isinstance(msg["content"], list):
                content_str = ""
                for item in msg["content"]:
                    if item.get("type") == "text":
                        content_str += item["text"]
                processed_msg = {"role": msg["role"], "content": content_str}
            else:
                processed_msg = msg
            processed_messages.append(processed_msg)

        print(processed_messages)

        response = self.model.create_chat_completion(
            messages=processed_messages,
            stop=completion_kwargs.get("stop", []),
            grammar=LlamaGrammar.from_string(grammar) if grammar else None,
            **filtered_kwargs,
        )

        print(response)

        if isinstance(response, dict):
            if "choices" in response:
                content = response["choices"][0]["message"]["content"]
            else:
                content = response.get("message", {}).get("content", "")
                if not content and "content" in response:
                    content = response["content"]
        else:
            content = response.choices[0].message.content

        if stop_sequences:
            content = remove_stop_sequences(content, stop_sequences)

        if isinstance(response, dict):
            usage = response.get("usage", {})
            self.last_input_token_count = usage.get("prompt_tokens", 0)
            self.last_output_token_count = usage.get("completion_tokens", 0)
        else:
            if hasattr(response, "usage"):
                self.last_input_token_count = response.usage.prompt_tokens
                self.last_output_token_count = response.usage.completion_tokens
            else:
                self.last_input_token_count = 0
                self.last_output_token_count = 0

        return ChatMessage(role="assistant", content=content)


class MLXModel(Model):
    """LLM inference backend using mlx-lm for Apple Silicon (M-series) GPUs."""

    def __init__(self, model_id, max_tokens=8192, **kwargs):
        super().__init__(**kwargs)
        import mlx_lm

        self.max_tokens = max_tokens
        self._model, self._tokenizer = mlx_lm.load(model_id)

    def __call__(
        self,
        messages,
        stop_sequences=None,
        tools_to_call_from=None,
        **kwargs,
    ):
        import mlx_lm
        from smolagents.models import ChatMessage, remove_stop_sequences

        # Flatten content lists to plain strings for the chat template
        flat_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg["role"]
                content = msg["content"]
            else:
                role = msg.role
                content = msg.content

            if isinstance(content, list):
                content = "".join(
                    item["text"] for item in content if item.get("type") == "text"
                )
            flat_messages.append({"role": role, "content": content})

        prompt = self._tokenizer.apply_chat_template(
            flat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        temperature = getattr(self, "temperature", 0.7) or 0.7
        response = mlx_lm.generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            temp=temperature,
            verbose=False,
        )

        if stop_sequences:
            response = remove_stop_sequences(response, stop_sequences)

        self.last_input_token_count = 0
        self.last_output_token_count = 0

        return ChatMessage(role="assistant", content=response)
