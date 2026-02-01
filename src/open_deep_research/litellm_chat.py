"""LiteLLM Chat Model wrapper for LangChain compatibility."""

import json
from typing import Any, List, Optional, Type, Union

import litellm
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
try:
    from langchain_core.pydantic_v1 import BaseModel, ValidationError
except Exception:  # pragma: no cover - fallback for environments without langchain_core shim
    from pydantic import BaseModel, ValidationError

from pydantic import TypeAdapter


class LiteLLMChat(BaseChatModel):
    """Wrapper around LiteLLM to be compatible with LangChain."""

    model: str
    """The model name (e.g., 'gpt-4', 'claude-3-sonnet', 'openai/wine-gemini-2.5-flash')"""
    api_key: Optional[str] = None
    """API key for the model"""
    base_url: Optional[str] = None
    """Base URL for custom LiteLLM gateway"""
    max_tokens: Optional[int] = None
    """Maximum tokens in response"""
    temperature: float = 0.7
    """Temperature for generation"""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response synchronously."""
        # Convert LangChain messages to litellm format
        formatted_messages = [
            {"role": msg.type, "content": msg.content}
            for msg in messages
        ]

        completion_kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_key": self.api_key,
        }
        
        if self.base_url:
            completion_kwargs["base_url"] = self.base_url
            
        completion_kwargs.update(kwargs)

        response = litellm.completion(**completion_kwargs)

        content = response.choices[0].message.content
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response asynchronously."""
        formatted_messages = [
            {"role": msg.type, "content": msg.content}
            for msg in messages
        ]

        completion_kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_key": self.api_key,
        }
        
        if self.base_url:
            completion_kwargs["base_url"] = self.base_url
            
        completion_kwargs.update(kwargs)

        response = await litellm.acompletion(**completion_kwargs)

        content = response.choices[0].message.content
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content))]
        )

    @property
    def _llm_type(self) -> str:
        return "litellm"

    def with_structured_output(
        self,
        schema: Union[Type[BaseModel], dict],
        **kwargs: Any,
    ) -> "LiteLLMChatStructured":
        """Return a structured output version of this model."""
        return LiteLLMChatStructured(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            schema=schema,
        )


class LiteLLMChatStructured(LiteLLMChat):
    """LiteLLM Chat model with structured output support."""

    schema: Union[Type[BaseModel], dict] = None
    """The output schema for structured generation"""

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate structured chat response synchronously."""
        # Convert LangChain messages to litellm format
        formatted_messages = [
            {"role": msg.type, "content": msg.content}
            for msg in messages
        ]

        # Add JSON schema to system prompt
        schema_str = self._schema_to_json_schema()
        system_prompt = f"""You must respond with valid JSON that matches this schema:
{schema_str}

Respond ONLY with the JSON object, no other text."""

        # Prepend schema to messages
        if formatted_messages and formatted_messages[0]["role"] == "system":
            formatted_messages[0]["content"] = system_prompt + "\n" + formatted_messages[0]["content"]
        else:
            formatted_messages.insert(0, {"role": "system", "content": system_prompt})

        completion_kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_key": self.api_key,
            "response_format": {"type": "json_object"},
        }

        if self.base_url:
            completion_kwargs["base_url"] = self.base_url

        completion_kwargs.update(kwargs)

        try:
            response = litellm.completion(**completion_kwargs)
            content = response.choices[0].message.content
            
            # Parse JSON response
            parsed = json.loads(content)
            
            # Validate against schema
            if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
                validated = self.schema(**parsed)
            else:
                validated = parsed
            
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=validated))]
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {content}")

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate structured chat response asynchronously."""
        formatted_messages = [
            {"role": msg.type, "content": msg.content}
            for msg in messages
        ]

        # Add JSON schema to system prompt
        schema_str = self._schema_to_json_schema()
        system_prompt = f"""You must respond with valid JSON that matches this schema:
{schema_str}

Respond ONLY with the JSON object, no other text."""

        # Prepend schema to messages
        if formatted_messages and formatted_messages[0]["role"] == "system":
            formatted_messages[0]["content"] = system_prompt + "\n" + formatted_messages[0]["content"]
        else:
            formatted_messages.insert(0, {"role": "system", "content": system_prompt})

        completion_kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_key": self.api_key,
            "response_format": {"type": "json_object"},
        }

        if self.base_url:
            completion_kwargs["base_url"] = self.base_url

        completion_kwargs.update(kwargs)

        try:
            response = await litellm.acompletion(**completion_kwargs)
            content = response.choices[0].message.content
            
            # Parse JSON response
            parsed = json.loads(content)
            
            # Validate against schema
            if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
                validated = self.schema(**parsed)
            else:
                validated = parsed
            
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=validated))]
            )
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {content}")

    def _schema_to_json_schema(self) -> str:
        """Convert Pydantic schema to JSON Schema string."""
        if isinstance(self.schema, type) and issubclass(self.schema, BaseModel):
            return json.dumps(self.schema.model_json_schema(), indent=2)
        elif isinstance(self.schema, dict):
            return json.dumps(self.schema, indent=2)
        else:
            return str(self.schema)
