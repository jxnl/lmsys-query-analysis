"""LLM-based row summarization for query datasets.

Transforms raw queries into structured, canonical representations using
instructor for type-safe structured outputs from LLMs.
"""

import logging
from typing import Any

import anyio
import instructor
from jinja2 import Template
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from ..db.models import Query

logger = logging.getLogger("lmsys")


class Property(BaseModel):
    """A single property extracted from query summarization."""

    name: str = Field(..., description="Property name (e.g., 'intent', 'complexity', 'language')")
    value: Any = Field(
        ...,
        description="Property value (can be string, int, float, bool, or list)",
    )


class QuerySummaryResponse(BaseModel):
    """Structured response from LLM for query summarization."""

    summary: str = Field(
        ...,
        description="Concise summary of the query (the transformed query text)",
    )
    properties: list[Property] = Field(
        default_factory=list,
        description="Optional extracted properties as key-value pairs",
    )


# Jinja2 template for serializing Query to XML
QUERY_XML_TEMPLATE = Template(
    """<query>
  <query_text>{{ query.query_text }}</query_text>
  <model>{{ query.model }}</model>
  <language>{{ query.language or 'unknown' }}</language>
  <timestamp>{{ query.timestamp or 'unknown' }}</timestamp>
  {% if query.extra_metadata %}
  <extra_metadata>
    {% for key, value in query.extra_metadata.items() %}
    <{{ key }}>{{ value }}</{{ key }}>
    {% endfor %}
  </extra_metadata>
  {% endif %}
</query>"""
)

# Jinja2 template for serializing few-shot examples to XML
EXAMPLES_XML_TEMPLATE = Template(
    """<examples>
  {% for input_query, expected_output in examples %}
  <example>
    <input>
{{ input_xml[loop.index0] | indent(6, first=True) }}
    </input>
    <output>{{ expected_output }}</output>
  </example>
  {% endfor %}
</examples>"""
)


def serialize_query_to_xml(query: Query) -> str:
    """Serialize a Query object to XML format.

    Args:
        query: Query object to serialize

    Returns:
        XML string representation of the query

    Example output:
        <query>
          <query_text>hey can you help me write python script?</query_text>
          <model>gpt-3.5-turbo</model>
          <language>en</language>
          <timestamp>2024-01-15T10:30:00Z</timestamp>
        </query>
    """
    return QUERY_XML_TEMPLATE.render(query=query)


def format_examples_xml(examples: list[tuple[Query, str]]) -> str:
    """Format few-shot examples as XML.

    Args:
        examples: List of (input_query, expected_output) tuples

    Returns:
        XML string with formatted examples

    Example output:
        <examples>
          <example>
            <input>
              <query>
                <query_text>help me code python</query_text>
                <model>gpt-3.5</model>
                <language>en</language>
              </query>
            </input>
            <output>Write Python code</output>
          </example>
        </examples>
    """
    # Pre-serialize all input queries to XML
    input_xml = [serialize_query_to_xml(query) for query, _ in examples]
    return EXAMPLES_XML_TEMPLATE.render(examples=examples, input_xml=input_xml)


class RowSummarizer:
    """Summarize individual queries using LLM with custom prompts.

    IMPORTANT: Write DETAILED, COMPREHENSIVE prompts (200-500 words recommended).
    Short prompts like "Extract intent: {query}" produce low-quality, inconsistent results.
    
    Good prompts include:
    - Clear task description and goals
    - Specific output format requirements
    - Edge case handling instructions
    - Property definitions with examples
    - Few-shot examples via {examples} placeholder
    
    Args:
        model: Model in format "provider/model_name" (e.g., "openai/gpt-4o-mini")
        prompt_template: DETAILED template string with {query} and optional {examples} placeholders
        examples: Optional list of (input_query, expected_output) for few-shot learning
        concurrency: Number of concurrent LLM requests (default: 100)
        api_key: Optional API key (or use env vars)

    Example:
        >>> prompt = '''
        ... You are analyzing user queries to extract structured information.
        ... 
        ... Task: Extract the core user intent and classify the query.
        ... 
        ... Output format:
        ... - summary: One clear sentence describing what the user wants (10-15 words)
        ... - properties: Extract these specific fields:
        ...   * intent: One of [code_generation, debugging, explanation, creative_writing, 
        ...     question_answering, translation, other]
        ...   * complexity: Integer 1-5 (1=simple, 5=requires multi-step reasoning)
        ...   * domain: Primary topic (e.g., "python", "javascript", "creative", "general")
        ...   * requires_code: Boolean - does user expect code in response?
        ... 
        ... Guidelines:
        ... - Ignore greetings, pleasantries, and filler words
        ... - If query is in non-English language, translate to English for summary
        ... - For vague queries, infer most likely intent from context
        ... - Code snippets in query suggest debugging/code_generation intent
        ... 
        ... Examples:
        ... {examples}
        ... 
        ... Now analyze this query:
        ... {query}
        ... '''
        >>> summarizer = RowSummarizer(
        ...     model="openai/gpt-4o-mini",
        ...     prompt_template=prompt,
        ...     examples=[(query1, "Write Python function"), (query2, "Debug JavaScript error")]
        ... )
        >>> summaries = summarizer.summarize_batch(queries)
    """

    def __init__(
        self,
        model: str,
        prompt_template: str,
        examples: list[tuple[Query, str]] | None = None,
        concurrency: int = 100,
        api_key: str | None = None,
    ):
        self.model = model
        self.prompt_template = prompt_template
        self.examples = examples or []
        self.concurrency = concurrency
        self.logger = logging.getLogger("lmsys")

        # Validate prompt template has {query} placeholder
        if "{query}" not in prompt_template:
            raise ValueError("prompt_template must contain {query} placeholder")

        # Pre-format examples if provided
        self.formatted_examples = ""
        if self.examples:
            self.formatted_examples = format_examples_xml(self.examples)
            if "{examples}" not in prompt_template:
                self.logger.warning(
                    "Examples provided but {examples} placeholder not found in prompt template. "
                    "Examples will be ignored."
                )

        # Initialize async instructor client
        if api_key:
            self.async_client = instructor.from_provider(model, api_key=api_key, async_client=True)
        else:
            self.async_client = instructor.from_provider(model, async_client=True)

    def summarize_batch(
        self,
        queries: list[Query],
    ) -> dict[int, dict]:
        """Summarize batch of queries concurrently.

        Args:
            queries: List of Query objects

        Returns:
            Dict mapping query.id -> {"summary": str, "properties": dict}
        """
        return anyio.run(self._async_summarize_batch, queries)

    async def _async_summarize_batch(
        self,
        queries: list[Query],
    ) -> dict[int, dict]:
        """Async concurrent summarization."""
        results: dict[int, dict] = {}
        semaphore = anyio.Semaphore(self.concurrency)

        async def worker(query: Query):
            async with semaphore:
                result = await self._summarize_single(query)
                results[query.id] = result

        async with anyio.create_task_group() as tg:
            for query in queries:
                tg.start_soon(worker, query)

        return results

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=8))
    async def _summarize_single(self, query: Query) -> dict:
        """Summarize single query with retries.

        Args:
            query: Full Query record (will be serialized to XML)

        Returns:
            Dict with "summary" and "properties" keys
            Example: {
                "summary": "Write Python code",
                "properties": {
                    "intent": "code_generation",
                    "complexity": 2,
                    "programming_language": "python"
                }
            }
        """
        # Serialize query to XML
        query_xml = serialize_query_to_xml(query)

        # Format prompt with XML query and examples
        prompt_content = self.prompt_template.format(
            query=query_xml, examples=self.formatted_examples
        )

        messages = [
            {
                "role": "user",
                "content": prompt_content,
            }
        ]

        # LLM returns structured JSON via instructor
        response = await self.async_client.chat.completions.create(
            response_model=QuerySummaryResponse,
            messages=messages,
        )

        # Convert properties list to dict for easy storage in extra_metadata
        properties_dict = {prop.name: prop.value for prop in response.properties}

        return {
            "summary": response.summary,
            "properties": properties_dict,
        }
