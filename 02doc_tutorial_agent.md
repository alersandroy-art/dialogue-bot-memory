https://haystack.deepset.ai/tutorials/43_building_a_tool_calling_agent
Tutorial: Build a Tool-Calling Agent

Last Updated: June 12, 2026
Level: Beginner
Time to complete: 15 minutes
Components Used: Agent, OpenAIChatGenerator, SerperDevWebSearch, ComponentTool, PipelineTool
Prerequisites: You must have an OpenAI API Key and a SerperDev API Key
Goal: After completing this tutorial, you’ll have learned how to create an Agent that can use both components and pipelines as tools to answer questions and perform tasks.
Overview
In this tutorial, you’ll learn how to create an agent that can use tools to answer questions and perform tasks. We’ll explore two approaches:

Using the Agent with a simple web search tool
Using the Agent with a more complex pipeline with multiple components
The Agent component allows you to create AI assistants that can use tools to gather information, perform actions, and interact with external systems. It uses a large language model (LLM) to understand user queries and decide which tools to use to answer them.

Preparing the Environment
First, let’s install Haystack and two other dependencies we’ll need later:

%%bash

pip install haystack-ai serperdev-haystack docstring-parser trafilatura
Enter API Keys
Enter your API keys for OpenAI and SerperDev:

from getpass import getpass
import os

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass("Enter OpenAI API key:")
if "SERPERDEV_API_KEY" not in os.environ:
    os.environ["SERPERDEV_API_KEY"] = getpass("Enter SerperDev API key: ")
Using Agent with a Component as a Tool
We start with a simple example of using the Agent as a standalone component with a web search tool. The tool can trigger web searches and fetch the search engine results page (SERP) containing the most relevant search hits.

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack_integrations.components.websearch.serperdev import SerperDevWebSearch
from haystack.dataclasses import ChatMessage
from haystack.tools.component_tool import ComponentTool

# Create a web search tool using SerperDevWebSearch
web_tool = ComponentTool(component=SerperDevWebSearch(), name="web_tool")

# Create the agent with the web search tool
agent = Agent(chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"), tools=[web_tool])

# Run the agent with a query
result = agent.run(messages=[ChatMessage.from_user("Find information about Haystack AI framework")])

# Print the final response
print(result["messages"][-1].text)
The Agent has a couple of optional parameters that let you customize it’s behavior:

system_prompt for defining a system prompt with instructions for the Agent’s LLM
exit_conditions that will cause the agent to return. It’s a list of strings and the items can be "text", which means that the Agent will exit as soon as the LLM replies only with a text response, or specific tool names, which make the Agent return right after a tool with that name was called.
state_schema for the State that is shared across one agent invocation run. It defines extra information – such as documents or context – that tools can read from or write to during execution. You can use this schema to pass parameters that tools can both produce and consume.
streaming_callback to stream the tokens from the LLM directly to output.
raise_on_tool_invocation_failure to decide if the agent should raise an exception when a tool invocation fails. If set to False, the exception will be turned into a chat message and passed to the LLM. It can then try to improve with the next tool invocation.
max_agent_steps to limit how many times the Agent can call tools and prevent endless loops.
When exit_conditions is set to the default [“text”], you can enable streaming so that we see the tokens of the response while they are being generated.

from haystack.components.generators.utils import print_streaming_chunk

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"), tools=[web_tool], streaming_callback=print_streaming_chunk
)

result = agent.run(messages=[ChatMessage.from_user("Find information about Haystack AI framework")])
You can easily switch out the ChatGenerator used in the Agent. Currently all of the following ChatGenerators support tools and thus can be used with Agent:

AmazonBedrockChatGenerator
AnthropicChatGenerator
AzureOpenAIChatGenerator
CohereChatGenerator
GoogleAIGeminiChatGenerator
HuggingFaceAPIChatGenerator
HuggingFaceLocalChatGenerator
MistralChatGenerator
OllamaChatGenerator
OpenAIChatGenerator
VertexAIGeminiChatGenerator
For example, if you have a HF_API_TOKEN and huggingface_hub[inference]>=0.27.0 installed, all you need to do is replace OpenAIChatGenerator by HuggingFaceAPIChatGenerator and run from haystack.components.generators.chat import HuggingFaceAPIChatGenerator

Using Agent with a Pipeline as Tool
Now, for a more sophisticated example, let’s build a research assistant that can search the web, fetch content from links, and generate comprehensive answers. In contrast to our previous Agent, we now want to follow the links on the search engine results page, access their content and parse their content through We’ll start with a Haystack Pipeline that the Agent can use as a tool:

from haystack.components.converters.html import HTMLToDocument
from haystack.components.converters.output_adapter import OutputAdapter
from haystack.components.fetchers.link_content import LinkContentFetcher
from haystack_integrations.components.websearch.serperdev import SerperDevWebSearch
from haystack.dataclasses import ChatMessage
from haystack.core.pipeline import Pipeline

search_pipeline = Pipeline()

search_pipeline.add_component("search", SerperDevWebSearch(top_k=10))
search_pipeline.add_component("fetcher", LinkContentFetcher(timeout=3, raise_on_failure=False, retry_attempts=2))
search_pipeline.add_component("converter", HTMLToDocument())
search_pipeline.add_component(
    "output_adapter",
    OutputAdapter(
        template="""
{%- for doc in docs -%}
  {%- if doc.content -%}
  <search-result url=\"{{ doc.meta.url }}\">
  {{ doc.content|truncate(25000) }}
  </search-result>
  {%- endif -%}
{%- endfor -%}
""",
        output_type=str,
    ),
)

search_pipeline.connect("search.links", "fetcher.urls")
search_pipeline.connect("fetcher.streams", "converter.sources")
search_pipeline.connect("converter.documents", "output_adapter.docs")
Creating a Tool from a Pipeline
Next, wrap the search_pipeline in a PipelineTool. PipelineTool directly exposes a pipeline as an LLM-callable tool, replacing the older pattern of wrapping a pipeline in a SuperComponent and then passing it to ComponentTool.

Use input_mapping and output_mapping to control which pipeline inputs and outputs are exposed. Here, input_mapping ensures only "query" is surfaced in the tool schema, and output_mapping extracts the formatted string produced by output_adapter.

Finally, you can initialize the Agent with the resulting search_tool.

💡 Learn alternative ways of creating tools in Tool and MCPTool documentation pages.

from haystack.tools import PipelineTool
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator

search_tool = PipelineTool(
    name="search",
    description="Use this tool to search for information on the internet.",
    pipeline=search_pipeline,
    input_mapping={"query": ["search.query"]},
    output_mapping={"output_adapter.output": "search_result"},
    outputs_to_string={"source": "search_result"},
)

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    tools=[search_tool],
    system_prompt="""
    You are a deep research assistant.
    You create comprehensive research reports to answer the user's questions.
    You use the 'search'-tool to answer any questions.
    You perform multiple searches until you have the information you need to answer the question.
    Make sure you research different aspects of the question.
    Use markdown to format your response.
    When you use information from the websearch results, cite your sources using markdown links.
    It is important that you cite accurately.
    """,
    exit_conditions=["text"],
    max_agent_steps=20,
)
Our Agent is ready to use! It is good practice to call agent.warm_up() before running an Agent, which makes sure models are loaded in case that’s required.

query = "What are the latest updates on the Artemis moon mission?"
messages = [ChatMessage.from_user(query)]

agent_output = agent.run(messages=messages)

print(agent_output["messages"][-1].text)
To render the Agent response in a markdown format, run the code snippet:

from IPython.display import Markdown, display

display(Markdown(agent_output["messages"][-1].text))
Let’s break down this last example in the tutorial. The Agent is the main component that orchestrates the interaction between the LLM and tools. We use ComponentTool as a wrapper that allows individual Haystack components to be used as tools by the agent. The PipelineTool wraps entire pipelines so that they can be used as tools directly, without needing an intermediate SuperComponent.

We created a sophisticated search pipeline that:

Searches the web using SerperDevWebSearch
Fetches content from the found links
Converts HTML content to Documents
Formats the results for the Agent
The Agent then uses this pipeline as a tool to gather information and generate comprehensive answers.

By the way, did you know that the Agent is a Haystack component itself? That means you can use and combine an Agent in your pipelines just like any other component!

What’s next
🎉 Congratulations! You’ve learned how to create a tool-calling Agent with Haystack. You can now:

Create simple agents with basic tools
Build complex pipelines with multiple components
Use the Agent component to create sophisticated AI assistants
Combine web search, content fetching, and document processing in your applications
If you liked this tutorial, you may also enjoy reusing pipelines from the following examples and make them tools of a powerful Agent:

Build a GitHub Issue Resolver Agent
Building Fallbacks with Conditional Routing
To stay up to date on the latest Haystack developments, you can subscribe to our newsletter and join Haystack discord community.