Get Started
Haystack is an open-source AI framework to build custom production-grade LLM applications such as AI agents, powerful RAG applications, and scalable search systems.

Installation
Use pip to install Haystack:

pip install haystack-ai
For more details, refer to our documentation.

Prerequisites
To run the example, you’ll need:

A SerperDev API Key for web search
Credentials for the model provider of your choice (e.g. OpenAI, Anthropic, Gemini, Amazon Bedrock)
🤖 Basic Agent with Haystack
You can build a working agent in just a few lines with the Agent component. It takes in a user question, decides whether to use a tool (like web search), and returns a response without manual routing.

Below is a minimal example using SerperDevWebSearch component as a tool with different models:

OpenAI

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import ComponentTool
from haystack.components.websearch import SerperDevWebSearch

os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI API KEY>"
os.environ["SERPERDEV_API_KEY"] = "<YOUR SERPERDEV API KEY>"

search_tool = ComponentTool(component=SerperDevWebSearch())

basic_agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    system_prompt="You are a helpful web agent.",
    tools=[search_tool],
)

result = basic_agent.run(messages=[ChatMessage.from_user("When was the first version of Haystack released?")])

print(result['last_message'].text)
⚙️ Advanced Agent Configurations
Once you’ve built your first agent, it’s simple to extend its capabilities to fit more advanced use cases. Haystack is designed to be modular and customizable, so you can easily fine-tune how your agent behaves, how tools return data, and how that data flows between components.

Here’s how to evolve the basic agent into a more advanced one:

🛠️ Customize the Tool
In the basic example, the SerperDevWebSearch component was turned into a tool with default behavior. For more control, you can:

Add name and description to the tool for the LLM to better understand when to use it.
Convert the tool’s outputs (e.g. Document objects) into strings using outputs_to_string so they can be directly used in prompts.
Save tool outputs into the agent’s internal state using outputs_to_state, making them accessible to future steps in the reasoning process.
from haystack.tools import ComponentTool
from haystack.components.websearch import SerperDevWebSearch

def doc_to_string(documents) -> str:
    result_str = ""
    for document in documents:
        result_str += f"File Content for {document.meta['link']}: {document.content}\n\n"
    return result_str

search_tool = ComponentTool(
    component=SerperDevWebSearch(top_k=5),
    name="web_search", 
    description="Search the web for up-to-date information on any topic",
    outputs_to_string={"source": "documents", "handler": doc_to_string}, # Convert Documents' content into strings before passing it back to the LLM
    outputs_to_state={"documents": {"source": "documents"}}, # Save Documents into Agent's state
)
🧠 Enhance Agent Behavior
The Agent itself can also be configured to handle more advanced tasks:

Custom system_prompt guides the agent’s personality and tool usage strategy.
exit_conditions let you define when the agent should stop reasoning (e.g., once it produces a final text response or calls a listed tool).
state_schema defines the shape of internal memor, e.g., to store retrieved documents between steps.
streaming_callback allows you to stream partial results, trace tool usage, and debug tool-agent interaction live.
from haystack.components.generators.utils import print_streaming_chunk
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, Document

agent = Agent(
    chat_generator=OpenAIChatGenerator(model="gpt-4o-mini"),
    system_prompt="""
    You are a helpful assistant that has access to web. User's ask you questions and you provide answers.
    Use the tools that you're provided with to get information. Don't use your own knowledge.
    Make sure the information that you retrieved is enough to resolve the user query.
    """,
    tools=[search_tool],
    exit_conditions=["text"], # Stop agent execution when there's a text response
    state_schema={"documents":{"type":list[Document]}}, # Define Agent state schema for saved documents 
    streaming_callback=print_streaming_chunk # Display streaming output chunks and print tool calls and tool call results
)

agent_results = agent.run(messages=[ChatMessage.from_user("What are some popular use cases for AI agents?")])
print(agent_results["last_message"].text)

## See the Documents saved in the Agent state
agent_results["documents"]
With just a few lines of configuration, your agent becomes more transparent, stateful, and useful. This flexible design allows you to build powerful multi-step assistants that reason, retrieve, and act intelligently without writing custom orchestration code from scratch.

For a hands-on guide on how to create an tool-calling agent that can use both components and pipelines as tools, see our tutorial.

