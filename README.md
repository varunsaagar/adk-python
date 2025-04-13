# Agent Development Kit (ADK)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

<html>
    <h1 align="center">
      <img src="assets/agent-development-kit.png" width="256"/>
    </h1>
    <h3 align="center">
      An open-source, code-first Python toolkit for building, evaluating, and deploying sophisticated AI agents with flexibility and control.
    </h3>
    <h3 align="center">
      Important Links:
      <a href="https://google.github.io/adk-docs/">Docs</a> &
      <a href="https://github.com/google/adk-samples">Samples</a>.
    </h3>
</html>

The Agent Development Kit (ADK) is designed for developers seeking fine-grained control and flexibility when building advanced AI agents that are tightly integrated with services in Google Cloud. It allows you to define agent behavior, orchestration, and tool use directly in code, enabling robust debugging, versioning, and deployment anywhere – from your laptop to the cloud.

---

## ✨ Key Features

* **Code-First Development:** Define agents, tools, and orchestration logic for maximum control, testability, and versioning.
* **Multi-Agent Architecture:** Build modular and scalable applications by composing multiple specialized agents in flexible hierarchies.
* **Rich Tool Ecosystem:** Equip agents with diverse capabilities using pre-built tools, custom Python functions, API specifications, or integrating existing tools.
* **Flexible Orchestration:** Define workflows using built-in agents for predictable pipelines, or leverage LLM-driven dynamic routing for adaptive behavior.
* **Integrated Developer Experience:** Develop, test, and debug locally with a CLI and visual web UI.
* **Built-in Evaluation:** Measure agent performance by evaluating response quality and step-by-step execution trajectory.
* **Deployment Ready:** Containerize and deploy your agents anywhere – scale with Vertex AI Agent Engine, Cloud Run, or Docker.
* **Native Streaming Support:** Build real-time, interactive experiences with native support for bidirectional streaming (text and audio).
* **State, Memory & Artifacts:** Manage short-term conversational context, configure long-term memory, and handle file uploads/downloads.
* **Extensibility:** Customize agent behavior deeply with callbacks and easily integrate third-party tools and services.

## 🚀 Installation

You can install the ADK using `pip`:

```bash
pip install google-adk
```

## 🏁 Getting Started

Create your first agent (`my_agent/agent.py`):

```python
# my_agent/agent.py
from google.adk.agents import Agent
from google.adk.tools import google_search

root_agent = Agent(
    name="search_assistant",
    model="gemini-2.0-flash-exp", # Or your preferred Gemini model
    instruction="You are a helpful assistant. Answer user questions using Google Search when needed.",
    description="An assistant that can search the web.",
    tools=[google_search]
)
```

Create `my_agent/__init__.py`:

```python
# my_agent/__init__.py
from . import agent
```

Run it via the CLI (from the directory *containing* `my_agent`):

```bash
adk run my_agent
```

Or launch the Web UI from the folder that contains `my_agent` folder:

```bash
adk web
```

For a full step-by-step guide, check out the [quickstart](https://google.github.io/adk-docs/get-started/quickstart/) or [sample agents](https://github.com/google/adk-samples).
Model Context Protocol Tools
This guide walks you through two ways of integrating Model Context Protocol (MCP) with ADK.

What is Model Context Protocol (MCP)?
The Model Context Protocol (MCP) is an open standard designed to standardize how Large Language Models (LLMs) like Gemini and Claude communicate with external applications, data sources, and tools. Think of it as a universal connection mechanism that simplifies how LLMs obtain context, execute actions, and interact with various systems.

MCP follows a client-server architecture, defining how data (resources), interactive templates (prompts), and actionable functions (tools) are exposed by an MCP server and consumed by an MCP client (which could be an LLM host application or an AI agent).

This guide covers two primary integration patterns:

Using Existing MCP Servers within ADK: An ADK agent acts as an MCP client, leveraging tools provided by external MCP servers.
Exposing ADK Tools via an MCP Server: Building an MCP server that wraps ADK tools, making them accessible to any MCP client.
Prerequisites
Before you begin, ensure you have the following set up:

Set up ADK: Follow the standard ADK [setup]() instructions in the quickstart.
Install/update Python: MCP requires Python version of 3.9 or higher.
Setup Node.js and npx: Many community MCP servers are distributed as Node.js packages and run using npx. Install Node.js (which includes npx) if you haven't already. For details, see https://nodejs.org/en.
Verify Installations: Confirm adk and npx are in your PATH within the activated virtual environment:

# Both commands should print the path to the executables.
which adk
which npx
1. Using MCP servers with ADK agents (ADK as an MCP client)
This section shows two examples of using MCP servers with ADK agents. This is the most common integration pattern. Your ADK agent needs to use functionality provided by an existing service that exposes itself as an MCP Server.

MCPToolset class
The examples use the MCPToolset class in ADK which acts as the bridge to the MCP server. Your ADK agent uses MCPToolset to:

Connect: Establish a connection to an MCP server process. This can be a local server communicating over standard input/output (StdioServerParameters) or a remote server using Server-Sent Events (SseServerParams).
Discover: Query the MCP server for its available tools (list_tools MCP method).
Adapt: Convert the MCP tool schemas into ADK-compatible BaseTool instances.
Expose: Present these adapted tools to the ADK LlmAgent.
Proxy Calls: When the LlmAgent decides to use one of these tools, MCPToolset forwards the call (call_tool MCP method) to the MCP server and returns the result.
Manage Connection: Handle the lifecycle of the connection to the MCP server process, often requiring explicit cleanup.
Example 1: File System MCP Server
This example demonstrates connecting to a local MCP server that provides file system operations.

Step 1: Attach the MCP Server to your ADK agent via MCPToolset
Create agent.py in ./adk_agent_samples/mcp_agent/ and use the following code snippet to define a function that initializes the MCPToolset.

Important: Replace "/path/to/your/folder" with the absolute path to an actual folder on your system.

# ./adk_agent_samples/mcp_agent/agent.py
import asyncio
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService # Optional
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams, StdioServerParameters

# Load environment variables from .env file in the parent directory
# Place this near the top, before using env vars like API keys
load_dotenv('../.env')

# --- Step 1: Import Tools from MCP Server ---
async def get_tools_async():
  """Gets tools from the File System MCP Server."""
  print("Attempting to connect to MCP Filesystem server...")
  tools, exit_stack = await MCPToolset.from_server(
      # Use StdioServerParameters for local process communication
      connection_params=StdioServerParameters(
          command='npx', # Command to run the server
          args=["-y",    # Arguments for the command
                "@modelcontextprotocol/server-filesystem",
                # TODO: IMPORTANT! Change the path below to an ABSOLUTE path on your system.
                "/path/to/your/folder"],
      )
      # For remote servers, you would use SseServerParams instead:
      # connection_params=SseServerParams(url="http://remote-server:port/path", headers={...})
  )
  print("MCP Toolset created successfully.")
  # MCP requires maintaining a connection to the local MCP Server.
  # exit_stack manages the cleanup of this connection.
  return tools, exit_stack

# --- Step 2: Agent Definition ---
async def get_agent_async():
  """Creates an ADK Agent equipped with tools from the MCP Server."""
  tools, exit_stack = await get_tools_async()
  print(f"Fetched {len(tools)} tools from MCP server.")
  root_agent = LlmAgent(
      model='gemini-2.0-flash', # Adjust model name if needed based on availability
      name='filesystem_assistant',
      instruction='Help user interact with the local filesystem using available tools.',
      tools=tools, # Provide the MCP tools to the ADK agent
  )
  return root_agent, exit_stack

# --- Step 3: Main Execution Logic ---
async def async_main():
  session_service = InMemorySessionService()
  # Artifact service might not be needed for this example
  artifacts_service = InMemoryArtifactService()

  session = session_service.create_session(
      state={}, app_name='mcp_filesystem_app', user_id='user_fs'
  )

  # TODO: Change the query to be relevant to YOUR specified folder.
  # e.g., "list files in the 'documents' subfolder" or "read the file 'notes.txt'"
  query = "list files in the tests folder"
  print(f"User Query: '{query}'")
  content = types.Content(role='user', parts=[types.Part(text=query)])

  root_agent, exit_stack = await get_agent_async()

  runner = Runner(
      app_name='mcp_filesystem_app',
      agent=root_agent,
      artifact_service=artifacts_service, # Optional
      session_service=session_service,
  )

  print("Running agent...")
  events_async = runner.run_async(
      session_id=session.id, user_id=session.user_id, new_message=content
  )

  async for event in events_async:
    print(f"Event received: {event}")

  # Crucial Cleanup: Ensure the MCP server process connection is closed.
  print("Closing MCP server connection...")
  await exit_stack.aclose()
  print("Cleanup complete.")

if __name__ == '__main__':
  try:
    asyncio.run(async_main())
  except Exception as e:
    print(f"An error occurred: {e}")
Step 2: Observe the result
Run the script from the adk_agent_samples directory (ensure your virtual environment is active):


cd ./adk_agent_samples
python3 ./mcp_agent/agent.py
The following shows the expected output for the connection attempt, the MCP server starting (via npx), the ADK agent events (including the FunctionCall to list_directory and the FunctionResponse), and the final agent text response based on the file listing. Ensure the exit_stack.aclose() runs at the end.


User Query: 'list files in the tests folder'
Attempting to connect to MCP Filesystem server...
# --> npx process starts here, potentially logging to stderr/stdout
Secure MCP Filesystem Server running on stdio
Allowed directories: [
  '/path/to/your/folder'
]
# <-- npx process output ends
MCP Toolset created successfully.
Fetched [N] tools from MCP server. # N = number of tools like list_directory, read_file etc.
Running agent...
Event received: content=Content(parts=[Part(..., function_call=FunctionCall(id='...', args={'path': 'tests'}, name='list_directory'), ...)], role='model') ...
Event received: content=Content(parts=[Part(..., function_response=FunctionResponse(id='...', name='list_directory', response={'result': CallToolResult(..., content=[TextContent(...)], ...)}), ...)], role='user') ...
Event received: content=Content(parts=[Part(..., text='https://developers.google.com/maps/get-started#enable-api-sdk')], role='model') ...
Closing MCP server connection...
Cleanup complete.
Example 2: Google Maps MCP Server
This follows the same pattern but targets the Google Maps MCP server.

Step 1: Get API Key and Enable APIs
Follow the directions at Use API keys to get a Google Maps API Key.

Enable Directions API and Routes API in your Google Cloud project. For instructions, see Getting started with Google Maps Platform topic.

Step 2: Update get_tools_async
Modify get_tools_async in agent.py to connect to the Maps server, passing your API key via the env parameter of StdioServerParameters.


# agent.py (modify get_tools_async and other parts as needed)
import asyncio
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import InMemoryArtifactService # Optional
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, SseServerParams, StdioServerParameters

load_dotenv('../.env')

async def get_tools_async():
  """ Step 1: Gets tools from the Google Maps MCP Server."""
  # IMPORTANT: Replace with your actual key
  google_maps_api_key = "YOUR_API_KEY_FROM_STEP_1"
  if "YOUR_API_KEY" in google_maps_api_key:
      raise ValueError("Please replace 'YOUR_API_KEY_FROM_STEP_1' with your actual Google Maps API key.")

  print("Attempting to connect to MCP Google Maps server...")
  tools, exit_stack = await MCPToolset.from_server(
      connection_params=StdioServerParameters(
          command='npx',
          args=["-y",
                "@modelcontextprotocol/server-google-maps",
          ],
          # Pass the API key as an environment variable to the npx process
          env={
              "GOOGLE_MAPS_API_KEY": google_maps_api_key
          }
      )
  )
  print("MCP Toolset created successfully.")
  return tools, exit_stack

# --- Step 2: Agent Definition ---
async def get_agent_async():
  """Creates an ADK Agent equipped with tools from the MCP Server."""
  tools, exit_stack = await get_tools_async()
  print(f"Fetched {len(tools)} tools from MCP server.")
  root_agent = LlmAgent(
      model='gemini-2.0-flash-exp', # Adjust if needed
      name='maps_assistant',
      instruction='Help user with mapping and directions using available tools.',
      tools=tools,
  )
  return root_agent, exit_stack

# --- Step 3: Main Execution Logic (modify query) ---
async def async_main():
  session_service = InMemorySessionService()
  artifacts_service = InMemoryArtifactService() # Optional

  session = session_service.create_session(
      state={}, app_name='mcp_maps_app', user_id='user_maps'
  )

  # TODO: Use specific addresses for reliable results with this server
  query = "What is the route from 1600 Amphitheatre Pkwy to 1165 Borregas Ave"
  print(f"User Query: '{query}'")
  content = types.Content(role='user', parts=[types.Part(text=query)])

  root_agent, exit_stack = await get_agent_async()

  runner = Runner(
      app_name='mcp_maps_app',
      agent=root_agent,
      artifact_service=artifacts_service, # Optional
      session_service=session_service,
  )

  print("Running agent...")
  events_async = runner.run_async(
      session_id=session.id, user_id=session.user_id, new_message=content
  )

  async for event in events_async:
    print(f"Event received: {event}")

  print("Closing MCP server connection...")
  await exit_stack.aclose()
  print("Cleanup complete.")

if __name__ == '__main__':
  try:
    asyncio.run(async_main())
  except Exception as e:
      print(f"An error occurred: {e}")
Step 3: Observe the Result
Run the script from the adk_agent_samples directory (ensure your virtual environment is active):


cd ./adk_agent_samples
python3 ./mcp_agent/agent.py
A successful run will show events indicating the agent called the relevant Google Maps tool (likely related to directions or routes) and a final response containing the directions. An example is shown below.


User Query: 'What is the route from 1600 Amphitheatre Pkwy to 1165 Borregas Ave'
Attempting to connect to MCP Google Maps server...
# --> npx process starts...
MCP Toolset created successfully.
Fetched [N] tools from MCP server.
Running agent...
Event received: content=Content(parts=[Part(..., function_call=FunctionCall(name='get_directions', ...))], role='model') ...
Event received: content=Content(parts=[Part(..., function_response=FunctionResponse(name='get_directions', ...))], role='user') ...
Event received: content=Content(parts=[Part(..., text='Head north toward Amphitheatre Pkwy...')], role='model') ...
Closing MCP server connection...
Cleanup complete.
2. Building an MCP server with ADK tools (MCP server exposing ADK)
This pattern allows you to wrap ADK's tools and make them available to any standard MCP client application. The example in this section exposes the load_web_page ADK tool through the MCP server.

Summary of steps
You will create a standard Python MCP server application using the model-context-protocol library. Within this server, you will:

Instantiate the ADK tool(s) you want to expose (e.g., FunctionTool(load_web_page)).
Implement the MCP server's @app.list_tools handler to advertise the ADK tool(s), converting the ADK tool definition to the MCP schema using adk_to_mcp_tool_type.
Implement the MCP server's @app.call_tool handler to receive requests from MCP clients, identify if the request targets your wrapped ADK tool, execute the ADK tool's .run_async() method, and format the result into an MCP-compliant response (e.g., types.TextContent).
Prerequisites
Install the MCP server library in the same environment as ADK:


pip install mcp
Step 1: Create the MCP Server Script
Create a new Python file, e.g., adk_mcp_server.py.

Step 2: Implement the Server Logic
Add the following code, which sets up an MCP server exposing the ADK load_web_page tool.


# adk_mcp_server.py
import asyncio
import json
from dotenv import load_dotenv

# MCP Server Imports
from mcp import types as mcp_types # Use alias to avoid conflict with genai.types
from mcp.server.lowlevel import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio

# ADK Tool Imports
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.load_web_page import load_web_page # Example ADK tool
# ADK <-> MCP Conversion Utility
from google.adk.tools.mcp_tool.conversion_utils import adk_to_mcp_tool_type

# --- Load Environment Variables (If ADK tools need them) ---
load_dotenv()

# --- Prepare the ADK Tool ---
# Instantiate the ADK tool you want to expose
print("Initializing ADK load_web_page tool...")
adk_web_tool = FunctionTool(load_web_page)
print(f"ADK tool '{adk_web_tool.name}' initialized.")
# --- End ADK Tool Prep ---

# --- MCP Server Setup ---
print("Creating MCP Server instance...")
# Create a named MCP Server instance
app = Server("adk-web-tool-mcp-server")

# Implement the MCP server's @app.list_tools handler
@app.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
  """MCP handler to list available tools."""
  print("MCP Server: Received list_tools request.")
  # Convert the ADK tool's definition to MCP format
  mcp_tool_schema = adk_to_mcp_tool_type(adk_web_tool)
  print(f"MCP Server: Advertising tool: {mcp_tool_schema.name}")
  return [mcp_tool_schema]

# Implement the MCP server's @app.call_tool handler
@app.call_tool()
async def call_tool(
    name: str, arguments: dict
) -> list[mcp_types.TextContent | mcp_types.ImageContent | mcp_types.EmbeddedResource]:
  """MCP handler to execute a tool call."""
  print(f"MCP Server: Received call_tool request for '{name}' with args: {arguments}")

  # Check if the requested tool name matches our wrapped ADK tool
  if name == adk_web_tool.name:
    try:
      # Execute the ADK tool's run_async method
      # Note: tool_context is None as we are not within a full ADK Runner invocation
      adk_response = await adk_web_tool.run_async(
          args=arguments,
          tool_context=None, # No ADK context available here
      )
      print(f"MCP Server: ADK tool '{name}' executed successfully.")
      # Format the ADK tool's response (often a dict) into MCP format.
      # Here, we serialize the response dictionary as a JSON string within TextContent.
      # Adjust formatting based on the specific ADK tool's output and client needs.
      response_text = json.dumps(adk_response, indent=2)
      return [mcp_types.TextContent(type="text", text=response_text)]

    except Exception as e:
      print(f"MCP Server: Error executing ADK tool '{name}': {e}")
      # Return an error message in MCP format
      # Creating a proper MCP error response might be more robust
      error_text = json.dumps({"error": f"Failed to execute tool '{name}': {str(e)}"})
      return [mcp_types.TextContent(type="text", text=error_text)]
  else:
      # Handle calls to unknown tools
      print(f"MCP Server: Tool '{name}' not found.")
      error_text = json.dumps({"error": f"Tool '{name}' not implemented."})
      # Returning error as TextContent for simplicity
      return [mcp_types.TextContent(type="text", text=error_text)]

# --- MCP Server Runner ---
async def run_server():
  """Runs the MCP server over standard input/output."""
  # Use the stdio_server context manager from the MCP library
  async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
    print("MCP Server starting handshake...")
    await app.run(
        read_stream,
        write_stream,
        InitializationOptions(
            server_name=app.name, # Use the server name defined above
            server_version="0.1.0",
            capabilities=app.get_capabilities(
                # Define server capabilities - consult MCP docs for options
                notification_options=NotificationOptions(),
                experimental_capabilities={},
            ),
        ),
    )
    print("MCP Server run loop finished.")

if __name__ == "__main__":
  print("Launching MCP Server exposing ADK tools...")
  try:
    asyncio.run(run_server())
  except KeyboardInterrupt:
    print("\nMCP Server stopped by user.")
  except Exception as e:
    print(f"MCP Server encountered an error: {e}")
  finally:
    print("MCP Server process exiting.")
# --- End MCP Server ---
Step 3: Test your MCP Server with ADK
Follow the same instructions in “Example 1: File System MCP Server” and create a MCP client. This time use your MCP Server file created above as input command:


# ./adk_agent_samples/mcp_agent/agent.py

# ...

async def get_tools_async():
  """Gets tools from the File System MCP Server."""
  print("Attempting to connect to MCP Filesystem server...")
  tools, exit_stack = await MCPToolset.from_server(
      # Use StdioServerParameters for local process communication
      connection_params=StdioServerParameters(
          command='python3', # Command to run the server
          args=[
                "/absolute/path/to/adk_mcp_server.py"],
      )
  )
Execute the agent script from your terminal similar to above (ensure necessary libraries like model-context-protocol and google-adk are installed in your environment):


cd ./adk_agent_samples
python3 ./mcp_agent/agent.py
The script will print startup messages and then wait for an MCP client to connect via its standard input/output to your MCP Server in adk_mcp_server.py. Any MCP-compliant client (like Claude Desktop, or a custom client using the MCP libraries) can now connect to this process, discover the load_web_page tool, and invoke it. The server will print log messages indicating received requests and ADK tool execution. Refer to the documentation, to try it out with Claude Desktop.

Key considerations
When working with MCP and ADK, keep these points in mind:

Protocol vs. Library: MCP is a protocol specification, defining communication rules. ADK is a Python library/framework for building agents. MCPToolset bridges these by implementing the client side of the MCP protocol within the ADK framework. Conversely, building an MCP server in Python requires using the model-context-protocol library.

ADK Tools vs. MCP Tools:

ADK Tools (BaseTool, FunctionTool, AgentTool, etc.) are Python objects designed for direct use within the ADK's LlmAgent and Runner.
MCP Tools are capabilities exposed by an MCP Server according to the protocol's schema. MCPToolset makes these look like ADK tools to an LlmAgent.
Langchain/CrewAI Tools are specific implementations within those libraries, often simple functions or classes, lacking the server/protocol structure of MCP. ADK offers wrappers (LangchainTool, CrewaiTool) for some interoperability.
Asynchronous nature: Both ADK and the MCP Python library are heavily based on the asyncio Python library. Tool implementations and server handlers should generally be async functions.

Stateful sessions (MCP): MCP establishes stateful, persistent connections between a client and server instance. This differs from typical stateless REST APIs.

Deployment: This statefulness can pose challenges for scaling and deployment, especially for remote servers handling many users. The original MCP design often assumed client and server were co-located. Managing these persistent connections requires careful infrastructure considerations (e.g., load balancing, session affinity).
ADK MCPToolset: Manages this connection lifecycle. The exit_stack pattern shown in the examples is crucial for ensuring the connection (and potentially the server process) is properly terminated when the ADK agent finishes.

##LLM Agent
The LlmAgent (often aliased simply as Agent) is a core component in ADK, acting as the "thinking" part of your application. It leverages the power of a Large Language Model (LLM) for reasoning, understanding natural language, making decisions, generating responses, and interacting with tools.

Unlike deterministic Workflow Agents that follow predefined execution paths, LlmAgent behavior is non-deterministic. It uses the LLM to interpret instructions and context, deciding dynamically how to proceed, which tools to use (if any), or whether to transfer control to another agent.

Building an effective LlmAgent involves defining its identity, clearly guiding its behavior through instructions, and equipping it with the necessary tools and capabilities.

Defining the Agent's Identity and Purpose
First, you need to establish what the agent is and what it's for.

name (Required): Every agent needs a unique string identifier. This name is crucial for internal operations, especially in multi-agent systems where agents need to refer to or delegate tasks to each other. Choose a descriptive name that reflects the agent's function (e.g., customer_support_router, billing_inquiry_agent). Avoid reserved names like user.

description (Optional, Recommended for Multi-Agent): Provide a concise summary of the agent's capabilities. This description is primarily used by other LLM agents to determine if they should route a task to this agent. Make it specific enough to differentiate it from peers (e.g., "Handles inquiries about current billing statements," not just "Billing agent").

model (Required): Specify the underlying LLM that will power this agent's reasoning. This is a string identifier like "gemini-2.0-flash-exp". The choice of model impacts the agent's capabilities, cost, and performance. See the Models page for available options and considerations.


# Example: Defining the basic identity
capital_agent = LlmAgent(
    model="gemini-2.0-flash-exp",
    name="capital_agent",
    description="Answers user questions about the capital city of a given country."
    # instruction and tools will be added next
)
Guiding the Agent: Instructions (instruction)
The instruction parameter is arguably the most critical for shaping an LlmAgent's behavior. It's a string (or a function returning a string) that tells the agent:

Its core task or goal.
Its personality or persona (e.g., "You are a helpful assistant," "You are a witty pirate").
Constraints on its behavior (e.g., "Only answer questions about X," "Never reveal Y").
How and when to use its tools. You should explain the purpose of each tool and the circumstances under which it should be called, supplementing any descriptions within the tool itself.
The desired format for its output (e.g., "Respond in JSON," "Provide a bulleted list").
Tips for Effective Instructions:

Be Clear and Specific: Avoid ambiguity. Clearly state the desired actions and outcomes.
Use Markdown: Improve readability for complex instructions using headings, lists, etc.
Provide Examples (Few-Shot): For complex tasks or specific output formats, include examples directly in the instruction.
Guide Tool Use: Don't just list tools; explain when and why the agent should use them.

# Example: Adding instructions
capital_agent = LlmAgent(
    model="gemini-2.0-flash-exp",
    name="capital_agent",
    description="Answers user questions about the capital city of a given country.",
    instruction="""You are an agent that provides the capital city of a country.
When a user asks for the capital of a country:
1. Identify the country name from the user's query.
2. Use the `get_capital_city` tool to find the capital.
3. Respond clearly to the user, stating the capital city.
Example Query: "What's the capital of France?"
Example Response: "The capital of France is Paris."
""",
    # tools will be added next
)
(Note: For instructions that apply to all agents in a system, consider using global_instruction on the root agent, detailed further in the Multi-Agents section.)

Equipping the Agent: Tools (tools)
Tools give your LlmAgent capabilities beyond the LLM's built-in knowledge or reasoning. They allow the agent to interact with the outside world, perform calculations, fetch real-time data, or execute specific actions.

tools (Optional): Provide a list of tools the agent can use. Each item in the list can be:
A Python function (automatically wrapped as a FunctionTool).
An instance of a class inheriting from BaseTool.
An instance of another agent (AgentTool, enabling agent-to-agent delegation - see Multi-Agents).
The LLM uses the function/tool names, descriptions (from docstrings or the description field), and parameter schemas to decide which tool to call based on the conversation and its instructions.


# Define a tool function
def get_capital_city(country: str) -> str:
  """Retrieves the capital city for a given country."""
  # Replace with actual logic (e.g., API call, database lookup)
  capitals = {"france": "Paris", "japan": "Tokyo", "canada": "Ottawa"}
  return capitals.get(country.lower(), f"Sorry, I don't know the capital of {country}.")

# Add the tool to the agent
capital_agent = LlmAgent(
    model="gemini-2.0-flash-exp",
    name="capital_agent",
    description="Answers user questions about the capital city of a given country.",
    instruction="""You are an agent that provides the capital city of a country... (previous instruction text)""",
    tools=[get_capital_city] # Provide the function directly
)
Learn more about Tools in the Tools section.

Advanced Configuration & Control
Beyond the core parameters, LlmAgent offers several options for finer control:

Fine-Tuning LLM Generation (generate_content_config)
You can adjust how the underlying LLM generates responses using generate_content_config.

generate_content_config (Optional): Pass an instance of google.genai.types.GenerateContentConfig to control parameters like temperature (randomness), max_output_tokens (response length), top_p, top_k, and safety settings.


from google.genai import types

agent = LlmAgent(
    # ... other params
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2, # More deterministic output
        max_output_tokens=250
    )
)
Structuring Data (input_schema, output_schema, output_key)
For scenarios requiring structured data exchange, you can use Pydantic models.

input_schema (Optional): Define a Pydantic BaseModel class representing the expected input structure. If set, the user message content passed to this agent must be a JSON string conforming to this schema. Your instructions should guide the user or preceding agent accordingly.

output_schema (Optional): Define a Pydantic BaseModel class representing the desired output structure. If set, the agent's final response must be a JSON string conforming to this schema.

Constraint: Using output_schema enables controlled generation within the LLM but disables the agent's ability to use tools or transfer control to other agents. Your instructions must guide the LLM to produce JSON matching the schema directly.
output_key (Optional): Provide a string key. If set, the text content of the agent's final response will be automatically saved to the session's state dictionary under this key (e.g., session.state[output_key] = agent_response_text). This is useful for passing results between agents or steps in a workflow.


from pydantic import BaseModel, Field

class CapitalOutput(BaseModel):
    capital: str = Field(description="The capital of the country.")

structured_capital_agent = LlmAgent(
    # ... name, model, description
    instruction="""You are a Capital Information Agent. Given a country, respond ONLY with a JSON object containing the capital. Format: {"capital": "capital_name"}""",
    output_schema=CapitalOutput, # Enforce JSON output
    output_key="found_capital"  # Store result in state['found_capital']
    # Cannot use tools=[get_capital_city] effectively here
)
Managing Context (include_contents)
Control whether the agent receives the prior conversation history.

include_contents (Optional, Default: 'default'): Determines if the contents (history) are sent to the LLM.

'default': The agent receives the relevant conversation history.
'none': The agent receives no prior contents. It operates based solely on its current instruction and any input provided in the current turn (useful for stateless tasks or enforcing specific contexts).

stateless_agent = LlmAgent(
    # ... other params
    include_contents='none'
)
Planning & Code Execution
For more complex reasoning involving multiple steps or executing code:

planner (Optional): Assign a BasePlanner instance to enable multi-step reasoning and planning before execution. (See Multi-Agents patterns).
code_executor (Optional): Provide a BaseCodeExecutor instance to allow the agent to execute code blocks (e.g., Python) found in the LLM's response. (See Tools/Built-in tools).
Putting It Together: Example
Code
Here's the complete basic capital_agent:


# Full example code for the basic capital agent
# --- Full example code demonstrating LlmAgent with Tools vs. Output Schema ---
import json # Needed for pretty printing dicts

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel, Field

# --- 1. Define Constants ---
APP_NAME = "agent_comparison_app"
USER_ID = "test_user_456"
SESSION_ID_TOOL_AGENT = "session_tool_agent_xyz"
SESSION_ID_SCHEMA_AGENT = "session_schema_agent_xyz"
MODEL_NAME = "gemini-2.0-flash-exp"

# --- 2. Define Schemas ---

# Input schema used by both agents
class CountryInput(BaseModel):
    country: str = Field(description="The country to get information about.")

# Output schema ONLY for the second agent
class CapitalInfoOutput(BaseModel):
    capital: str = Field(description="The capital city of the country.")
    # Note: Population is illustrative; the LLM will infer or estimate this
    # as it cannot use tools when output_schema is set.
    population_estimate: str = Field(description="An estimated population of the capital city.")

# --- 3. Define the Tool (Only for the first agent) ---
def get_capital_city(country: str) -> str:
    """Retrieves the capital city of a given country."""
    print(f"\n-- Tool Call: get_capital_city(country='{country}') --")
    country_capitals = {
        "united states": "Washington, D.C.",
        "canada": "Ottawa",
        "france": "Paris",
        "japan": "Tokyo",
    }
    result = country_capitals.get(country.lower(), f"Sorry, I couldn't find the capital for {country}.")
    print(f"-- Tool Result: '{result}' --")
    return result

# --- 4. Configure Agents ---

# Agent 1: Uses a tool and output_key
capital_agent_with_tool = LlmAgent(
    model=MODEL_NAME,
    name="capital_agent_tool",
    description="Retrieves the capital city using a specific tool.",
    instruction="""You are a helpful agent that provides the capital city of a country using a tool.
The user will provide the country name in a JSON format like {"country": "country_name"}.
1. Extract the country name.
2. Use the `get_capital_city` tool to find the capital.
3. Respond clearly to the user, stating the capital city found by the tool.
""",
    tools=[get_capital_city],
    input_schema=CountryInput,
    output_key="capital_tool_result", # Store final text response
)

# Agent 2: Uses output_schema (NO tools possible)
structured_info_agent_schema = LlmAgent(
    model=MODEL_NAME,
    name="structured_info_agent_schema",
    description="Provides capital and estimated population in a specific JSON format.",
    instruction=f"""You are an agent that provides country information.
The user will provide the country name in a JSON format like {{"country": "country_name"}}.
Respond ONLY with a JSON object matching this exact schema:
{json.dumps(CapitalInfoOutput.model_json_schema(), indent=2)}
Use your knowledge to determine the capital and estimate the population. Do not use any tools.
""",
    # *** NO tools parameter here - using output_schema prevents tool use ***
    input_schema=CountryInput,
    output_schema=CapitalInfoOutput, # Enforce JSON output structure
    output_key="structured_info_result", # Store final JSON response
)

# --- 5. Set up Session Management and Runners ---
session_service = InMemorySessionService()

# Create separate sessions for clarity, though not strictly necessary if context is managed
session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID_TOOL_AGENT)
session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID_SCHEMA_AGENT)

# Create a runner for EACH agent
capital_runner = Runner(
    agent=capital_agent_with_tool,
    app_name=APP_NAME,
    session_service=session_service
)
structured_runner = Runner(
    agent=structured_info_agent_schema,
    app_name=APP_NAME,
    session_service=session_service
)

# --- 6. Define Agent Interaction Logic ---
async def call_agent_and_print(
    runner_instance: Runner,
    agent_instance: LlmAgent,
    session_id: str,
    query_json: str
):
    """Sends a query to the specified agent/runner and prints results."""
    print(f"\n>>> Calling Agent: '{agent_instance.name}' | Query: {query_json}")

    user_content = types.Content(role='user', parts=[types.Part(text=query_json)])

    final_response_content = "No final response received."
    async for event in runner_instance.run_async(user_id=USER_ID, session_id=session_id, new_message=user_content):
        # print(f"Event: {event.type}, Author: {event.author}") # Uncomment for detailed logging
        if event.is_final_response() and event.content and event.content.parts:
            # For output_schema, the content is the JSON string itself
            final_response_content = event.content.parts[0].text

    print(f"<<< Agent '{agent_instance.name}' Response: {final_response_content}")

    current_session = session_service.get_session(app_name=APP_NAME,
                                                  user_id=USER_ID,
                                                  session_id=session_id)
    stored_output = current_session.state.get(agent_instance.output_key)

    # Pretty print if the stored output looks like JSON (likely from output_schema)
    print(f"--- Session State ['{agent_instance.output_key}']: ", end="")
    try:
        # Attempt to parse and pretty print if it's JSON
        parsed_output = json.loads(stored_output)
        print(json.dumps(parsed_output, indent=2))
    except (json.JSONDecodeError, TypeError):
         # Otherwise, print as string
        print(stored_output)
    print("-" * 30)


# --- 7. Run Interactions ---
async def main():
    print("--- Testing Agent with Tool ---")
    await call_agent_and_print(capital_runner, capital_agent_with_tool, SESSION_ID_TOOL_AGENT, '{"country": "France"}')
    await call_agent_and_print(capital_runner, capital_agent_with_tool, SESSION_ID_TOOL_AGENT, '{"country": "Canada"}')

    print("\n\n--- Testing Agent with Output Schema (No Tool Use) ---")
    await call_agent_and_print(structured_runner, structured_info_agent_schema, SESSION_ID_SCHEMA_AGENT, '{"country": "France"}')
    await call_agent_and_print(structured_runner, structured_info_agent_schema, SESSION_ID_SCHEMA_AGENT, '{"country": "Japan"}')

if __name__ == "__main__":
    await main()
(This example demonstrates the core concepts. More complex agents might incorporate schemas, context control, planning, etc.)

Related Concepts (Deferred Topics)
While this page covers the core configuration of LlmAgent, several related concepts provide more advanced control and are detailed elsewhere:

Callbacks: Intercepting execution points (before/after model calls, before/after tool calls) using before_model_callback, after_model_callback, etc. See Callbacks.
Multi-Agent Control: Advanced strategies for agent interaction, including planning (planner), controlling agent transfer (disallow_transfer_to_parent, disallow_transfer_to_peers), and system-wide instructions (global_instruction). See Multi-Agents.

##Multi-Agent Systems in ADK
As agentic applications grow in complexity, structuring them as a single, monolithic agent can become challenging to develop, maintain, and reason about. The Agent Development Kit (ADK) supports building sophisticated applications by composing multiple, distinct BaseAgent instances into a Multi-Agent System (MAS).

In ADK, a multi-agent system is an application where different agents, often forming a hierarchy, collaborate or coordinate to achieve a larger goal. Structuring your application this way offers significant advantages, including enhanced modularity, specialization, reusability, maintainability, and the ability to define structured control flows using dedicated workflow agents.

You can compose various types of agents derived from BaseAgent to build these systems:

LLM Agents: Agents powered by large language models. (See LLM Agents)
Workflow Agents: Specialized agents (SequentialAgent, ParallelAgent, LoopAgent) designed to manage the execution flow of their sub-agents. (See Workflow Agents)
Custom agents: Your own agents inheriting from BaseAgent with specialized, non-LLM logic. (See Custom Agents)
The following sections detail the core ADK primitives—such as agent hierarchy, workflow agents, and interaction mechanisms—that enable you to construct and manage these multi-agent systems effectively.

2. ADK Primitives for Agent Composition
ADK provides core building blocks—primitives—that enable you to structure and manage interactions within your multi-agent system.

2.1. Agent Hierarchy (parent_agent, sub_agents)
The foundation for structuring multi-agent systems is the parent-child relationship defined in BaseAgent.

Establishing Hierarchy: You create a tree structure by passing a list of agent instances to the sub_agents argument when initializing a parent agent. ADK automatically sets the parent_agent attribute on each child agent during initialization (google.adk.agents.base_agent.py - model_post_init).
Single Parent Rule: An agent instance can only be added as a sub-agent once. Attempting to assign a second parent will result in a ValueError.
Importance: This hierarchy defines the scope for Workflow Agents and influences the potential targets for LLM-Driven Delegation. You can navigate the hierarchy using agent.parent_agent or find descendants using agent.find_agent(name).

# Conceptual Example: Defining Hierarchy
from google.adk.agents import LlmAgent, BaseAgent

# Define individual agents
greeter = LlmAgent(name="Greeter", model="gemini-2.0-flash-exp")
task_doer = BaseAgent(name="TaskExecutor") # Custom non-LLM agent

# Create parent agent and assign children via sub_agents
coordinator = LlmAgent(
    name="Coordinator",
    model="gemini-2.0-flash-exp",
    description="I coordinate greetings and tasks.",
    sub_agents=[ # Assign sub_agents here
        greeter,
        task_doer
    ]
)

# Framework automatically sets:
# assert greeter.parent_agent == coordinator
# assert task_doer.parent_agent == coordinator
2.2. Workflow Agents as Orchestrators
ADK includes specialized agents derived from BaseAgent that don't perform tasks themselves but orchestrate the execution flow of their sub_agents.

SequentialAgent: Executes its sub_agents one after another in the order they are listed.

Context: Passes the same InvocationContext sequentially, allowing agents to easily pass results via shared state.

# Conceptual Example: Sequential Pipeline
from google.adk.agents import SequentialAgent, LlmAgent

step1 = LlmAgent(name="Step1_Fetch", output_key="data") # Saves output to state['data']
step2 = LlmAgent(name="Step2_Process", instruction="Process data from state key 'data'.")

pipeline = SequentialAgent(name="MyPipeline", sub_agents=[step1, step2])
# When pipeline runs, Step2 can access the state['data'] set by Step1.
ParallelAgent: Executes its sub_agents in parallel. Events from sub-agents may be interleaved.

Context: Modifies the InvocationContext.branch for each child agent (e.g., ParentBranch.ChildName), providing a distinct contextual path which can be useful for isolating history in some memory implementations.
State: Despite different branches, all parallel children access the same shared session.state, enabling them to read initial state and write results (use distinct keys to avoid race conditions).

# Conceptual Example: Parallel Execution
from google.adk.agents import ParallelAgent, LlmAgent

fetch_weather = LlmAgent(name="WeatherFetcher", output_key="weather")
fetch_news = LlmAgent(name="NewsFetcher", output_key="news")

gatherer = ParallelAgent(name="InfoGatherer", sub_agents=[fetch_weather, fetch_news])
# When gatherer runs, WeatherFetcher and NewsFetcher run concurrently.
# A subsequent agent could read state['weather'] and state['news'].
LoopAgent: Executes its sub_agents sequentially in a loop.

Termination: The loop stops if the optional max_iterations is reached, or if any sub-agent yields an Event with actions.escalate=True.
Context & State: Passes the same InvocationContext in each iteration, allowing state changes (e.g., counters, flags) to persist across loops.

# Conceptual Example: Loop with Condition
from google.adk.agents import LoopAgent, LlmAgent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator

class CheckCondition(BaseAgent): # Custom agent to check state
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        status = ctx.session.state.get("status", "pending")
        is_done = (status == "completed")
        yield Event(author=self.name, actions=EventActions(escalate=is_done)) # Escalate if done

process_step = LlmAgent(name="ProcessingStep") # Agent that might update state['status']

poller = LoopAgent(
    name="StatusPoller",
    max_iterations=10,
    sub_agents=[process_step, CheckCondition(name="Checker")]
)
# When poller runs, it executes process_step then Checker repeatedly
# until Checker escalates (state['status'] == 'completed') or 10 iterations pass.
2.3. Interaction & Communication Mechanisms
Agents within a system often need to exchange data or trigger actions in one another. ADK facilitates this through:

a) Shared Session State (session.state)
The most fundamental way for agents operating within the same invocation (and thus sharing the same Session object via the InvocationContext) to communicate passively.

Mechanism: One agent (or its tool/callback) writes a value (context.state['data_key'] = processed_data), and a subsequent agent reads it (data = context.state.get('data_key')). State changes are tracked via CallbackContext.
Convenience: The output_key property on LlmAgent automatically saves the agent's final response text (or structured output) to the specified state key.
Nature: Asynchronous, passive communication. Ideal for pipelines orchestrated by SequentialAgent or passing data across LoopAgent iterations.
See Also: State Management

# Conceptual Example: Using output_key and reading state
from google.adk.agents import LlmAgent, SequentialAgent

agent_A = LlmAgent(name="AgentA", instruction="Find the capital of France.", output_key="capital_city")
agent_B = LlmAgent(name="AgentB", instruction="Tell me about the city stored in state key 'capital_city'.")

pipeline = SequentialAgent(name="CityInfo", sub_agents=[agent_A, agent_B])
# AgentA runs, saves "Paris" to state['capital_city'].
# AgentB runs, its instruction processor reads state['capital_city'] to get "Paris".
b) LLM-Driven Delegation (Agent Transfer)
Leverages an LlmAgent's understanding to dynamically route tasks to other suitable agents within the hierarchy.

Mechanism: The agent's LLM generates a specific function call: transfer_to_agent(agent_name='target_agent_name').
Handling: The AutoFlow, used by default when sub-agents are present or transfer isn't disallowed, intercepts this call. It identifies the target agent using root_agent.find_agent() and updates the InvocationContext to switch execution focus.
Requires: The calling LlmAgent needs clear instructions on when to transfer, and potential target agents need distinct descriptions for the LLM to make informed decisions. Transfer scope (parent, sub-agent, siblings) can be configured on the LlmAgent.
Nature: Dynamic, flexible routing based on LLM interpretation.

# Conceptual Setup: LLM Transfer
from google.adk.agents import LlmAgent

booking_agent = LlmAgent(name="Booker", description="Handles flight and hotel bookings.")
info_agent = LlmAgent(name="Info", description="Provides general information and answers questions.")

coordinator = LlmAgent(
    name="Coordinator",
    instruction="You are an assistant. Delegate booking tasks to Booker and info requests to Info.",
    description="Main coordinator.",
    # AutoFlow is typically used implicitly here
    sub_agents=[booking_agent, info_agent]
)
# If coordinator receives "Book a flight", its LLM should generate:
# FunctionCall(name='transfer_to_agent', args={'agent_name': 'Booker'})
# ADK framework then routes execution to booking_agent.
c) Explicit Invocation (AgentTool)
Allows an LlmAgent to treat another BaseAgent instance as a callable function or Tool.

Mechanism: Wrap the target agent instance in AgentTool and include it in the parent LlmAgent's tools list. AgentTool generates a corresponding function declaration for the LLM.
Handling: When the parent LLM generates a function call targeting the AgentTool, the framework executes AgentTool.run_async. This method runs the target agent, captures its final response, forwards any state/artifact changes back to the parent's context, and returns the response as the tool's result.
Nature: Synchronous (within the parent's flow), explicit, controlled invocation like any other tool.
(Note: AgentTool needs to be imported and used explicitly).

# Conceptual Setup: Agent as a Tool
from google.adk.agents import LlmAgent, BaseAgent
from google.adk.tools import AgentTool # Assuming AgentTool exists
from pydantic import BaseModel

# Define a target agent (could be LlmAgent or custom BaseAgent)
class ImageGeneratorAgent(BaseAgent): # Example custom agent
    name: str = "ImageGen"
    description: str = "Generates an image based on a prompt."
    # ... internal logic ...
    async def _run_async_impl(self, ctx): # Simplified run logic
        prompt = ctx.session.state.get("image_prompt", "default prompt")
        # ... generate image bytes ...
        image_bytes = b"..."
        yield Event(author=self.name, content=types.Content(parts=[types.Part.from_bytes(image_bytes, "image/png")]))

image_agent = ImageGeneratorAgent()
image_tool = AgentTool(agent=image_agent) # Wrap the agent

# Parent agent uses the AgentTool
artist_agent = LlmAgent(
    name="Artist",
    model="gemini-2.0-flash-exp",
    instruction="Create a prompt and use the ImageGen tool to generate the image.",
    tools=[image_tool] # Include the AgentTool
)
# Artist LLM generates a prompt, then calls:
# FunctionCall(name='ImageGen', args={'image_prompt': 'a cat wearing a hat'})
# Framework calls image_tool.run_async(...), which runs ImageGeneratorAgent.
# The resulting image Part is returned to the Artist agent as the tool result.
These primitives provide the flexibility to design multi-agent interactions ranging from tightly coupled sequential workflows to dynamic, LLM-driven delegation networks.

3. Common Multi-Agent Patterns using ADK Primitives
By combining ADK's composition primitives, you can implement various established patterns for multi-agent collaboration.

Coordinator/Dispatcher Pattern
Structure: A central LlmAgent (Coordinator) manages several specialized sub_agents.
Goal: Route incoming requests to the appropriate specialist agent.
ADK Primitives Used:
Hierarchy: Coordinator has specialists listed in sub_agents.
Interaction: Primarily uses LLM-Driven Delegation (requires clear descriptions on sub-agents and appropriate instruction on Coordinator) or Explicit Invocation (AgentTool) (Coordinator includes AgentTool-wrapped specialists in its tools).

# Conceptual Code: Coordinator using LLM Transfer
from google.adk.agents import LlmAgent

billing_agent = LlmAgent(name="Billing", description="Handles billing inquiries.")
support_agent = LlmAgent(name="Support", description="Handles technical support requests.")

coordinator = LlmAgent(
    name="HelpDeskCoordinator",
    model="gemini-2.0-flash-exp",
    instruction="Route user requests: Use Billing agent for payment issues, Support agent for technical problems.",
    description="Main help desk router.",
    # allow_transfer=True is often implicit with sub_agents in AutoFlow
    sub_agents=[billing_agent, support_agent]
)
# User asks "My payment failed" -> Coordinator's LLM should call transfer_to_agent(agent_name='Billing')
# User asks "I can't log in" -> Coordinator's LLM should call transfer_to_agent(agent_name='Support')
Sequential Pipeline Pattern
Structure: A SequentialAgent contains sub_agents executed in a fixed order.
Goal: Implement a multi-step process where the output of one step feeds into the next.
ADK Primitives Used:
Workflow: SequentialAgent defines the order.
Communication: Primarily uses Shared Session State. Earlier agents write results (often via output_key), later agents read those results from context.state.

# Conceptual Code: Sequential Data Pipeline
from google.adk.agents import SequentialAgent, LlmAgent

validator = LlmAgent(name="ValidateInput", instruction="Validate the input.", output_key="validation_status")
processor = LlmAgent(name="ProcessData", instruction="Process data if state key 'validation_status' is 'valid'.", output_key="result")
reporter = LlmAgent(name="ReportResult", instruction="Report the result from state key 'result'.")

data_pipeline = SequentialAgent(
    name="DataPipeline",
    sub_agents=[validator, processor, reporter]
)
# validator runs -> saves to state['validation_status']
# processor runs -> reads state['validation_status'], saves to state['result']
# reporter runs -> reads state['result']
Parallel Fan-Out/Gather Pattern
Structure: A ParallelAgent runs multiple sub_agents concurrently, often followed by a later agent (in a SequentialAgent) that aggregates results.
Goal: Execute independent tasks simultaneously to reduce latency, then combine their outputs.
ADK Primitives Used:
Workflow: ParallelAgent for concurrent execution (Fan-Out). Often nested within a SequentialAgent to handle the subsequent aggregation step (Gather).
Communication: Sub-agents write results to distinct keys in Shared Session State. The subsequent "Gather" agent reads multiple state keys.

# Conceptual Code: Parallel Information Gathering
from google.adk.agents import SequentialAgent, ParallelAgent, LlmAgent

fetch_api1 = LlmAgent(name="API1Fetcher", instruction="Fetch data from API 1.", output_key="api1_data")
fetch_api2 = LlmAgent(name="API2Fetcher", instruction="Fetch data from API 2.", output_key="api2_data")

gather_concurrently = ParallelAgent(
    name="ConcurrentFetch",
    sub_agents=[fetch_api1, fetch_api2]
)

synthesizer = LlmAgent(
    name="Synthesizer",
    instruction="Combine results from state keys 'api1_data' and 'api2_data'."
)

overall_workflow = SequentialAgent(
    name="FetchAndSynthesize",
    sub_agents=[gather_concurrently, synthesizer] # Run parallel fetch, then synthesize
)
# fetch_api1 and fetch_api2 run concurrently, saving to state.
# synthesizer runs afterwards, reading state['api1_data'] and state['api2_data'].
Hierarchical Task Decomposition
Structure: A multi-level tree of agents where higher-level agents break down complex goals and delegate sub-tasks to lower-level agents.
Goal: Solve complex problems by recursively breaking them down into simpler, executable steps.
ADK Primitives Used:
Hierarchy: Multi-level parent_agent/sub_agents structure.
Interaction: Primarily LLM-Driven Delegation or Explicit Invocation (AgentTool) used by parent agents to assign tasks to children. Results are returned up the hierarchy (via tool responses or state).

# Conceptual Code: Hierarchical Research Task
from google.adk.agents import LlmAgent
from google.adk.tools import AgentTool # Assuming AgentTool exists

# Low-level tool-like agents
web_searcher = LlmAgent(name="WebSearch", description="Performs web searches for facts.")
summarizer = LlmAgent(name="Summarizer", description="Summarizes text.")

# Mid-level agent combining tools
research_assistant = LlmAgent(
    name="ResearchAssistant",
    model="gemini-2.0-flash-exp",
    description="Finds and summarizes information on a topic.",
    tools=[AgentTool(agent=web_searcher), AgentTool(agent=summarizer)]
)

# High-level agent delegating research
report_writer = LlmAgent(
    name="ReportWriter",
    model="gemini-2.0-flash-exp",
    instruction="Write a report on topic X. Use the ResearchAssistant to gather information.",
    tools=[AgentTool(agent=research_assistant)]
    # Alternatively, could use LLM Transfer if research_assistant is a sub_agent
)
# User interacts with ReportWriter.
# ReportWriter calls ResearchAssistant tool.
# ResearchAssistant calls WebSearch and Summarizer tools.
# Results flow back up.
Review/Critique Pattern (Generator-Critic)
Structure: Typically involves two agents within a SequentialAgent: a Generator and a Critic/Reviewer.
Goal: Improve the quality or validity of generated output by having a dedicated agent review it.
ADK Primitives Used:
Workflow: SequentialAgent ensures generation happens before review.
Communication: Shared Session State (Generator uses output_key to save output; Reviewer reads that state key). The Reviewer might save its feedback to another state key for subsequent steps.

# Conceptual Code: Generator-Critic
from google.adk.agents import SequentialAgent, LlmAgent

generator = LlmAgent(
    name="DraftWriter",
    instruction="Write a short paragraph about subject X.",
    output_key="draft_text"
)

reviewer = LlmAgent(
    name="FactChecker",
    instruction="Review the text in state key 'draft_text' for factual accuracy. Output 'valid' or 'invalid' with reasons.",
    output_key="review_status"
)

# Optional: Further steps based on review_status

review_pipeline = SequentialAgent(
    name="WriteAndReview",
    sub_agents=[generator, reviewer]
)
# generator runs -> saves draft to state['draft_text']
# reviewer runs -> reads state['draft_text'], saves status to state['review_status']
Iterative Refinement Pattern
Structure: Uses a LoopAgent containing one or more agents that work on a task over multiple iterations.
Goal: Progressively improve a result (e.g., code, text, plan) stored in the session state until a quality threshold is met or a maximum number of iterations is reached.
ADK Primitives Used:
Workflow: LoopAgent manages the repetition.
Communication: Shared Session State is essential for agents to read the previous iteration's output and save the refined version.
Termination: The loop typically ends based on max_iterations or a dedicated checking agent setting actions.escalate=True when the result is satisfactory.

# Conceptual Code: Iterative Code Refinement
from google.adk.agents import LoopAgent, LlmAgent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator

# Agent to generate/refine code based on state['current_code'] and state['requirements']
code_refiner = LlmAgent(
    name="CodeRefiner",
    instruction="Read state['current_code'] (if exists) and state['requirements']. Generate/refine Python code to meet requirements. Save to state['current_code'].",
    output_key="current_code" # Overwrites previous code in state
)

# Agent to check if the code meets quality standards
quality_checker = LlmAgent(
    name="QualityChecker",
    instruction="Evaluate the code in state['current_code'] against state['requirements']. Output 'pass' or 'fail'.",
    output_key="quality_status"
)

# Custom agent to check the status and escalate if 'pass'
class CheckStatusAndEscalate(BaseAgent):
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        status = ctx.session.state.get("quality_status", "fail")
        should_stop = (status == "pass")
        yield Event(author=self.name, actions=EventActions(escalate=should_stop))

refinement_loop = LoopAgent(
    name="CodeRefinementLoop",
    max_iterations=5,
    sub_agents=[code_refiner, quality_checker, CheckStatusAndEscalate(name="StopChecker")]
)
# Loop runs: Refiner -> Checker -> StopChecker
# State['current_code'] is updated each iteration.
# Loop stops if QualityChecker outputs 'pass' (leading to StopChecker escalating) or after 5 iterations.
Human-in-the-Loop Pattern
Structure: Integrates human intervention points within an agent workflow.
Goal: Allow for human oversight, approval, correction, or tasks that AI cannot perform.
ADK Primitives Used (Conceptual):
Interaction: Can be implemented using a custom Tool that pauses execution and sends a request to an external system (e.g., a UI, ticketing system) waiting for human input. The tool then returns the human's response to the agent.
Workflow: Could use LLM-Driven Delegation (transfer_to_agent) targeting a conceptual "Human Agent" that triggers the external workflow, or use the custom tool within an LlmAgent.
State/Callbacks: State can hold task details for the human; callbacks can manage the interaction flow.
Note: ADK doesn't have a built-in "Human Agent" type, so this requires custom integration.

# Conceptual Code: Using a Tool for Human Approval
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import FunctionTool

# --- Assume external_approval_tool exists ---
# This tool would:
# 1. Take details (e.g., request_id, amount, reason).
# 2. Send these details to a human review system (e.g., via API).
# 3. Poll or wait for the human response (approved/rejected).
# 4. Return the human's decision.
# async def external_approval_tool(amount: float, reason: str) -> str: ...
approval_tool = FunctionTool(func=external_approval_tool)

# Agent that prepares the request
prepare_request = LlmAgent(
    name="PrepareApproval",
    instruction="Prepare the approval request details based on user input. Store amount and reason in state.",
    # ... likely sets state['approval_amount'] and state['approval_reason'] ...
)

# Agent that calls the human approval tool
request_approval = LlmAgent(
    name="RequestHumanApproval",
    instruction="Use the external_approval_tool with amount from state['approval_amount'] and reason from state['approval_reason'].",
    tools=[approval_tool],
    output_key="human_decision"
)

# Agent that proceeds based on human decision
process_decision = LlmAgent(
    name="ProcessDecision",
    instruction="Check state key 'human_decision'. If 'approved', proceed. If 'rejected', inform user."
)

approval_workflow = SequentialAgent(
    name="HumanApprovalWorkflow",
    sub_agents=[prepare_request, request_approval, process_decision]
)
These patterns provide starting points for structuring your multi-agent systems. You can mix and match them as needed to create the most effective architecture for your specific application.

##Function tools
What are function tools?
When out-of-the-box tools don't fully meet specific requirements, developers can create custom function tools. This allows for tailored functionality, such as connecting to proprietary databases or implementing unique algorithms.

For example, a function tool, "myfinancetool", might be a function that calculates a specific financial metric. ADK also supports long running functions, so if that calculation takes a while, the agent can continue working on other tasks.

ADK offers several ways to create functions tools, each suited to different levels of complexity and control:

Function Tool
Long Running Function Tool
Agents-as-a-Tool
1. Function Tool
Transforming a function into a tool is a straightforward way to integrate custom logic into your agents. This approach offers flexibility and quick integration.

Parameters
Define your function parameters using standard JSON-serializable types (e.g., string, integer, list, dictionary). It's important to avoid setting default values for parameters, as the language model (LLM) does not currently support interpreting them.

Return Type
The preferred return type for a Python Function Tool is a dictionary. This allows you to structure the response with key-value pairs, providing context and clarity to the LLM. If your function returns a type other than a dictionary, the framework automatically wraps it into a dictionary with a single key named "result".

Strive to make your return values as descriptive as possible. For example, instead of returning a numeric error code, return a dictionary with an "error_message" key containing a human-readable explanation. Remember that the LLM, not a piece of code, needs to understand the result. As a best practice, include a "status" key in your return dictionary to indicate the overall outcome (e.g., "success", "error", "pending"), providing the LLM with a clear signal about the operation's state.

Docstring
The docstring of your function serves as the tool's description and is sent to the LLM. Therefore, a well-written and comprehensive docstring is crucial for the LLM to understand how to use the tool effectively. Clearly explain the purpose of the function, the meaning of its parameters, and the expected return values.

Example
Best Practices
While you have considerable flexibility in defining your function, remember that simplicity enhances usability for the LLM. Consider these guidelines:

Fewer Parameters are Better: Minimize the number of parameters to reduce complexity.
Simple Data Types: Favor primitive data types like str and int over custom classes whenever possible.
Meaningful Names: The function's name and parameter names significantly influence how the LLM interprets and utilizes the tool. Choose names that clearly reflect the function's purpose and the meaning of its inputs. Avoid generic names like do_stuff().
2. Long Running Function Tool
Designed for tasks that require a significant amount of processing time without blocking the agent's execution. This tool is a subclass of FunctionTool.

When using a LongRunningFunctionTool, your Python function can initiate the long-running operation and optionally return an intermediate result to keep the model and user informed about the progress. The agent can then continue with other tasks. An example is the human-in-the-loop scenario where the agent needs human approval before proceeding with a task.

How it Works
You wrap a Python generator function (a function using yield) with LongRunningFunctionTool.

Initiation: When the LLM calls the tool, your generator function starts executing.

Intermediate Updates (yield): Your function should yield intermediate Python objects (typically dictionaries) periodically to report progress. The ADK framework takes each yielded value and sends it back to the LLM packaged within a FunctionResponse. This allows the LLM to inform the user (e.g., status, percentage complete, messages).

Completion (return): When the task is finished, the generator function uses return to provide the final Python object result.

Framework Handling: The ADK framework manages the execution. It sends each yielded value back as an intermediate FunctionResponse. When the generator completes, the framework sends the returned value as the content of the final FunctionResponse, signaling the end of the long-running operation to the LLM.

Creating the Tool
Define your generator function and wrap it using the LongRunningFunctionTool class:


from google.adk.tools import LongRunningFunctionTool

# Define your generator function (see example below)
def my_long_task_generator(*args, **kwargs):
    # ... setup ...
    yield {"status": "pending", "message": "Starting task..."} # Framework sends this as FunctionResponse
    # ... perform work incrementally ...
    yield {"status": "pending", "progress": 50}               # Framework sends this as FunctionResponse
    # ... finish work ...
    return {"status": "completed", "result": "Final outcome"} # Framework sends this as final FunctionResponse

# Wrap the function
my_tool = LongRunningFunctionTool(func=my_long_task_generator)
Intermediate Updates
Yielding structured Python objects (like dictionaries) is crucial for providing meaningful updates. Include keys like:

status: e.g., "pending", "running", "waiting_for_input"

progress: e.g., percentage, steps completed

message: Descriptive text for the user/LLM

estimated_completion_time: If calculable

Each value you yield is packaged into a FunctionResponse by the framework and sent to the LLM.

Final Result
The Python object your generator function returns is considered the final result of the tool execution. The framework packages this value (even if it's None) into the content of the final FunctionResponse sent back to the LLM, indicating the tool execution is complete.

Example: File Processing Simulation
Key aspects of this example
process_large_file: This generator simulates a lengthy operation, yielding intermediate status/progress dictionaries.

LongRunningFunctionTool: Wraps the generator; the framework handles sending yielded updates and the final return value as sequential FunctionResponses.

Agent instruction: Directs the LLM to use the tool and understand the incoming FunctionResponse stream (progress vs. completion) for user updates.

Final return: The function returns the final result dictionary, which is sent in the concluding FunctionResponse to indicate completion.

3. Agent-as-a-Tool
This powerful feature allows you to leverage the capabilities of other agents within your system by calling them as tools. The Agent-as-a-Tool enables you to invoke another agent to perform a specific task, effectively delegating responsibility. This is conceptually similar to creating a Python function that calls another agent and uses the agent's response as the function's return value.

Key difference from sub-agents
It's important to distinguish an Agent-as-a-Tool from a Sub-Agent.

Agent-as-a-Tool: When Agent A calls Agent B as a tool (using Agent-as-a-Tool), Agent B's answer is passed back to Agent A, which then summarizes the answer and generates a response to the user. Agent A retains control and continues to handle future user input.

Sub-agent: When Agent A calls Agent B as a sub-agent, the responsibility of answering the user is completely transferred to Agent B. Agent A is effectively out of the loop. All subsequent user input will be answered by Agent B.

Usage
To use an agent as a tool, wrap the agent with the AgentTool class.


tools=[AgentTool(agent=agent_b)]
Customization
The AgentTool class provides the following attributes for customizing its behavior:

skip_summarization: bool: If set to True, the framework will bypass the LLM-based summarization of the tool agent's response. This can be useful when the tool's response is already well-formatted and requires no further processing.
Example

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

APP_NAME="summary_agent"
USER_ID="user1234"
SESSION_ID="1234"

summary_agent = Agent(
    model="gemini-2.0-flash-exp",
    name="summary_agent",
    instruction="""You are an expert summarizer. Please read the following text and provide a concise summary.""",
    description="Agent to summarize text",
)

root_agent = Agent(
    model='gemini-2.0-flash-exp',
    name='root_agent',
    instruction="""You are a helpful assistant. When the user provides a long text, use the 'summarize' tool to get a summary and then present it to the user.""",
    tools=[AgentTool(agent=summary_agent)]
)

# Session and Runner
session_service = InMemorySessionService()
session = session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
runner = Runner(agent=summary_agent, app_name=APP_NAME, session_service=session_service)


# Agent Interaction
def call_agent(query):
    content = types.Content(role='user', parts=[types.Part(text=query)])
    events = runner.run(user_id=USER_ID, session_id=SESSION_ID, new_message=content)

    for event in events:
        if event.is_final_response():
            final_response = event.content.parts[0].text
            print("Agent Response: ", final_response)


long_text = """Quantum computing represents a fundamentally different approach to computation, 
leveraging the bizarre principles of quantum mechanics to process information. Unlike classical computers 
that rely on bits representing either 0 or 1, quantum computers use qubits which can exist in a state of superposition - effectively 
being 0, 1, or a combination of both simultaneously. Furthermore, qubits can become entangled, 
meaning their fates are intertwined regardless of distance, allowing for complex correlations. This parallelism and 
interconnectedness grant quantum computers the potential to solve specific types of incredibly complex problems - such 
as drug discovery, materials science, complex system optimization, and breaking certain types of cryptography - far 
faster than even the most powerful classical supercomputers could ever achieve, although the technology is still largely in its developmental stages."""


call_agent(long_text)
How it works
When the main_agent receives the long text, its instruction tells it to use the 'summarize' tool for long texts.
The framework recognizes 'summarize' as an AgentTool that wraps the summary_agent.
Behind the scenes, the main_agent will call the summary_agent with the long text as input.
The summary_agent will process the text according to its instruction and generate a summary.
The response from the summary_agent is then passed back to the main_agent.
The main_agent can then take the summary and formulate its final response to the user (e.g., "Here's a summary of the text: ...")

## 📚 Resources

Explore the full documentation for detailed guides on building, evaluating, and deploying agents:

*   **[Get Started](https://google.github.io/adk-docs/get-started/)**
*   **[Browse Sample Agents](https://github.com/google/adk-samples)**
*   **[Evaluate Agents](https://google.github.io/adk-docs/evaluate/)**
*   **[Deploy Agents](https://google.github.io/adk-docs/deploy/)**
*   **[API Reference](https://google.github.io/adk-docs/api-reference/)**

## 🤝 Contributing

We welcome contributions from the community! Whether it's bug reports, feature requests, documentation improvements, or code contributions, please see our [**Contributing Guidelines**](./CONTRIBUTING.md) to get started.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

*Happy Agent Building!*
