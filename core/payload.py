# We will design this using the Builder Pattern. The class will be responsible for taking raw strings, tool data, and system instructions, and assembling them into the exact JSON schema the API demands.

# Class: PayloadBuilder
# Members:
# model (String): The LLM identifier (e.g., "deepseek-ai/DeepSeek-R1").
# messages (List of Dicts): The running list of formatted message payloads.
# tools (List of Dicts): The JSON schema of available tools the model can use.
# temperature / top_p (Floats): Sampling parameters to control determinism.
# Functions (Payload Operations):
# __init__(self, model, tools=None): Initializes the builder with the target model and an optional list of available tool schemas.
# add_system_prompt(self, instructions): Appends the {"role": "system", "content": instructions} dictionary. This must always be at index 0.
# add_user_message(self, prompt): Appends the standard {"role": "user", "content": prompt}.
# add_assistant_message(self, content, tool_calls=None): Appends the assistant's response. If the model decided to call a tool, this function must append the tool_calls array exactly as the model provided it, including the unique id, function.name, and function.arguments.
# add_tool_result(self, tool_call_id, function_name, result_content): This is critical for agentic loops. When your Python code finishes running a tool, you must send the result back to the LLM attached to the exact tool_call_id so the model knows which action was completed. It appends {"role": "tool", "tool_call_id": tool_call_id, "name": function_name, "content": result_content}.
# build(self): The final compilation step. It returns the fully structured dictionary (containing model, messages, tools, and sampling parameters) ready to be passed directly as **kwargs to the Hugging Face InferenceClient.
# truncate_context(self, max_tokens): (Helper Operation) A defensive function that checks the estimated token length of the messages array and removes older conversation turns (while preserving the system prompt and recent tool results) to prevent the API from throwing a context-window limit error.
# How it connects to your workflow
# Instead of your orchestrator (agent.py) manually building dictionaries, the flow becomes:

# agent.py pulls history from session.py.
# agent.py instantiates PayloadBuilder.
# agent.py iterates through the history, using the builder's add_* methods.
# agent.py calls builder.build() and hands that exact dictionary to core.llm_client.

