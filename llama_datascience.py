#!/usr/bin/env python3
from llm_sandbox import ArtifactSandboxSession
import base64
from pathlib import Path
import ollama
import argparse
import json
import sys
import asyncio
from termcolor import cprint
import time

# --- System Prompts ---
EXPERT_SYSTEM_PROMPT = """You are a senior data scientist with expertise in Python programming and data analysis. Your task is to answer the users questions by executing Python code.

**Objective:** Assist the user in achieving their data analysis goals by executing Python, **with emphasis on avoiding assumptions and ensuring accuracy.**
Reaching that goal can involve **multiple steps**, executing Python code and reviewing the output multiple times to get to the final answer.

**Output Visibility:** **All output** must be returned with `print`. For example to display the first few rows of the DataFrame to understand its structure use `print(df.head())`. Or to return a computed variable use `print(f'{{variable=}}')`.

**Isolated and Temporary:** Each call to execute Python is in a temporary isolated environment and must be complete, including importing libraries, defining variables and loading data. All data, variable, and state will be lost after each execution. You must re-import libraries and re-read data files each time you execute Python code.

**Libraries:** You may use the following Python libraries, which are already installed:
- io
- math
- re
- matplotlib
- numpy
- pandas
- scipy

**Data in files:** The user's data will be in the file '/sandbox/data.csv'. Do not use any other files.

**No Assumptions:** **Crucially, avoid making assumptions about the nature of the data or column names.** Base findings solely on the data itself. Always use the information obtained from `explore_df` to guide your analysis.

**Answerability:** Some queries may not be answerable with the available data. In those cases, inform the user why you cannot process their query and suggest what type of data would be needed to fulfill their request.

**Data in prompt:** Some queries contain the input data directly in the prompt. You have to parse that data into a pandas DataFrame. ALWAYS parse all the data. NEVER edit the data that are given to you.

**Data Science:** All information about data should come from executing Python. You should **never** generate data outputs yourself.
"""

async def python_datascience(code: str, data_file: str, execution_timeout: int = 3000) -> str:
    # Clean up the code by removing the markdown code block
    if code.startswith("```python"):
        code = code[len("```python"):]
    if code.endswith("```"):
        code = code[:-len("```")]
    code = code.strip()

    with ArtifactSandboxSession(
            lang="python",
            execution_timeout=execution_timeout,
            session_timeout=execution_timeout * 10
        ) as session:
        session.copy_to_runtime(data_file, "/sandbox/data.csv")
        result = session.run(code, timeout=600, libraries=["matplotlib", "numpy", "io", "pandas", "scipy", "re", "math"])

        if result.exit_code == 0:
            for i, plot in enumerate(result.plots):
                plot_path = Path(f"plot_{i + 1}.{plot.format.value}")
                with plot_path.open("wb") as f:
                    f.write(base64.b64decode(plot.content_base64))
            return f"output:\n{result.stdout[result.stdout.find('\n', result.stdout.find('\n') + 1) + 1:]}"
        
        return f"error:\n{result.stderr}"

class ExpertSystem:
    def __init__(self, expert_config: tuple[str, float], max_reasoning_steps: int):
        self.expert_model, self.expert_temp = expert_config
        self.client = ollama.AsyncClient()
        self.max_reasoning_steps = max_reasoning_steps
        cprint(f"Expert: {self.expert_model} (temp: {self.expert_temp})", "green", file=sys.stderr)
        self.tools = [
            {
                'type': 'function',
                'function': {
                    'name': 'python_datascience',
                    'description': "Execute Python code for data science tasks using the provided code and data file.",
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'code': {'type': 'string', 'description': 'The Python code to execute.'},
                            'reason': {'type': 'string', 'description': 'Why you chose to execute this code.'}
                        },
                        'required': ['code', 'reason']
                    }
                }
            }
        ]
    
    async def _python_datascience(self, code: str, data_file: str, execution_timeout: int = 3000) -> str:
        raw_output = await python_datascience(code, data_file, execution_timeout)
        cprint(f"> Code execution complete.\n{raw_output}", "blue", file=sys.stderr)
        return raw_output

    async def get_expert_answer(self, user_prompt: str, data_file: str = None, verbose: bool = False, thinking: bool = False, write_convo: bool = False, prior_conversation: list = None, execution_timeout: int = None):
        if prior_conversation:
            cprint(f"\n> Starting from prior conversation context", "blue", file=sys.stderr)
            conversation_history = prior_conversation
            conversation_history.append({'role': 'user', 'content': user_prompt})
        else: 
            conversation_history = [{'role': 'system', 'content': EXPERT_SYSTEM_PROMPT}, {'role': 'user', 'content': user_prompt}]
        
        for i in range(self.max_reasoning_steps):
            cprint(f"\n--- Expert Reasoning Step {i+1}/{self.max_reasoning_steps} ---", "magenta", file=sys.stderr)
            response = await self.client.chat(
                model=self.expert_model,
                messages=conversation_history,
                tools=self.tools,
                options={'temperature': self.expert_temp, 'think': thinking, 'verbose': verbose}
            )
            response_message = response['message']
            if verbose and 'thinking' in response_message:
                print(f"\n--- Expert Thinking ---\n{response_message['thinking']}\n")
            
            if not response_message.get('tool_calls'):
                conversation_history.append({
                        'role': "system",
                        'content': f"'{response_message}'"})
                if response_message.get('content'):
                    print(response_message.get('content'))
                    if write_convo:
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        fname = f"llama-panel-{timestamp}.convo"
                        with open(fname, "w") as f:
                            json.dump(conversation_history, f, indent=2)
                        cprint(f"\nConversation history written to {fname}", "green", file=sys.stderr)
                else:
                    print(response)
                return conversation_history

            for tool_call in response_message['tool_calls']:
                function_call = tool_call['function']
                tool_name = function_call['name']
                tool_args = function_call['arguments']
                tool_reason = tool_args.get("reason", '')

                cprint(f"> Expert wants to use tool: '{tool_name}'", "yellow", file=sys.stderr)
                cprint(f"> Reason for tool selection: {tool_reason}", "yellow", file=sys.stderr)

                tool_output = None
                if tool_name == "python_datascience":
                    code = tool_args.get('code', '')
                    cprint(f"> Code:\n{code}", "yellow", file=sys.stderr)
                    tool_output = await self._python_datascience(code, data_file)
                    conversation_history.append({
                        'role': "system",
                        'content': f"Executed code to '{tool_reason}'\nCode:\n{code}\nOutput:\n{tool_output}"})
                else:
                    cprint(f"Error: Expert called an unknown tool: {tool_name}", "red", file=sys.stderr)
                    print(response)
                    return
        
        cprint("\n--- Max Reasoning Steps Reached ---", "red", attrs=["bold"], file=sys.stderr)

def parse_model_temp(value: str) -> tuple[str, float]:
    try:
        parts = value.rsplit(':', 1)
        if len(parts) == 2 and parts[1].replace('.', '', 1).isdigit():
            return parts[0], float(parts[1])
        raise ValueError()
    except (ValueError, IndexError):
        raise argparse.ArgumentTypeError(f"Invalid format: '{value}'. Use 'model_name:temperature'.")

async def main():
    parser = argparse.ArgumentParser(description="Llama Data Science: Chat with an AI data scientist.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("question", type=str, nargs='?', default=None, help="A single question to ask the panel.")
    parser.add_argument("--data-file", type=str, default=None, help="Path to the data file to use for analysis.")
    parser.add_argument("--expert", type=parse_model_temp, default='gpt-oss:0.5', help="Expert model. Format: 'model:temp'.")
    parser.add_argument("--max-steps", type=int, default=20, help="Maximum reasoning steps for the expert system.")
    parser.add_argument("--verbose", action="store_true", help="Print expert and panel thinking to stdout.")
    parser.add_argument("--thinking", action="store_true", help="Enable expert model thinking output if supported.")
    parser.add_argument("--write-convo", action="store_true", help="Write conversation history to a file after final answer.")
    parser.add_argument("--execution-timeout", type=int, default=600, help="Timeout for Python code execution in seconds.")
    args = parser.parse_args()

    try:
        system = ExpertSystem(args.expert, args.max_steps)
        if args.question:
            await system.get_expert_answer(args.question, args.data_file, args.verbose, args.thinking, args.write_convo, execution_timeout=args.execution_timeout)
        else:
            conversation_history = []
            cprint("\nWelcome to Llama Panel Interactive Chat!", "blue", attrs=["bold"])
            while True:
                user_input = input("\n👤 You: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                conversation_history.append(await system.get_expert_answer(user_input, args.data_file, args.verbose, args.thinking, args.write_convo, conversation_history, execution_timeout=args.execution_timeout))
            cprint("Goodbye!", "blue")
    except (ollama.ResponseError, argparse.ArgumentTypeError) as e:
        cprint(f"\nFatal Error: {getattr(e, 'error', str(e))}", "red", attrs=["bold"], file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        cprint(f"\nAn unexpected error occurred: {e}", "red", attrs=["bold"], file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        cprint("\nGoodbye!", "blue")
        sys.exit(0)
