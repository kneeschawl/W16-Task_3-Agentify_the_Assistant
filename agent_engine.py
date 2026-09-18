import time
import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.tools import tool

# --- 1. RAG & KNOWLEDGE BASE SETUP ---
sample_doc_content = """
The AI Assistant system is built with a hybrid approach:
- Local LLM: Llama 3.1 8B served via Ollama for privacy and offline tasks.
- Vector Database: ChromaDB for storing and retrieving document embeddings.
- RAG Architecture: Augments LLM responses using semantic search over knowledge bases.
- Tool Integration: Executes dynamic python functions to compute system metrics.
- Security Policy: All external system access requires dual-factor approval.
"""

documents = [Document(page_content=sample_doc_content)]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

embeddings = OllamaEmbeddings(model="llama3.1:8b")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- 2. TOOLS FOR AGENTIC LOOP ---
@tool
def rag_search(query: str) -> str:
    """Searches the knowledge base for relevant system documentation."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant documentation found."
    return " ".join([d.page_content for d in docs])

@tool
def calculate_system_uptime(days_active: int) -> str:
    """Calculates operational uptime percentage based on days active."""
    total_hours = days_active * 24
    return f"Active for {days_active} days ({total_hours} hours). Estimated Uptime: 99.9%."

@tool
def verify_security_policy(policy_name: str) -> str:
    """Verifies security requirements for system commands."""
    if "security" in policy_name.lower():
        return "Security requirement: Dual-factor approval required for all administrative actions."
    return "Standard operational policy applies."

TOOLS = {
    "rag_search": rag_search,
    "calculate_system_uptime": calculate_system_uptime,
    "verify_security_policy": verify_security_policy
}

# --- 3. CONTEXT ENGINEERING: COMPACTION ---
def compact_observation(tool_name: str, raw_output: str) -> str:
    """
    Context Engineering Technique: Compaction
    Prunes verbose tool outputs into concise representations to prevent
    context window bloat over multi-iteration loops.
    """
    cleaned = raw_output.replace("\n", " ").strip()
    if len(cleaned) > 200:
        cleaned = cleaned[:197] + "..."
    return f"[{tool_name} Summary]: {cleaned}"

# --- 4. AGENTIC REACT LOOP ENGINE ---
class AgentEngine:
    def __init__(self, model_name: str = "llama3.1:8b", max_iterations: int = 4):
        self.llm = ChatOllama(model=model_name, temperature=0.1)
        self.max_iterations = max_iterations

    def run(self, user_query: str, injected_failure: str = None) -> Dict[str, Any]:
        """
        Executes an agentic loop allowing multi-step reasoning, tool evaluation,
        and stopping condition enforcement.
        """
        trajectory = []
        token_count = 0
        current_step = 0
        completed = False
        final_answer = ""
        
        # System prompt setting up the decision boundary
        messages = [
            ("system", "You are an agentic assistant. You can call tools multiple times if needed. "
                       "Tools available: rag_search, calculate_system_uptime, verify_security_policy. "
                       "Evaluate tool responses. When you have sufficient info, state 'FINAL ANSWER: <your answer>'."),
            ("user", user_query)
        ]

        start_time = time.time()

        while current_step < self.max_iterations and not completed:
            current_step += 1
            step_log = {"step": current_step, "action": None, "observation": None}

            # Simulate Token Accounting
            prompt_tokens = sum(len(str(m)) for m in messages) // 4
            token_count += prompt_tokens

            try:
                # Failure Injection Check
                if injected_failure == "tool_unavailable" and current_step == 1:
                    raise RuntimeError("Tool execution service unreachable (Injected Failure)")

                # Execute LLM call
                llm_with_tools = self.llm.bind_tools(list(TOOLS.values()))
                response = llm_with_tools.invoke(messages)
                
                output_tokens = len(str(response.content)) // 4
                token_count += output_tokens

                # Check for tool calls
                if hasattr(response, "tool_calls") and response.tool_calls:
                    tool_call = response.tool_calls[0]
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    
                    step_log["action"] = f"Call Tool: {tool_name}({tool_args})"
                    
                    # Execute tool
                    if tool_name in TOOLS:
                        raw_obs = TOOLS[tool_name].invoke(tool_args)
                    else:
                        raw_obs = f"Error: Tool '{tool_name}' not found."

                    # Apply Context Engineering: Compaction
                    compacted_obs = compact_observation(tool_name, str(raw_obs))
                    step_log["observation"] = compacted_obs

                    # Update history with compacted context
                    messages.append(("assistant", f"I called tool {tool_name} with {tool_args}"))
                    messages.append(("user", f"Tool output: {compacted_obs}"))

                else:
                    # Model provided final answer without calling tools or reached conclusion
                    content = str(response.content)
                    step_log["action"] = "Generate Response"
                    step_log["observation"] = content
                    
                    if "FINAL ANSWER:" in content or current_step > 1:
                        final_answer = content.replace("FINAL ANSWER:", "").strip()
                        completed = True
                    else:
                        messages.append(("assistant", content))

            except Exception as e:
                step_log["action"] = "Error Encountered"
                step_log["observation"] = str(e)
                completed = False
                final_answer = f"Agent execution halted due to failure: {str(e)}"
                break

            trajectory.append(step_log)

        if not completed and not final_answer:
            final_answer = "Iteration limit reached before task completion."

        latency = round(time.time() - start_time, 2)

        return {
            "query": user_query,
            "final_answer": final_answer,
            "trajectory": trajectory,
            "iterations": current_step,
            "completed": completed,
            "tokens_consumed": token_count,
            "latency_seconds": latency
        }