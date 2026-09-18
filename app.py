import streamlit as st
from agent_engine import AgentEngine

st.set_page_config(page_title="Agentic AI Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Task 3: Agentic AI Assistant")
st.caption("Powered by Llama 3.1 8B, ChromaDB, ReAct Agentic Loop & Context Compaction")

if "agent" not in st.session_state:
    st.session_state.agent = AgentEngine(max_iterations=4)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar options
st.sidebar.header("Agent Settings")
inject_failure = st.sidebar.checkbox("Inject Failure (Test Resiliency)")
max_steps = st.sidebar.slider("Max Loop Iterations", 1, 5, 4)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "trajectory" in msg:
            with st.expander("🔍 View Agent Trajectory & Thinking Loop"):
                for step in msg["trajectory"]:
                    st.write(f"**Step {step['step']}:** `{step['action']}`")
                    st.caption(f"Observation: {step['observation']}")

# User query input
if prompt := st.chat_input("Ask a multi-step query..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Agent thinking & running iterations...", expanded=True) as status:
            failure_mode = "tool_unavailable" if inject_failure else None
            result = st.session_state.agent.run(prompt, injected_failure=failure_mode)
            
            for step in result["trajectory"]:
                st.write(f"**Step {step['step']} Action:** `{step['action']}`")
                if step['observation']:
                    st.caption(f"Observation: {step['observation']}")

            if result["completed"]:
                status.update(label=f"Completed in {result['iterations']} steps ({result['latency_seconds']}s)", state="complete")
            else:
                status.update(label="Halted / Failed", state="error")

        st.markdown(result["final_answer"])
        st.info(f"📊 **Metrics:** {result['tokens_consumed']} Tokens | {result['iterations']} Iterations | {result['latency_seconds']}s Latency")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["final_answer"],
        "trajectory": result["trajectory"]
    })