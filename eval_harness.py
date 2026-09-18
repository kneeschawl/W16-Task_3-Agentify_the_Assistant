import json
from agent_engine import AgentEngine

def run_evaluation():
    agent = AgentEngine(max_iterations=4)
    
    test_suite = [
        {
            "id": "Q1",
            "query": "What vector database is used in this system?",
            "expected_tool": "rag_search"
        },
        {
            "id": "Q2",
            "query": "Calculate system uptime for 10 active days and verify the security policy.",
            "expected_tool": "calculate_system_uptime"
        },
        {
            "id": "Q3_Failure_Injection",
            "query": "Search for documentation on hardware specs.",
            "inject_failure": "tool_unavailable"
        }
    ]

    results = []
    print("\n=======================================================")
    print("      RUNNING AGENTIC ASSISTANT EVALUATION HARNESS     ")
    print("=======================================================\n")

    for test in test_suite:
        print(f"Executing Test {test['id']}...")
        failure_mode = test.get("inject_failure", None)
        
        output = agent.run(test["query"], injected_failure=failure_mode)
        
        # Determine failure taxonomy
        failure_type = "None"
        if failure_mode:
            failure_type = "Hard Failure (Injected)"
        elif not output["completed"]:
            failure_type = "Soft Failure (Incomplete)"

        # Check tool correctness
        tool_correct = False
        for step in output["trajectory"]:
            if test.get("expected_tool") and test["expected_tool"] in str(step["action"]):
                tool_correct = True
                break

        res_summary = {
            "Test ID": test["id"],
            "Completed": output["completed"],
            "Iterations": output["iterations"],
            "Tokens Used": output["tokens_consumed"],
            "Latency (s)": output["latency_seconds"],
            "Tool Correctness": tool_correct if "expected_tool" in test else "N/A",
            "Failure Category": failure_type
        }
        results.append(res_summary)

    print("\n---------------- EVALUATION RESULTS TABLE ----------------")
    print(f"{'ID':<20} | {'Completed':<10} | {'Steps':<6} | {'Tokens':<8} | {'Failure Category'}")
    print("-" * 75)
    for r in results:
        print(f"{r['Test ID']:<20} | {str(r['Completed']):<10} | {r['Iterations']:<6} | {r['Tokens Used']:<8} | {r['Failure Category']}")
    
    print("\n=======================================================\n")

if __name__ == "__main__":
    run_evaluation()