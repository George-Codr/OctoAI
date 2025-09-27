import asyncio
from brain import SuperBrain

QUESTION_TYPE_NAMES = {
    "reasoning": "Reasoning / Logic",
    "coding": "Coding / Programming",
    "search": "Web Search / Latest Info",
    "creativity": "Creativity / Story / Poem"
}

async def handle_query(brain):
    question = input("\nEnter your question:\n> ")
    if not question.strip():
        print("Empty question. Try again.")
        return

    # Use AI to automatically detect question type(s)
    print("Detecting question type...")
    regions = await brain.classify_regions(question)
    region_names = [QUESTION_TYPE_NAMES.get(r,r) for r in regions]
    print("Detected type(s):", ", ".join(region_names))

    print("Processing question...")
    expert_responses = await brain.run_region_experts(regions, question)
    final_answer = await brain.aggregate(question, expert_responses)

    print("\n=== SuperBrain Answer ===")
    print(final_answer)
    print("\nRegions processed:", regions)
    print("Expert outputs:")
    for k,v in expert_responses.items():
        print(f"- {k}: {v['provider']}/{v['model']}")

def main():
    brain = SuperBrain()
    print("Welcome to SuperBrain AI CLI (AI auto-detects question type).")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() in ["exit","quit"]:
            print("Goodbye!")
            break
        asyncio.run(handle_query(brain))

if __name__ == "__main__":
    main()
