from crewai import Agent, Task, Crew
from config.llm_config import crew_llm

def create_subject_expert(subject):
    """Create a subject-specific NEET expert agent."""
    return Agent(
        role=f"NEET {subject} Expert",
        goal=f"Generate detailed {subject} questions following NEET examination standards",
        backstory=f"""
            You are an experienced NEET {subject} expert with deep understanding 
            of the examination pattern and curriculum. You excel at creating 
            questions that test conceptual understanding.
        """,
        allow_delegation=False,
        verbose=True,
        llm=crew_llm
    )

def create_question_task(agent, subject, topic, processed_input, context):
    """Create a task for question generation with ReAct prompting."""
    context_text = "\n".join(context) if context else "No additional context"

    prompt = f"""Follow this ReAct (Reasoning and Acting) framework to generate a NEET question:

1. Thought: Analyze the input and context
   - What is the key concept from the {subject} input?
   - What relevant information is available in the context?
   - What difficulty level is appropriate for NEET?

2. Action: Plan the question structure
   - Identify the core concept to test
   - Determine question format (MCQ/numerical/theoretical)
   - Plan necessary calculations or reasoning steps

3. Observation: Review available materials
   - Input material: {processed_input}
   - Context material: {context_text}
   - Topic focus: {topic}

4. Question Generation:
   - Create a clear, unambiguous question
   - Include any necessary diagrams or data
   - Provide 4 options if MCQ
   
5. Answer Explanation:
   - Provide the correct answer
   - Give a detailed step-by-step solution
   - Explain the underlying concept
   - Add relevant NEET exam tips
"""

    return Task(
        description=prompt,
        expected_output="""A complete NEET question with:
                           1. Question text
                           2. Multiple choice options or solution approach
                           3. Correct answer
                           4. Detailed explanation
                           5. Conceptual insights""",
        agent=agent,
        llm=crew_llm
    )

def generate_question(subject, topic, input_data, input_type, context=None):
    """Generate a NEET question based on input and context."""
    from utils.input_processor import process_input
    
    processed_input = process_input(input_data, input_type)
    expert = create_subject_expert(subject)
    task = create_question_task(expert, subject, topic, processed_input, context or [])

    crew = Crew(
        agents=[expert],
        tasks=[task],
        llm=crew_llm
    )

    result = crew.kickoff()
    return result