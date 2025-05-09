# Schema for structured output
import subprocess
import sys
from pydantic import BaseModel, Field
import argparse

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

LLM_MODEL = "qwen3-30b-a3b"
WHISPER_MODEL_PATH = "./ggml-medium.bin"
LANGUAGE = "sk"

prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", """
        You are a helpful assistant and your task is to summarize a transcription of a talk. 
        Generate a short brief report, highlighting the main topics and ideas mentioned.
        Output it in a markdown format.
        Generate 3 questions that an expert on the topic could ask the presenter at the end of the talk.
    """),
    ("user", """
        Generate a markdow summary from the following transcription, include the 3 questions as well:
        {transcript}
                
        Output:
    """),
])


llm = ChatOpenAI(
    model=LLM_MODEL,
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio",
)

class State(BaseModel):
    input_path: str = Field(..., description="The path to the audio recording.")
    transcript: str = Field(..., description="The transcript of the talk.")
    summary: str = Field(..., description="The markdown summary.")
    output_path: str = Field(..., description="The path where the output MD file will be saved.")

def get_transcript(state: State) -> State:
    print(f"Transcribing {state.input_path}")
    process = subprocess.Popen(
        ["whisper-cpp", "--model", WHISPER_MODEL_PATH, "-l", LANGUAGE, state.input_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Initialize transcript text
    transcript = ""
    
    # Read and print output in real-time
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
        transcript += line
        sys.stdout.flush()
    
    # Wait for process to complete
    process.wait()
    
    return State(input_path=state.input_path, transcript=transcript, summary="", output_path=state.output_path)

def get_summary(state: State) -> State:
    print("Generating summary from the transcript")
    response = (prompt | llm).invoke({"transcript": state.transcript})
    summary = response.content if hasattr(response, 'content') else str(response)
    summary = summary.split("</think>")[-1].replace("```markdown", "").replace("```", "")

    with open(state.output_path, "w") as md_file:
        md_file.write(summary)
    return State(input_path=state.input_path, transcript=state.transcript, summary=summary, output_path=state.output_path)

def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate a summary from an audio recording')
    parser.add_argument('-i', '--input-path', type=str, required=True, help='Path of the audio file')
    parser.add_argument('-o', '--output-path', type=str, required=True, help='Path where the output MD file will be saved')
    args = parser.parse_args()

    # Build workflow
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("get_transcript", get_transcript)
    workflow.add_node("get_summary", get_summary)

    # Add edges to connect nodes
    workflow.add_edge(START, "get_transcript")
    workflow.add_edge("get_transcript", "get_summary")
    workflow.add_edge("get_summary", END)

    # Compile
    graph = workflow.compile()
    # Invoke

    inputs = {
        "input_path": args.input_path,
        "transcript": "",
        "summary": "",
        "output_path": args.output_path
    }

    for message_chunk, metadata in graph.stream(inputs, stream_mode="messages"):
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
    
    print(f"Summary saved to {args.output_path}")

if __name__ == "__main__":
    main()