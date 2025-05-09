# Schema for structured output
from pydantic import BaseModel, Field
import numpy as np
import trafilatura
import argparse

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

import soundfile as sf
from dia.model import Dia

SAMPLE_AUDIO_PATH = "sample_audio.wav"
SAMPLE_AUDIO_TEXT = """
[S1] Hey. how are you doing?  
[S2] Pretty good. Pretty good. What about you? 
[S1] I'm great. So happy to be speaking to you.  
[S2] Me too. This is some cool stuff. Huh?  
[S1] Yeah. I have been reading more about speech generation. 
[S2] Yeah. 
[S1] And it really seems like context is important. 
[S2] Definitely.
""".strip()

prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages([
    ("system", """
        You are a helpful assistant that can generate a short podcast script. 
        The podcast has 2 speakers. 
        Generate a script talking about the main topics of the given text. 
        Stick to the expected output format.
    """),
    ("user", """
        Generate a podcast script from the following text:
        {content}
        
        The script should be in the following format:
        [S1] Something the first person says.
        [S2] Something the second person says.
        [S1] Something the first person says.
        [S2] Something the second person says.
        ...
     
        Example output:
        [S1] Dia is an open weights text to dialogue model. 
        [S2] You get full control over scripts and voices. 
        [S1] Wow. Amazing. (laughs) 
        [S2] Try it now on Git hub or Hugging Face.
     
        Make the podcast fun to listen to and engaging. You can use the following qualifiers to indicate speaker's emotions where applicable:
        (laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles)
     
        Output:
    """),
])


llm = ChatOpenAI(
    model="lmstudio-community/gemma-3-27b-it-qat",
    openai_api_base="http://localhost:1234/v1",
    openai_api_key="lm-studio",
    temperature=0.6,
)

class State(BaseModel):
    input_url: str = Field(..., description="The URL of the input document.")
    content: str = Field(..., description="The content of the input document.")
    podcast_script: str = Field(..., description="The transcript of the podcast to generate.")
    output_path: str = Field(..., description="The path where the output MP3 file will be saved.")

def get_content(state: State) -> State:
    print(f"Getting content from {state.input_url}")
    downloaded = trafilatura.fetch_url(state.input_url)
    content = trafilatura.extract(downloaded)
    print(f"Content:\n{content}")
    return State(input_url=state.input_url, content=content, podcast_script="", output_path=state.output_path)

def generate_podcast_transcript(state: State) -> State:
    print("Generating podcast transcript")
    response = (prompt | llm).invoke({"content": state.content})
    podcast_script = response.content if hasattr(response, 'content') else str(response)

    # Format the script for Dia
    # Remove any empty lines and ensure proper formatting
    lines = [line.strip() for line in podcast_script.split('\n') if line.strip() and (line.strip().startswith('[S1]') or line.strip().startswith('[S2]'))]
    formatted_text = '\n'.join(lines)
    return State(input_url=state.input_url, content=state.content, podcast_script=formatted_text, output_path=state.output_path)

def slow_down_audio(audio: np.ndarray) -> np.ndarray:
    original_len = len(audio)
    speed_factor = 0.8
    target_len = int(original_len / speed_factor)  # Target length based on speed_factor
    if target_len != original_len and target_len > 0:  # Only interpolate if length changes and is valid
        x_original = np.arange(original_len)
        x_resampled = np.linspace(0, original_len - 1, target_len)
        return np.interp(x_resampled, x_original, audio)
    else:
        # Nothing to do.
        return audio

def generate_podcast_audio(state: State) -> State:
    print("\nGenerating podcast audio:\n")
    model = Dia.from_pretrained("nari-labs/Dia-1.6B", device='cpu')
    # Process 2 lines at a time to go around the short context window of the model.
    lines = state.podcast_script.split('\n')
    root_file_name = state.output_path.split('.')[0]
    for i in range(0, len(lines), 2):
        text = '\n'.join(lines[i:i+2])
        print(f"Generating audio for {text}")
        output = model.generate(SAMPLE_AUDIO_TEXT + text, audio_prompt_path=SAMPLE_AUDIO_PATH, use_torch_compile=True)
        # output = slow_down_audio(output)
        sf.write(f"{root_file_name}_{i//2}.mp3", output, 44100)

    # Merge all the audio files into a single file
    audio_segments = []
    for i in range(len(lines) // 2):
        audio_data, sr = sf.read(f"{root_file_name}_{i}.mp3")
        audio_segments.append(audio_data)
    merged_audio = np.concatenate(audio_segments)
    sf.write(state.output_path, merged_audio, sr)
      
    return state

def main() -> None:
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate a podcast from a URL')
    parser.add_argument('-i', '--input-url', type=str, required=True, help='URL of the input document')
    parser.add_argument('-o', '--output-path', type=str, required=True, help='Path where the output MP3 file will be saved')
    args = parser.parse_args()

    # Build workflow
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("get_content", get_content)
    workflow.add_node("generate_podcast_transcript", generate_podcast_transcript)
    workflow.add_node("generate_podcast_audio", generate_podcast_audio)

    # Add edges to connect nodes
    workflow.add_edge(START, "get_content")
    workflow.add_edge("get_content", "generate_podcast_transcript")
    workflow.add_edge("generate_podcast_transcript", "generate_podcast_audio")
    workflow.add_edge("generate_podcast_audio", END)

    # Compile
    graph = workflow.compile()
    # Invoke

    inputs = {
        "input_url": args.input_url,
        "content": "",
        "podcast_script": "",
        "output_path": args.output_path
    }

    for message_chunk, metadata in graph.stream(inputs, stream_mode="messages"):
        if message_chunk.content:
            print(message_chunk.content, end="", flush=True)
    
    print(f"Podcast audio saved to {args.output_path}")

if __name__ == "__main__":
    main()