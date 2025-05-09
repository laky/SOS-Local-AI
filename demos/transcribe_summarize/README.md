# Transcribe and Summarize

The workflow here uses Whisper.cpp to transcribe an audio recording and pass it to an LLM running in LM Studio to create a summary in markdown format.

## Requirements

Make sure you have LM Studio installed and the right model installed. For the demo I used `qwen3-30b-a3b`, but you can easily define some other model to be called in the python file by changing the `LLM_MODEL` variable to the model identifier.

Once you have LM Studio, enable developer mode, start the server, and load the model. I recommend increasing the context size when loading the model as your recording might not fit otherwise. 

We need ffmpeg and whispercpp installed as well. On a Mac, you can just run

    brew install ffmpeg whisper-cpp

You need to download the whispercpp model from https://huggingface.co/ggerganov/whisper.cpp/tree/main and make sure `WHISPER_MODEL_PATH` points to the correct file. For the demo I used `ggml-medium.bin`, but feel free to use any other model and see how it does.

Then just install the Python dependencies

    python -m venv venv 
    source venv/bin/activate
    pip install -r requirements.txt 

Once done, make sure the file you want to use is in the correct wav format supported by whisper. You can do that by calling something like

    ffmpeg -i ~/Downloads/recording.m4a -acodec pcm_s16le -ar 16000 -ac 1 -f wav ~/Downloads/recording.wav

To run the workflow, do

    python transcribe.py -i recording.wav -o summary.md

Common issues:
- If whisper fails, check that the model file path is passed in correctly and the input file is in the expected format. Try running the whisper cli on its own and see what errors you get. 
- If the summarisation step fails, check your LM Studio server is running, your model is loaded, and you have enough context set for the model. LM Studio shows logs on the Developer tab where you launch the server and load the models to serve.