# Generate a Podcast (NotebookLM-style)

The workflow here uses a recent text-to-speech model Dia (https://github.com/nari-labs/dia) to create a podcast with 2 speakers discussing a topic covered in the URL you provide. 

Example:
<video src="https://github.com/laky/SOS-Local-AI/raw/refs/heads/main/demos/podcast/podcast.mp3"></video>

## Requirements

Make sure you have LM Studio installed and the right model installed. For the demo I used `lmstudio-community/gemma-3-27b-it-qat`, but you can easily define some other model to be called in the python file by changing the `LLM_MODEL` variable to the model identifier.

Once you have LM Studio, enable developer mode, start the server, and load the model. I recommend increasing the context size when loading the model as your recording might not fit otherwise. 

We need ffmpeg installed as well. On a Mac, you can just run

    brew install ffmpeg

Then just install the Python dependencies

    python -m venv venv 
    source venv/bin/activate
    pip install -r requirements.txt 

To run the workflow, do

    python workflow.py -i recording.wav -o summary.md

Common issues:
- The version of Dia I used is very very slow (at least on a Mac). Dia can be cutting off ends of sentences. It may also create audio that sounds too fast. Feel free to play with the `slow_down_audio` method. Upgrading to a newer version of Dia might also solve some issues perhaps.
- It may happen that the website is not scraped correctly, because it uses javascript to load the content, or some bot detection prevents from accessing it. There are ways to load these using Selenium / Playwright, let me know if you are interested in this.
- If the podcast transcript generation step fails, check your LM Studio server is running, your model is loaded, and you have enough context set for the model. LM Studio shows logs on the Developer tab where you launch the server and load the models to serve.