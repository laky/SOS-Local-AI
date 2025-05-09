# SOS-Local-AI
Getting Started with Local LLMs - SOS Demo Material, May 2025

## Repository Structure
- This README contains information about the basic setup to get started with local LLMs, resources, and some tips and tricks.
- After reading this README, you should be able to have a local LLM set up and a way to use it.
- The `demos` folder contains a few interesting things you can try beyond basic LLM chatting. These are built as Langgraph workflows.

## Why Run LLMs Locally?
- **Privacy** - Be aware that many user-oriented applications like ChatGPT, Gemini, and Claude might be gathering data as you use them, potentially using it for training. More privacy-focused APIs exist, but require additional setup. Sending data over the internet always presents risks. For truly personal data such as private notes, health information, journal entries, WhatsApp messages, and personal photos, local open-source LLMs provide a more secure processing option.
- **Restrictions** - Some data may not be allowed to be uploaded to third-party services. Many companies have policies about securing their data or block the use of applications like ChatGPT. When you use a local LLM, the data never leaves your machine.
- **Costs** - Running large amounts of data through GPT, Gemini, or Anthropic APIs can be prohibitively expensive. If you have the necessary computing resources available, using a small local model might save you money. For example, using a local LLM to process and index several terabytes of photos would be more feasible than uploading that data to cloud APIs, even if the local processing takes longer.
    - Having said that, a disclaimer: Buying hardware just to run LLMs locally will most likely cost you a lot more than you'd pay for the online alternatives and you'll get less performant models to use. Unless you want privacy, the ability to tinker with the models, or have other use for your hardware / have hardware already available, it is not worth it for financial reasons.

## Setting Expectations
- Local large language models are an order of magnitude (or two, depending on your hardware) smaller than their cloud counterparts. Expect them to handle smaller, well-defined and specified tasks. Do not expect them to tackle complex problems the way the latest GPT models can.
- Speed is significantly slower, especially if you provide lots of context (documents, long prompts, long chats). You may need to wait a few minutes if you feed 10,000 words to the LLM.
- Good prompting and splitting problems into smaller tasks becomes much more important!
- Tooling is often open-source and aimed at tinkerers, providing great customizability but often poorly selected default values. It's easy to end up with suboptimal results due to incorrect parameters or model configuration.
- You'll typically run models quantized by the community through binaries developed by the community. The field is moving extremely quickly. Expect bugs and breaking changes, especially if you're using bleeding-edge models and tools.

## Hardware Requirements
LLM inference is counter-intuitive when it comes to hardware. Unlike many other applications, the processing speed of the CPU or GPU are generally not the bottlenecks.

### The 2 Main Variables to Optimize for Running Local LLMs:
- **RAM size** - impacts which models you can use. The larger your RAM, the more parameters you can fit.
- **RAM bandwidth** - generally the most important hardware metric that impacts how fast you can run local LLMs. Most CPUs have the computing power needed for LLMs but end up bottlenecked by how fast they can access the model weights stored in memory.

### ML Always Needs a GPU, Right? Well, It Depends in This Case:
- Many smaller models and the larger Mixture of Experts (MoE) models work pretty well on the CPU if you have fast RAM - see more below in the Model Recommendations section.
- GPUs have an order of magnitude faster memory bandwidth, which shows when running local LLMs. If you can fit the LLM into your GPU memory, you will see huge speed improvements over CPU.
- However, most consumer GPUs still come with small amounts of VRAM, which severely limits the size of the model (and context) you can use + high-memory GPUs are very expensive at the moment:
    - Largest non-server GPU is the recently released Nvidia RTX 6000 Pro with 96GB selling for ~$8,000
    - Older 48GB versions can be found used for ~$3,000-4,000 
    - Best Nvidia bang-for-the-buck for LLMs would probably be RTX 3090 with 24GB for about ~$700 (unless you want to take a gamble at some very old workstation GPUs)
    - You can try cheaper AMD and Intel GPUs as well and there is decent support for them in the LLM engines, though Nvidia still has an upper hand in performance and breadth of supported libraries thanks to CUDA
- GPUs especially shine when processing the prompt. Even having models that partially fit into GPU and partially into system RAM can result in noticeable speed-ups if you plan to provide a lot of context through very long prompts (e.g., long document processing, large codebases).

### Local LLM Gotcha: Context Length, i.e., How Much Info Can You Feed in a Prompt
- Before the LLM can start generating the response, it needs to process the whole prompt. This is generally an order of magnitude faster than the generation speed (two orders of magnitude when using a model that can fully fit within your total GPU memory).
- For some use cases you may want to provide detailed prompts, give examples (which improves LLM's ability to answer by a lot!), and large amounts of context like long documents or large code bases. Recent local LLMs support 32,000 - 128,000 tokens of context (1 token is about 0.75 words => ~20,000 - ~90,000 words).
- Long context inflates the RAM requirements by several GBs, which can be a problem when trying to fit the model on a GPU especially.
- Processing large context on a CPU instead of GPU is an order of magnitude slower, so you may end up waiting up to 20 minutes for an answer if you really use a lot of context.
- A clever trick: KV Caching can be used when you only need to run the long context processing once and you can save the several GB of context locally, then load it when needed [TO ADD]. (I think this is the most under-utilized feature at the moment when running LLMs locally)

### Casual Use
Most modern computers should be able to run a small LLM at a usable speed for chatting. The easiest way to find out whether your machine can manage is to download one of the recommendations from the Software Setup section and try. See Model Recommendations section for figuring out what might be a good LLM to try.

In my opinion, the best out-of-the-box hardware to play with LLMs are M-series Macs (M1-M4 MacBooks, Mac Minis, Mac Studios, etc.). They are ideal for quickly testing models as they usually have higher RAM bandwidth than PCs.

Recent Nvidia DGX Spark platform (+ ASUS Ascent GX10) and AMD Halo Strix (laptops and PCs with AMD AI Max 395+ chips) are catching up and becoming an interesting alternative, but they're only at the level of the basic M-series Mac chips and still severely lag behind the Max and Ultra versions (these are still 2-4x faster in terms of RAM bandwidth).

A gaming PC with a GPU with 16+ GB of memory will also work quite well, but you will be limited to smaller models. For running larger models, you may need to get creative about fitting (and powering) multiple GPUs in your PC to get enough combined GPU memory.

External GPUs are surprisingly performant. The bottleneck of using the USB/Thunderbolt connection instead of PCI Express on your motherboard directly is relatively minor. Hence, connecting multiple GPUs (even via Thunderbolt docks or daisy chaining) is an option. (I have put together a couple, hit me up with questions if interested)

If you want to add a GPU to your setup, I would recommend a used RTX 3090 24GB for around $700 as the best bang-for-the-buck Nvidia GPU for LLMs at the moment + it games very well (if you get bored of LLMs...)!

### Playing with the State-of-the-Art
Again, the best out-of-the-box solution would be the Apple Mac M-series Max and Ultra hardware. MacBooks go up to 128GB RAM (cca $5k) and Mac Studio M3 Ultra is up to 512GB RAM (cca $10k).

Custom PC builds using motherboards with support for 8-channel and 12-channel RAM and server CPUs (Intel Xeon, AMD EPYC / Threadripper) can reach competitive RAM bandwidths and are currently the only feasible option to run the largest models locally (DeepSeek R1 has 671 billion parameters and needs 512GB-1TB of memory, DeepSeek R2 is rumored to be double that size). Expect costs of ~$5,000 upwards unless you figure out how to combine and buy less mainstream server parts very cheaply. (I am contemplating building something like [this](https://www.reddit.com/r/LocalLLaMA/comments/1k8xyvp/finally_got_10ts_deepseek_v30324_hybrid_fp8q4_k_m/))

### Training and Fine-Tuning
When it comes to LLM training and fine-tuning, the tables turn, and Nvidia GPUs are strongly preferred.

For quick tests, Google Colab is a great option! - https://colab.research.google.com/

You can try running some fine-tuning on RTX 3090 locally, but renting out cloud GPUs is likely the best option here for anything usable. Some providers to consider are:
- https://vast.ai/ 
- https://www.runpod.io/
- https://lambda.ai/ 
- Any of your usual cloud providers will probably offer a GPU machine

*Note: The first 2 are peer-to-peer services so you are relying on the host not doing any snooping on top of the usual cloud privacy issues.*

## Software Setup

### Mac

#### LM Studio:
- Link: https://lmstudio.ai/
- LM Studio is excellent for running local models on Mac due to MLX support (think of it as "Apple's CUDA" - allows use of Mac GPUs for 10-15% faster performance when using MLX models).
- Download and load a model and you can start using it straight away.
- Includes a basic chat UI and simple document processing capabilities.
- Provides a local server API for third-party use.

#### Other Notable Local Apps for Mac:
- **JanAI** - https://jan.ai/
  - More streamlined than LM Studio, making chatting and basic use simpler.
  - Doesn't support serving the models via a local API, which limits third-party use of the LLM.
  - If you don't plan to use models via other tools, JanAI is recommended as it hides unnecessary complexities and offers a more user-friendly experience.
- **AnythingLLM** - https://anythingllm.com/
  - Provides more advanced features for local document processing and agentic tools (e.g., web search and site crawling).
  - Slightly less intuitive to get started with than JanAI or LM Studio.
- **Ollama (+ OpenWebUI)** - https://ollama.com/
  - Ollama is the fastest way to get up and running, download, run `ollama run MODEL_NAME` and that's it.
  - Lacks the MLX support of LM Studio, which is why LM Studio is my recommendation for Macs.
  - See below for extending with a ChatGPT-like web-based OpenWebUI.

### Linux and Windows
I have less personal experience with running LLMs on these platforms.

#### LM Studio:
- Link: https://lmstudio.ai/
- LM Studio seems to work on these platforms as well, but I haven't tested it for comparison.

#### Ollama + OpenWebUI:
- Ollama: https://ollama.com/
- Ollama is the fastest way to get up and running. Just install and use in the command line with `ollama run MODEL_NAME`. 
- Ollama contains a server that can be used by other tools.
- You can use ollama on its own from command line and via the API.
- OpenWebUI (https://github.com/open-webui/open-webui) provides a web-based interface similar to ChatGPT. It has a lot of functionality and support for some basic RAG and document use, but that needs some setup.

#### Kobold.cpp:
- Link: https://github.com/lostruins/koboldcpp
- Aimed at chatting and handles KV caching very well, making the web chat experience faster than the alternatives in my experience.

## Model Recommendations

### Model Sizes
- Generally, the larger the model, the more "intelligent" it is—it can handle more complex and ambiguous tasks.
- "Thinking" models are relatively recent and try to make use of Test-Time Scaling [TO ADD] locally. They take longer to respond as they first need to generate intermediate reasoning steps.

To peruse some popular models, visit Ollama: https://ollama.com/library. Generally, the download size is close to the bare minimum of what amount of memory you need to run the model on your system + add a few GB buffer for the context in the prompt to be safe. 

### Dense vs MoE (Mixture of Experts)
- MoE models need more parameters (more RAM) for performance equivalent to dense models but run much faster on CPUs due to fewer active parameters needed to generate each token.
- Most used local MoE models: DeepSeek R1, Qwen3 235b / 30b, Llama 4 Maverick / Scout, Mixtral
- Most used local dense models: Llama 3.3 70b, Qwen 2.5, Gemma 3, Phi 4, Mistral
- As a rule of thumb, people expect MoE models with `N` total and `M` active parameters to be equivalent to dense models with `sqrt(N*M)`, but there's no actual paper to back up this claim. 

### Model Quantization
Models are usually optimized for local use by processing their trained weights into lower precision through quantization. This allows larger models to take up less memory and run faster. Quantization generally causes models to be more unpredictable than the originals, but the ability to run larger models faster typically makes up for this drawback.

Some rules of thumb:
- 8-bit quantizations (q8_0) are considered nearly as good as the original float precision.
- 4-bit quantizations (q4_k_m) and above are generally usable.
- A lower quantization of a larger model is typically better than a higher quantization of a smaller model (e.g., q4_k_m of a 70B model is usually better than q8_0 of a 32B model and takes up roughly the same amount of memory). This isn't always true, so testing is recommended.
- It's expected some models are more sensitive to quantization. For certain models Quantization-Aware-Training (QAT) models were released, most notable one being 4-bit quantized Gemma3 QAT models from Google. 

You can experiment a bit with different quantizations and model settings if things are not working as expected. In LM Studio, you can find different quantizations in the search usually, or in the dropdown of the model card. In Ollama you get `q4_k_m` by default, but you can usually specify you want a different quantization, e.g., `ollama run qwen3:4b-q8_0`.

On Mac, you may want to search for mlx-community models for slightly faster speeds instead of the links provided below.

There are also many fine-tuned models from e.g., Nous, Hermes, Dolphin that might be interesting, but I tend to stick with the main ones for now. Quantized models from bartowski and unsloth are also generally high quality and can be used instead of the lmstudio-community ones. I tend to avoid models from other entities and users, there's already so much complexity and uncertainty when using LLMs that I do not want to add to it.

### Models for different systems
Below is a bunch of models that I think should work well on all sorts of systems. Lower your expectations for the really small models.

If you want to try something that you cannot fit on your machine, there are some options:
- vast.ai (https://vast.ai/) - Rent a machine with GPUs / lots of RAM
- RunPod (https://www.runpod.io/) - Rent a machine with GPUs
- OpenRouter (https://openrouter.ai/) - APIs to use different models, your milage may vary based on what provider you get linked to...
- Together AI (https://www.together.ai/) - APIs to use different models
- Groq (https://groq.com/) (not to be confused with Elon's Grok...) - Super fast inference HW with online APIs for models

*Note: vast.ai and RunPod use peer machines and people might be able to access the data. OpenRouter is popular, but I haven't played with. I've had good experience with Together AI and Groq so far, but they are more abstract, i.e., you don't set up your model etc, you just use the already working API.*

#### For Systems with RAM (+ VRAM) < 8GB
This is pushing it a bit and make sure you don't have high expectations about the models. They are best used for very clear and easy tasks (summarization, rewriting an email, little fun chats, etc.). Don't expect models to deal with ambiguity, show solid logic or reasoning, and don't rely on them when they pretend to know things.
- Qwen3 4B (<4GB) - A recent text-only model from Alibaba with 4 billion parameters (very small) and ability to toggle thinking mode via /think and /no_think in the prompt (/think is the default). Quite decent for the size.
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/Qwen3-4B-GGUF
    - Ollama: `ollama run qwen3:4b`
- Qwen3 8B - An 8 billion parameter version of the above. Many people are impressed by its performance. 
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/Qwen3-8B-GGUF
    - Ollama: `ollama run qwen3:8b`
- Gemma 4B QAT (<4GB) - A multimodal model from Google (i.e., can work with images as input). Decent, but the larger versions of this model (12B and 27B) are a lot stronger.
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/gemma-3-4B-it-qat-GGUF
    - Ollama: `ollama run gemma3:4b-it-qat`
- Llama 3.2 3B (<4GB) - More of an honorable mention from Meta, older model now, but small and fast.
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/Llama-3.2-3B-Instruct-GGUF
    - Ollama: `ollama run llama3.2`
- Llama 3.1 8B - Quite a decent model from Meta in this range, a bit older at this point.
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF 
    - Ollama: `ollama run llama3.1`

#### For Systems with RAM (+ VRAM) 8-16GB
The models at this level are starting to be usable. They can do some simple reasoning, follow clear instructions, call tools, understand images. Ambiguity will still stump them and hallucination is still a serious issue.
- Phi 4 14B - A very solid model from Microsoft, especially useful for instruction following / tool calling in workflows (I often use it for quick tests there and it usually doesn't disappoint). 
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/phi-4-GGUF / https://model.lmstudio.ai/download/unsloth/phi-4-GGUF
    - Ollama: `ollama run phi4`
- Phi 4 Reasoning 14B - A new model from Microsoft that has support for reasoning, looks potentially interesting.
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/Phi-4-reasoning-plus-GGUF
    - Ollama: `ollama run phi4-reasoning`
- Qwen3 14B - A recent text-only model from Alibaba with 14 billion parameters and ability to toggle thinking mode via /think and /no_think in the prompt (/think is the default). Very solid model for this memory range.
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/Qwen3-14B-GGUF
    - Ollama: `ollama run qwen3:14b`
- Qwen2.5-coder 14B - A previous generation model from Alibaba optimized for coding, generally regarded as a very strong local coding model.
    - LM Studio: https://model.lmstudio.ai/download/Qwen/Qwen2.5-Coder-14B-Instruct-GGUF
    - Ollama: `ollama run qwen2.5-coder:14b`
- Gemma3 12B QAT - Bigger multimodal (can ingest images) model from Google, a big step up from Gemma 4B, not quite Gemma 27B, can be a bit memory hungry and slow with images with this amount of RAM.
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/gemma-3-12B-it-qat-GGUF
    - Ollama: `ollama run gemma3:12b-it-qat`
- Qwen2 VL 7B - A rather small, fast, and capable local model that can ingest images.
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/Qwen2-VL-7B-Instruct-GGUF
    - Ollama: *I've not found any reputable ones here*
- Llava - The original model with focus on processing images, more of an honorable mention at this point.
    - LM Studio: https://model.lmstudio.ai/download/second-state/Llava-v1.5-7B-GGUF
    - Ollama: `ollama run llava`

#### For Systems with RAM (+ VRAM) 16-32GB
At this point the models are becoming quite solid. The recent Gemma3 and Qwen3 releases are really becoming quite interesting and the Qwen-coder model can be used in Aider to supplement the Cursor / Windsurf cloud tooling.
- Qwen3 30B - MoE model from Alibaba, very fast on CPU and quite decent with the ability to toggle thinking mode via /think and /no_think in the prompt (/think is the default). Incredibly usable model on machines without GPUs. Will likely become a go-to model for fast responses.
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/Qwen3-30B-A3B-GGUF / https://model.lmstudio.ai/download/unsloth/Qwen3-30B-A3B-GGUF
    - Ollama: `ollama run qwen3:30b`
- Gemma3 27B QAT - Dense model, which means it might be a bit slow at this size if not run on GPU or on a fast Mac. Very solid model from Google and for me it became a go-to for local workflows and tool calling. Can ingest images.
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/gemma-3-27B-it-qat-GGUF
    - Ollama: `ollama run gemma3:27b-it-qat`
- Qwen2.5-coder 32B - A previous generation dense model from Alibaba optimized for coding, generally regarded as still one of the strongest local models for coding.
    - LM Studio: https://model.lmstudio.ai/download/Qwen/Qwen2.5-Coder-32B-Instruct-GGUF
    - Ollama: `ollama run qwen2.5-coder:32b`
- Qwen3 32B - Largest dense model from the recent Qwen3 family from Alibaba. More capable than the Qwen 3 30B MoE model above, but a lot slower, especially on a CPU.
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/Qwen3-32B-GGUF
    - Ollama: `ollama run qwen3:32b`
- Llama4 Scout / Maverick - Huge MoE models with 109B / 401B parameters respectively that should not be in this category if it wasn't for the fact that, apparently, you may be able to get this huge model to run (or rather slowly walk) with most of the weights offloaded to a fast SSD. (https://www.reddit.com/r/LocalLLaMA/comments/1k2uztr/llama_4_is_actually_goat/)
    - No Ollama or LM Studio links here, you need to get the underlying engine and play with the command line a bit to make it work...

#### For Systems with RAM (+ VRAM) 32GB-128GB
On the higher end of this range, you get to run some very capable models at like GPT-3.5+ quality.
- Qwen3 30B - MoE model from Alibaba's latest generation. This model is so capable and fast that it requires a mention in this section as well as the above one. If you have more memory, I'd opt for 8-bit quantization of this model. 
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/Qwen3-30B-A3B-GGUF (select q8_0 in the dropdown)
    - Ollama: `ollama run qwen3:30b-q8_0`
- Qwen3 32B q8 - Dense model from the same Alibaba generation, GPU recommended. Use higher quantization if you can fit it and if it runs fast enough.
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/Qwen3-32B-GGUF (select q8_0 in the dropdown)
    - Ollama: `ollama run qwen3:32b-q8_0`
- Llama 3.3 70B (>48GB) - Very capable last generation model from Meta. It's a dense model, so GPU recommended at this size.
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/Llama-3.3-70B-Instruct-GGUF
    - Ollama: `ollama run llama3.3`
- Qwen 2.5 72B (>48GB) - A very capable last generation dense model from Alibaba. Some people prefered it over the llama 3.3, I haven't quite used it much. GPU recommended for any kind of usable speed. 
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/Qwen2.5-72B-Instruct-GGUF
    - Ollama: `ollama run qwen2.5:72b`
- Qwen3 235B 3-bit (>100GB) - The largest model released by Alibaba. Same as the rest of the Qwen3 family, just bigger and more capable. It is an MoE and runs well on CPU, but needs a lot of memory.
    - LM Studio: https://model.lmstudio.ai/download/unsloth/Qwen3-235B-A22B-GGUF - q3 quants should just about fit into 128GB memory
    - Ollama: `ollama run qwen3:235b` *(needs >150GB RAM as they do not have lower than 4-bit quants in the default repository)*
- Llama4 Scout (>64GB) 109B 3-bit - 6-bit - The latest MoE model from Meta. It received a lot of criticism on release, but might be quite capable and runs very fast on CPU.
    - LM Studio: https://model.lmstudio.ai/download/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF
    - Ollama: `ollama run llama4:scout`

#### For Systems with RAM (+ VRAM) > 128GB
You can run the absolute state of the art of the local open-source models. These models are at GPT-4+ level and can reasonably replace a lot of the cloud workflows.
- Qwen3 235B - Large MoE from the most recent Alibaba release, runs well on CPU.
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/Qwen3-235B-A22B-GGUF
    - Ollama: `ollama run qwen3:235b`
- Llama4 Maverick 401B (>256GB) - The latest MoE model from Meta with a huge amount of parameters and RAM requirements, but runs super fast on CPU due to only 17B active parameters.
    - LM Studio: https://model.lmstudio.ai/download/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF
    - Ollama: `ollama run llama4:maverick`
- DeepSeek R1 671B (~512GB RAM) for the current latest greatest open source model, good luck on finding the RAM needed... If you have the means to run this model locally, I'm very impressed and curious to hear more. 
    - LM Studio: https://model.lmstudio.ai/download/unsloth/DeepSeek-R1-GGUF
    - Ollama: `ollama run deepseek-r1:671b`

#### What I Play with at the Moment (M4 Max MacBook Pro with 128GB RAM)
- Phi 4 - Supposedly good at instruction following and quite fast.
    - LM Studio: unsloth q8_0 version from https://model.lmstudio.ai/download/unsloth/phi-4-GGUF
- Gemma3 27b QAT - Good overall, multimodal, i.e., can process images really well.
    - LM Studio: https://model.lmstudio.ai/download/lmstudio-community/gemma-3-27B-it-qat-GGUF
- Llama 3.3 70b - A bit slower and last-gen, but performed quite well for me in the past.
    - LM Studio: q4_k_m and q8_0 both work from https://model.lmstudio.ai/download/lmstudio-community/Llama-3.3-70B-Instruct-GGUF
- Qwen3 30b - Super fast, thinking mode, likely will replace Phi4 and Gemma3 for most text work.
    - LM Studio: https://model.lmstudio.ai/download/mlx-community/Qwen3-30B-A3B-8bit
- Qwen3 235b (3-bit) - The largest model I can fit at the moment on the machine. Need to properly test what it is capable of.
    - LM Studio: Q3_k_xl version from https://model.lmstudio.ai/download/unsloth/Qwen3-235B-A22B-GGUF
- Llama Scout (4-bit / 6-bit) - Similar to the above. Largest model I can fit at the moment, but this one also can use some higher quantization version. I need to still test what it is properly capable of. 
    - LM Studio: q4_k_xl and q6_k from https://model.lmstudio.ai/download/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF

Contemplating building a workstation/server with 512-1024 GB multi-channel RAM to be able to play with running DeepSeek R1, Llama Maverick, Qwen 235B locally. Hit me up if you're curious about this...

## Demos

### Langgraph
- Workflows with smaller and more specific tasks for LLMs are more likely to succeed.
- Allows combining tasks in non-trivial ways and using AI agents.
- Quick intro to Workflows and Agents 101 and the differences between them: [video](https://www.youtube.com/watch?v=aHCDrAbH_go), [docs](https://langchain-ai.github.io/langgraph/tutorials/workflows/)

### Transcribe + Summarize a Recording
See [Transcribe and Summarize](demos/transcribe_summarize/README.md) for details on how to transcribe and summarize audio recordings using whisper.cpp and LM Studio.

### Generate a Podcast About a Site
See [Generate a Podcast](demos/podcast/README.md) for details on how to generate a podcast with two speakers discussing a topic from a URL using Dia and LM Studio.

## References and Resources
- LocalLlama Reddit (https://www.reddit.com/r/LocalLLaMA/) - Excellent community of tinkerers using local LLMs
- llama.cpp (https://github.com/ggml-org/llama.cpp) - The underlying LLM engine used by Ollama and LM Studio. Kind of a start of this whole run LLMs locally in a fast and easy way craze.
  - Alternatives: ExLlama, vLLM
- Hugging Face (https://huggingface.co/) - Where all the models are stored
- Unsloth AI (https://unsloth.ai/) - Fine-tuning with efficient use of GPU memory. They have very nice Colab examples for finetuning different models, e.g. https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp.
