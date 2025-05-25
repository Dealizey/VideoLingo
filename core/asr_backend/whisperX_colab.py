import os
import json
import shutil
from core.utils.config_utils import load_key
from rich import print as rprint

def transcribe_audio_colab(raw_audio_file, vocal_audio_file, start, end):
    """
    Manages the transcription of an audio file using WhisperX on Google Colab
    by generating a Jupyter notebook and guiding the user through the process.
    """
    # 1. Get paths and settings from config
    colab_audio_path = load_key("whisper.colab_audio_path")
    colab_result_path = load_key("whisper.colab_result_path")
    model_name = load_key("whisper.model")
    language_code = load_key("whisper.language") # Use 'None' for auto-detect if appropriate for whisperx
    model_dir_config = load_key("model_dir") # Local model dir, used for context

    # Ensure language_code is None if empty or 'auto', as whisperx might expect None for auto-detection
    if language_code and language_code.lower() in ["auto", "none", ""]:
        language_code = None

    # 2. Prepare audio paths
    os.makedirs(colab_audio_path, exist_ok=True)
    os.makedirs(colab_result_path, exist_ok=True)

    target_audio_filename = os.path.join(colab_audio_path, "audio_for_colab.mp3")
    
    # For now, copy the entire vocal_audio_file.
    # TODO: Implement audio segment clipping if start and end are provided and significant.
    # This would require a library like pydub or librosa.
    # Example using shutil.copy:
    shutil.copy(vocal_audio_file, target_audio_filename)
    rprint(f"[green]Audio file copied to:[/green] {os.path.abspath(target_audio_filename)}")

    # 3. Generate Jupyter Notebook
    notebook_filename = os.path.join(colab_audio_path, "whisperx_transcribe_colab.ipynb")
    # Model directory inside Colab, assuming VideoLingo is cloned there
    model_dir_colab = "/content/VideoLingo/_model_cache" 

    notebook_content = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.x", # Example, Colab might have a different version
                "mimetype": "text/x-python",
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                }
            }
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# WhisperX Transcription on Google Colab\n\n",
                    "This notebook will guide you through transcribing an audio file using WhisperX."
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "!echo \"Cloning VideoLingo repository (if not already present)...\"\n",
                    "!if [ ! -d \"VideoLingo\" ]; then git clone https://github.com/Huanshere/VideoLingo.git; else echo \"VideoLingo already cloned.\"; fi\n",
                    "import os\n",
                    "os.chdir('VideoLingo')\n",
                    "!echo \"Current directory: $(pwd)\"\n",
                    "\n",
                    "!echo \"Installing FFmpeg...\"\n",
                    "!apt-get update && apt-get install -y ffmpeg\n",
                    "\n",
                    "!echo \"Installing PyTorch (for CUDA 11.8)...\"\n",
                    "!pip install torch==2.0.0 torchaudio==2.0.0 --index-url https://download.pytorch.org/whl/cu118 -q\n",
                    "\n",
                    "!echo \"Installing whisperx and its dependencies...\"\n",
                    # This specific commit of whisperx will pull its tested versions of faster-whisper, ctranslate2, transformers, etc.
                    "!pip install git+https://github.com/m-bain/whisperx.git@7307306a9d8dd0d261e588cc933322454f853853 -q\n",
                    "\n",
                    "!echo \"Installing other potentially required libraries...\"\n",
                    # Add other direct dependencies from requirements.txt that whisperx or its components might need,
                    # or that are good for a robust audio processing environment.
                    # ctranslate2 is pulled by faster-whisper (a whisperx dep). transformers is pulled by whisperx.
                    # pandas, nltk are pulled by whisperx.
                    # pyannote.audio is pulled by whisperx.
                    "!pip install librosa==0.10.2.post1 numpy==1.26.4 rich PyYAML==6.0.2 requests==2.32.3 -q\n",
                    # Removed: pandas, as whisperx should install its preferred version.
                    # Added PyYAML and requests as they are small, often used, and listed in VideoLingo's install.py first steps.
                    "\n",
                    "!echo \"Dependency installation complete.\""
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 1: Upload your audio file\n\n",
                    f"Please upload the audio file located at `{os.path.abspath(target_audio_filename)}` on your local machine."
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "from google.colab import files\n",
                    "import os\n\n",
                    "print('Please upload the audio file (e.g., audio_for_colab.mp3)')\n",
                    "uploaded = files.upload()\n\n",
                    "if not uploaded:\n",
                    "  raise Exception('No file uploaded. Please upload the audio file to proceed.')\n\n",
                    "audio_file_path = list(uploaded.keys())[0]\n",
                    "print(f\"Uploaded '{audio_file_path}' successfully.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Step 2: Transcribe the audio using WhisperX"
                ]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": [
                    "import whisperx\n",
                    "import torch\n",
                    "import json\n",
                    "import gc\n\n",
                    f"model_name = \"{model_name}\"\n",
                    f"language_code = {repr(language_code)} # Handles None correctly\n",
                    f"model_dir_colab = \"{model_dir_colab}\"\n\n",
                    "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
                    "compute_type = \"float16\" if device == \"cuda\" else \"int8\"\n",
                    "if device == \"cuda\" and torch.cuda.get_device_capability()[0] < 7:\n",
                    "    print(\"Warning: GPU with CUDA capability < 7.0 found. Using bfloat16 if available, else float32.\")\n",
                    "    if hasattr(torch, 'bfloat16'):\n",
                    "        compute_type = \"bfloat16\"\n",
                    "    else:\n",
                    "        compute_type = \"float32\"\n\n",
                    "print(f\"Loading model: {model_name} on device: {device} with compute_type: {compute_type}\")\n",
                    "print(f\"Language for transcription: {language_code if language_code else 'auto-detect'}\")\n",
                    "print(f\"Model cache directory in Colab: {model_dir_colab}\")\n\n",
                    "try:\n",
                    "    model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language_code, download_root=model_dir_colab)\n",
                    "except Exception as e:\n",
                    "    print(f\"Error loading model: {e}\")\n",
                    "    print(\"Attempting to download model assets explicitly...\")\n",
                    "    try:\n",
                    "        from whisperx.utils import _download, _MODELS\n",
                    "        _download(_MODELS[model_name], model_dir_colab, model_name + '.pt')\n",
                    "        model = whisperx.load_model(model_name, device, compute_type=compute_type, language=language_code, download_root=model_dir_colab)\n",
                    "    except Exception as e_retry:\n",
                    "        print(f\"Retry failed: {e_retry}\")\n",
                    "        raise\n\n",
                    "print(f\"Transcribing {audio_file_path}...\")\n",
                    "audio = whisperx.load_audio(audio_file_path)\n\n",
                    "# Basic transcription\n",
                    "result = model.transcribe(audio, batch_size=16 if device == \"cuda\" else 4)\n",
                    "print(\"Initial transcription complete.\")\n\n",
                    "# Optional: Align model (if language is supported and alignment is desired)\n",
                    "align_model_a = None\n",
                    "try:\n",
                    "    print(f\"Attempting to load alignment model for language: {result['language']}\")\n",
                    "    align_model_a, align_metadata = whisperx.load_align_model(language_code=result[\"language\"], device=device, model_dir=model_dir_colab)\n",
                    "    print(\"Align model loaded.\")\n",
                    "    result = whisperx.align(result[\"segments\"], align_model_a, align_metadata, audio, device, return_char_alignments=False)\n",
                    "    print(\"Alignment complete.\")\n",
                    "except Exception as e:\n",
                    "    print(f\"Could not load or apply alignment model: {e}. Proceeding with unaligned segments.\")\n\n",
                    "output_transcription_file = \"transcription_result.json\"\n",
                    "with open(output_transcription_file, \"w\", encoding='utf-8') as f:\n",
                    "    json.dump(result, f, ensure_ascii=False, indent=2)\n\n",
                    "print(f\"Transcription saved to {output_transcription_file}\")\n",
                    "files.download(output_transcription_file)\n\n",
                    "# Clean up to free memory\n",
                    "del model, audio, result\n",
                    "if align_model_a: del align_model_a\n",
                    "gc.collect()\n",
                    "if device == 'cuda': torch.cuda.empty_cache()\n",
                    "print('Cleanup complete.')"
                ]
            }
        ]
    }

    try:
        with open(notebook_filename, "w", encoding='utf-8') as f:
            json.dump(notebook_content, f, indent=2)
        rprint(f"[green]Jupyter Notebook for Colab generated at:[/green] {os.path.abspath(notebook_filename)}")
    except IOError as e:
        rprint(f"[red]Error writing notebook file:[/red] {e}")
        # Handle error appropriately, perhaps by raising it or returning an error state
        raise

    # 4. Pause and instruct user
    rprint("\n[bold yellow]ACTION REQUIRED:[/bold yellow]")
    rprint(f"1. [cyan]Upload the generated audio file[/cyan] to Google Colab when prompted by the notebook.")
    rprint(f"   Your local audio file is: [italic]{os.path.abspath(target_audio_filename)}[/italic]")
    rprint(f"2. [cyan]Open and run the Jupyter Notebook[/cyan] in Google Colab:")
    rprint(f"   Notebook path: [italic]{os.path.abspath(notebook_filename)}[/italic]")
    rprint(f"3. After running the notebook, [cyan]download the 'transcription_result.json'[/cyan] file from Colab.")
    rprint(f"4. [cyan]Place the downloaded 'transcription_result.json'[/cyan] into the directory:")
    rprint(f"   [italic]{os.path.abspath(colab_result_path)}[/italic]\n")

    # 5. Wait for user input
    result_file_path = os.path.join(colab_result_path, "transcription_result.json")
    
    while not os.path.exists(result_file_path):
        try:
            input(f"Press Enter once 'transcription_result.json' is placed in '{os.path.abspath(colab_result_path)}' (or type 'skip' to abort)...")
            if "skip" in _.lower(): # Check if user typed skip
                 rprint("[yellow]Transcription process skipped by user.[/yellow]")
                 return {"segments": []} # Return empty segments or raise specific exception
        except KeyboardInterrupt:
            rprint("\n[bold red]Operation cancelled by user (Ctrl+C).[/bold red]")
            # Clean up copied audio file if desired
            if os.path.exists(target_audio_filename):
                try:
                    os.remove(target_audio_filename)
                    rprint(f"Cleaned up: {target_audio_filename}")
                except OSError as e:
                    rprint(f"[red]Error cleaning up audio file {target_audio_filename}: {e}[/red]")
            return {"segments": []} # Or raise an exception

        if not os.path.exists(result_file_path):
            rprint(f"[red]File not found: {result_file_path}. Please ensure the file is correctly placed and try again.[/red]")


    # 6. Load results
    try:
        with open(result_file_path, "r", encoding='utf-8') as f:
            transcription_data = json.load(f)
        rprint(f"[green]Successfully loaded transcription from:[/green] {os.path.abspath(result_file_path)}")
        
        # Optional: Clean up the result file after loading
        # os.remove(result_file_path) 
        # rprint(f"Cleaned up: {result_file_path}")
        # Optional: Clean up the copied audio file
        # if os.path.exists(target_audio_filename):
        #     os.remove(target_audio_filename)
        #     rprint(f"Cleaned up: {target_audio_filename}")

        return transcription_data
    except json.JSONDecodeError as e:
        rprint(f"[red]Error decoding JSON from {result_file_path}:[/red] {e}")
        raise  # Or handle more gracefully
    except IOError as e:
        rprint(f"[red]Error reading results file {result_file_path}:[/red] {e}")
        raise # Or handle more gracefully

if __name__ == '__main__':
    # This is for testing the notebook generation and user interaction flow locally.
    # You would need to manually create dummy files for this test.
    rprint("[bold yellow]Testing whisperX_colab.py script...[/bold yellow]")

    # Create dummy config entries for testing (usually loaded by load_key)
    # In a real scenario, these would come from your config.yaml or be mocked.
    if not os.path.exists("output/colab_audio"): os.makedirs("output/colab_audio")
    if not os.path.exists("output/colab_results"): os.makedirs("output/colab_results")
    if not os.path.exists("_model_cache"): os.makedirs("_model_cache")

    # Create a dummy vocal audio file
    dummy_vocal_file = "dummy_vocal.mp3"
    with open(dummy_vocal_file, "w") as f:
        f.write("This is a dummy audio file content.") # Actual audio content not needed for this test
    
    rprint(f"Created dummy vocal file: {dummy_vocal_file}")

    # Mock load_key if not running within the full application context
    # For simplicity, we assume config values are as per defaults in the function
    # or that you have a config.yaml that load_key can access.

    try:
        rprint("Simulating `transcribe_audio_colab` call...")
        # Note: 'raw_audio_file' is not used in the current simplified version,
        # but it's part of the function signature.
        result = transcribe_audio_colab(
            raw_audio_file="dummy_raw.mp3", # Not used by current function logic
            vocal_audio_file=dummy_vocal_file,
            start=None, 
            end=None
        )
        
        if result and 'segments' in result:
            rprint("[bold green]Colab transcription process completed (simulated).[/bold green]")
            rprint(f"Number of segments: {len(result['segments'])}")
            if result['segments']:
                rprint("First segment:", result['segments'][0])
        else:
            rprint("[bold red]Colab transcription process simulation might have been skipped or failed.[/bold red]")
            rprint("Result:", result)

    except Exception as e:
        rprint(f"[bold red]An error occurred during the test:[/bold red] {e}")
    finally:
        # Clean up dummy files
        if os.path.exists(dummy_vocal_file):
            os.remove(dummy_vocal_file)
            rprint(f"Cleaned up dummy vocal file: {dummy_vocal_file}")
        # The script also creates 'output/colab_audio/audio_for_colab.mp3'
        # and 'output/colab_audio/whisperx_transcribe_colab.ipynb'
        # and expects 'output/colab_results/transcription_result.json'
        # These can be cleaned up manually or by adding more cleanup code here.
        # For example:
        if os.path.exists("output/colab_audio/audio_for_colab.mp3"):
             os.remove("output/colab_audio/audio_for_colab.mp3")
        if os.path.exists("output/colab_audio/whisperx_transcribe_colab.ipynb"):
             os.remove("output/colab_audio/whisperx_transcribe_colab.ipynb")
        # Be cautious with recursive directory removal (shutil.rmtree)
        # if os.path.exists("output/colab_audio"): shutil.rmtree("output/colab_audio")
        # if os.path.exists("output/colab_results"): shutil.rmtree("output/colab_results")
        rprint("Test cleanup finished.")

# Example of how to mock load_key if needed for standalone testing:
# _original_load_key = load_key
# def mock_load_key(key_path, default=None):
#     if key_path == "whisper.colab_audio_path": return "output/colab_audio/"
#     if key_path == "whisper.colab_result_path": return "output/colab_results/"
#     if key_path == "whisper.model": return "large-v3"
#     if key_path == "whisper.language": return "en"
#     if key_path == "model_dir": return "_model_cache"
#     return _original_load_key(key_path, default)
# core.utils.config_utils.load_key = mock_load_key # monkeypatch if needed for testing outside app
# then restore with: core.utils.config_utils.load_key = _original_load_key
