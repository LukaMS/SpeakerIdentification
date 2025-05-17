# Speaker Identification Tool

A simple command-line Python application to detect occurrences of a known speaker in an audio recording using the PyAnnote framework.

## Features

* **Speaker Diarization**: Splits audio into speech segments and labels them.
* **Embedding Extraction**: Generates embeddings for each segment.
* **Speaker Matching**: Compares segment embeddings against a reference speaker embedding to classify as `USER` or `UNKNOWN`.
* **Duration Filtering**: Ignores very short segments below a configurable threshold.
* **Merged Output**: Merges adjacent segments with the same label for a clean timeline.

## Requirements

* Python 3.8+
* PyTorch
* [PyAnnote](https://github.com/pyannote/pyannote-audio)
* NumPy
* SciPy

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/speaker-identification.git
   cd speaker-identification
   ```
2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\\Scripts\\activate    # Windows
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

```bash
python main.py \
  --enroll path/to/reference_speaker.wav \
  --audio path/to/test_recording.wav \
  [--threshold 0.7] \
  [--cpu] \
  [--hf_token YOUR_HUGGINGFACE_TOKEN]
```

### Arguments

* `--enroll` (required): Path to a clean reference audio file of the speaker to enroll.
* `--audio` (required): Path to the target audio recording to scan.
* `--threshold` (optional): Cosine distance threshold for matching (default: `0.7`). Lower values are stricter.
* `--cpu` (optional): Force CPU usage even if CUDA is available.
* `--hf_token` (optional): Hugging Face authentication token (or set `HF_TOKEN` in the environment).

### Example

```bash
python main.py --enroll data/clerk.wav --audio data/test.wav --threshold 0.7 --hf_token hf_xxx123
```

```text
INFO: Using device: cuda
INFO: Processing reference audio data/john_doe.wav...
INFO: Generated reference for 'SPEAKER_00' from 5 segments.
INFO: Successfully obtained reference embedding for Enrolled Voice.
INFO: Processing test audio data/meeting.wav...

--- Speaker Identification Result ---
[00:00:02.500 --> 00:00:10.200] Speaker: USER
[00:00:10.200 --> 00:00:15.000] Speaker: UNKNOWN
...
INFO: Processing finished.
```

## Configuration

* Adjust `MIN_SEGMENT_DURATION_S` and `SIMILARITY_DISTANCE_THRESHOLD` in `main.py` for finer control over segment filtering and matching sensitivity.

## License

This project is released under the MIT License. Feel free to use, modify, and distribute.
