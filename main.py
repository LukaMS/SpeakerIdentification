# ──────── Silence everything ────────
import os
import argparse
import warnings
import logging
import io
from contextlib import redirect_stdout, redirect_stderr

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["NUMEXPR_MAX_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger().setLevel(logging.ERROR)

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

for name in (
    "numexpr", 
    "speechbrain", 
    "pyannote.audio", 
    "pyannote", 
    "pytorch_lightning", 
    "transformers"
):
    logging.getLogger(name).setLevel(logging.ERROR)
    logging.getLogger(name).propagate = False

# ──────── Now import the rest silently ────────
buf = io.StringIO()
with redirect_stdout(buf), redirect_stderr(buf):
    import torch
    import numpy as np
    from scipy.spatial.distance import cosine
    from pyannote.audio import Pipeline, Model, Inference # type: ignore
    from pyannote.core import Segment # type: ignore

def parse_args():
    p = argparse.ArgumentParser(
        description="Detect occurrences of a known speaker in a recording using PyAnnote.")
    p.add_argument("--enroll", required=True, help="Clean reference WAV/FLAC of the speaker to enrol.")
    p.add_argument("--audio", required=True, help="Target recording to scan.")
    p.add_argument("--threshold", type=float, default=0.7, help="Cosine‑similarity threshold (↑ stricter).")
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available.")
    p.add_argument("--hf_token", help="Hugging Face token (or set HF_TOKEN env var).")
    return p.parse_args()

# --- Setup ---
args = parse_args()
device = torch.device(device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
print(f"INFO: Using device: {device}")

# --- Configuration ---
HF_TOKEN = args.hf_token
ENROLLED_AUDIO_PATH = args.enroll
TEST_AUDIO_PATH = args.audio
MIN_SEGMENT_DURATION_S = 0.2
SIMILARITY_DISTANCE_THRESHOLD = args.threshold

# 1) Diarization pipeline
print("INFO: Loading diarization pipeline...")
with open(os.devnull, 'w') as devnull, \
     redirect_stdout(devnull), redirect_stderr(devnull):
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN
    )

diarization_pipeline.to(device)
print("INFO: Diarization pipeline loaded.")

#Load Embedding Model
print("INFO: Loading embedding model…")
with open(os.devnull, 'w') as devnull, \
     redirect_stdout(devnull), redirect_stderr(devnull):
    embedding_model = Model.from_pretrained(
        "pyannote/embedding",
        use_auth_token=HF_TOKEN
    )
embedding_model.to(device)
embedding_inference = Inference(embedding_model, window="whole")
print("INFO: Embedding model loaded.")

# --- Helper Function ---
def format_time(seconds: float) -> str:
    """Formats time in seconds to HH:MM:SS.mmm string."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    msecs = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}.{msecs:03d}"

# --- Function to get reference speaker embeddings ---
def get_reference_embeddings(audio_file_path, dia_pipeline, emb_inference):
    print(f"INFO: Processing reference audio {audio_file_path}...")
    diarization_output = dia_pipeline(audio_file_path)
    speaker_segment_embeddings = {}

    for segment, _, speaker_label in diarization_output.itertracks(yield_label=True):
        if segment.duration < MIN_SEGMENT_DURATION_S:
            continue
        try:
            current_segment_embedding = emb_inference.crop(audio_file_path, segment)
            if speaker_label not in speaker_segment_embeddings:
                speaker_segment_embeddings[speaker_label] = []
            speaker_segment_embeddings[speaker_label].append(current_segment_embedding)
        except RuntimeError as e:
            #print(f"WARNING: Runtime error on reference segment {segment} (Duration: {segment.duration:.3f}s): {e}. Skipping segment.")
            continue
        except Exception as e:
            #print(f"WARNING: Unexpected error on reference segment {segment} (Duration: {segment.duration:.3f}s): {e}. Skipping segment.")
            continue

    reference_embeddings = {}
    for speaker_label, embeddings_list in speaker_segment_embeddings.items():
        if embeddings_list:
            stacked_embeddings = np.stack(embeddings_list)
            mean_embedding = np.mean(stacked_embeddings, axis=0)
            reference_embeddings[speaker_label] = mean_embedding
            print(f"INFO: Generated reference for '{speaker_label}' from {len(embeddings_list)} segments in {audio_file_path}.")
        else:
            print(f"INFO: No valid segments for '{speaker_label}' in {audio_file_path} to create reference embedding.")
            
    if not reference_embeddings:
        print(f"WARNING: No reference embeddings generated for {audio_file_path} (all segments might have been too short or caused errors).")
    return reference_embeddings

# --- Main logic ---

# 1. Get reference embedding for Enrolled
Enrolled_reference_data = get_reference_embeddings(
    ENROLLED_AUDIO_PATH,
    diarization_pipeline,
    embedding_inference
)

if not Enrolled_reference_data:
    raise ValueError(f"FATAL: Could not generate any reference speaker embeddings from {ENROLLED_AUDIO_PATH}. Check warnings above.")

try:
    Enrolled_embedding = next(iter(Enrolled_reference_data.values())) # Assumes first speaker is Enrolled Voice
    print(f"INFO: Successfully obtained reference embedding for Enrolled Voice.")
except StopIteration:
    raise ValueError(f"FATAL: No speaker embeddings found in Enrolled_reference_data for {ENROLLED_AUDIO_PATH} after processing. Ensure input .wav contains usable audio for the primary speaker.")
except Exception as e:
    raise ValueError(f"FATAL: An error occurred while trying to get User's embedding from reference_data: {e}")

# 2. Process the test audio file
print(f"INFO: Processing test audio {TEST_AUDIO_PATH}...")
diarization_test = diarization_pipeline(TEST_AUDIO_PATH)

# 3. Classify segments from the test audio
classified_segments = []

for segment, _, speaker_label_pipeline in diarization_test.itertracks(yield_label=True):

    if segment.duration < MIN_SEGMENT_DURATION_S:
        continue

    try:
        current_segment_embedding = embedding_inference.crop(TEST_AUDIO_PATH, segment)
        dist = cosine(current_segment_embedding, Enrolled_embedding)

        speaker_identity = "UNKNOWN"
        if dist <= SIMILARITY_DISTANCE_THRESHOLD:
            speaker_identity = "USER"
        
        classified_segments.append({
            'start': segment.start,
            'end': segment.end,
            'label': speaker_identity
        })

    except RuntimeError as e:
        #print(f"WARNING: Runtime error during embedding extraction for segment {segment} (Duration: {segment.duration:.3f}s): {e}. Skipping segment.")
        continue
    except Exception as e:
        #print(f"WARNING: Unexpected error processing segment {segment} (Duration: {segment.duration:.3f}s): {e}. Skipping segment.")
        continue

# 4. Merge adjacent segments with the same identified speaker label
merged_segments = []
if classified_segments:
    # Start with the first classified segment
    current_merged_start = classified_segments[0]['start']
    current_merged_end = classified_segments[0]['end']
    current_merged_label = classified_segments[0]['label']

    for i in range(1, len(classified_segments)):
        next_segment = classified_segments[i]
        # If the next segment has the same label, extend the current merged segment
        if next_segment['label'] == current_merged_label:
            current_merged_end = next_segment['end']
        else:
            # Different label, so finalize the current merged segment and add it
            merged_segments.append({
                'start': current_merged_start,
                'end': current_merged_end,
                'label': current_merged_label
            })
            # Start a new merged segment
            current_merged_start = next_segment['start']
            current_merged_end = next_segment['end']
            current_merged_label = next_segment['label']
    
    # Add the last processed merged segment
    merged_segments.append({
        'start': current_merged_start,
        'end': current_merged_end,
        'label': current_merged_label
    })

# 5. Print the final neat, merged output
print("\n--- Speaker Identification Result ---")
if not merged_segments:
    print("No speaker segments identified or all were too short/problematic.")
else:
    for seg in merged_segments:
        print(f"[{format_time(seg['start'])} --> {format_time(seg['end'])}] Speaker: {seg['label']}")

print("\nINFO: Processing finished.")