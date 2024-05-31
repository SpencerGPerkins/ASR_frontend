import torch 
import torchaudio
import torchaudio.io

# Load ASR system Pipelines
bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
feature_extraction = bundle.get_streaming_feature_extractor()
decoder = bundle.get_decoder()
token_processor = bundle.get_token_processor()

# Set parameters
sr = bundle.sample_rate
segment_len = bundle.segment_length * bundle.hop_length
context_len = bundle.right_context_length * bundle._hop_length

class ContextCacher():
    """ Cache context 
    Args:
        segment_length: int, length of the primary segment
                used for padding the input chunk if < segment_length
        context_length: int, length of context
            
    Returns:
        chunk_with_context: tensor, concatination of input context and input chunk
    """

    def __init__(self, segment_length: int, context_length: int):
        self.seg_length = segment_length
        self.context_length = context_length
        self.context = torch.zeros([context_length])

    def __call__(self, chunk: torch.Tensor):
        if chunk.size(0) < self.seg_length:
            chunk = torch.nn.functional.pad(
                    chunk, (0, self.seg_length - chunk.size(0))
            )

        chunk_with_context = torch.cat((self.context, chunk))
        self.context = chunk[-self.context_length :]

        return chunk_with_context
        
cacher = ContextCacher(segment_len, context_len)
state, hypothesis = None, None

@torch.inference_mode()
def run_inference(n=100):
    global state, hypothesis, segment_len, sr
    chunks = []
    feats = []
    for i, (chunk, ) in enumerate(stream_iter, start=1):
        segment = cacher(chunk[:, 0])
        features, length = feature_extraction(segment)
        hypos, state = decoder.infer(
            features, length, 10, state=state, hypothesis=hypothesis
        )
        hypothesis = hypos
        trans = token_processor(hypothesis[0][0], lstrip=False)
        print("\n")
        print(trans, end=" ", flush=True)

        chunks.append(chunk)
        feats.append(features)
        # if i == n: # Comment out for full transcript
        #     break

src = "../Data/wedding.mp3"
# src = "../Data/LJ001-0001.wav"

# Configure streamer
streamer = torchaudio.io.StreamReader(src)
streamer.add_basic_audio_stream(frames_per_chunk=segment_len,
                                sample_rate=sr
                                )
cacher = ContextCacher(segment_len, context_len)
sate, hypothesis = None, None
stream_iter = streamer.stream()

run_inference()
