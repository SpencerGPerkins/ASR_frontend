import torch
import torchaudio
import torchaudio.io
import sounddevice as sd

bundle = torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
feature_extraction = bundle.get_streaming_feature_extractor()
decoder = bundle.get_decoder()
token_processor = bundle.get_token_processor()

sr = bundle.sample_rate
segment_len = bundle.segment_length * bundle.hop_length
context_len = bundle.right_context_length * bundle.hop_length

print(f"sample rate : {sr}")
print(f"segment length: {segment_len}")
print(f"Right context: {context_len}")

class ContextCacher():
    """ Cache context 
    Args:
        segment_length: int, length of the primary segment
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

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_chunk = torch.tensor(indata[:, 0], dtype=torch.float32)
    run_inference(audio_chunk)
    
@torch.inference_mode()
def run_inference(chunk):
        global state, hypothesis, segment_len, sr
        segment = cacher(chunk)
        features, length = feature_extraction(segment)
        hypos,state = decoder.infer(
            features, length, 10, state=state, hypothesis=hypothesis 
        )
        hypothesis = hypos
        trans = token_processor(hypothesis[0][0], lstrip=False)
        print("\n")
        print(trans, end=" ", flush=True)

cacher = ContextCacher(segment_len, context_len)
state, hypothesis = None, None
input_devices =sd.query_devices(kind="input")

with sd.InputStream(callback=callback, channels=1, samplerate=sr):
    print("Listening.... Press Ctrl+c to stop.")
    sd.sleep(int(10 * sr))