import sys
import json
import os

# Fix HuggingFace symlink errors on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

import torch
import torchaudio
import gc
import traceback
import warnings

warnings.filterwarnings("ignore")

def process(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments = data.get("segments", [])
    lang = data.get("lang", "vi")
    ref_audio = data.get("ref_audio", "")
    ref_text = data.get("ref_text", None)
    speed = data.get("speed", 1.0)
    out_dir = data.get("out_dir", "")
    
    result = {"status": "error", "message": "Unknown"}
    
    try:
        from omnivoice import OmniVoice
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Load the model once
        model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map=device, dtype=dtype)
        
        for i, text in enumerate(segments):
            if not text.strip():
                continue
            
            # Generate audio for the segment
            kwargs = {
                "text": text,
                "speed": speed
            }
            if os.path.exists(ref_audio):
                kwargs["ref_audio"] = ref_audio
            if ref_text:
                kwargs["ref_text"] = ref_text
                
            audio_arrays = model.generate(**kwargs)
            
            if audio_arrays and len(audio_arrays) > 0:
                audio_data = audio_arrays[0]
                
                if isinstance(audio_data, torch.Tensor):
                    audio_t = audio_data.cpu().float()
                else:
                    audio_t = torch.from_numpy(audio_data)
                
                if audio_t.ndim == 1:
                    audio_t = audio_t.unsqueeze(0)
                    
                out_path = os.path.join(out_dir, f"seg_{i:04d}.wav")
                torchaudio.save(out_path, audio_t, 24000)
        
        result["status"] = "success"
        result["message"] = "Tất cả file đã được tạo"
        
    except Exception as e:
        result["message"] = str(e)
        traceback.print_exc()
        
    finally:
        # Cleanup VRAM aggressively
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        gc.collect()
        
    out_result_file = os.path.join(out_dir, "result.json")
    with open(out_result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python omnivoice_worker.py <input.json>")
        sys.exit(1)
    
    process(sys.argv[1])
