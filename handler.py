import runpod
import torch
import torchaudio
import base64
import io
import traceback
from omnivoice import OmniVoice

print("Initialisation du conteneur et chargement du modèle...")

try:
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", 
        device_map="cuda:0", 
        dtype=torch.float16
    )
    print("Modèle OmniVoice chargé.")
except Exception as e:
    print(f"Erreur modèle: {e}")

def handler(job):
    try:
        job_input = job.get('input', {})
        text = job_input.get("text", "")
        instruct = job_input.get("instruct", "")
        ref_audio_b64 = job_input.get("ref_audio_b64", "") # Le fameux fichier audio de clonage !
        
        if not text:
            return {"status": "error", "message": "Le texte est manquant."}

        kwargs = {"text": text}
        if instruct:
            kwargs["instruct"] = instruct

        # --- GESTION DU CLONAGE VOCAL ---
        if ref_audio_b64:
            ref_path = "/tmp/reference.wav" # On crée un fichier temporaire sur RunPod
            with open(ref_path, "wb") as f:
                # On nettoie l'entête envoyée par le HTML et on décode l'audio
                clean_b64 = ref_audio_b64.split(",")[-1] 
                f.write(base64.b64decode(clean_b64))
            
            # On indique à OmniVoice d'utiliser ce fichier temporaire
            kwargs["ref_audio"] = ref_path
            # Note: Pas besoin de 'ref_text', OmniVoice utilise Whisper en interne pour le deviner !

        # Génération
        audio_tensor = model.generate(**kwargs)
        
        # Renvoi de l'audio vers ton PC
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor[0], 24000, format="wav")
        audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "status": "success", 
            "audio_base64": audio_base64
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        return {"status": "error", "message": str(e), "trace": error_trace}

runpod.serverless.start({"handler": handler})
