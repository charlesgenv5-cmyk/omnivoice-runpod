import runpod
import torch
import torchaudio
import base64
import io
import traceback
from omnivoice import OmniVoice

print("Initialisation du conteneur et chargement du modèle...")

# Chargement du modèle au démarrage (hors de la fonction handler pour ne le faire qu'une fois)
try:
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice", 
        device_map="cuda:0", 
        dtype=torch.float16
    )
    print("Modèle OmniVoice chargé avec succès sur le GPU.")
except Exception as e:
    print(f"Erreur critique lors du chargement du modèle: {e}")

def handler(job):
    """
    Fonction principale appelée par RunPod à chaque nouvelle requête.
    """
    try:
        job_input = job.get('input', {})
        text = job_input.get("text", "")
        instruct = job_input.get("instruct", "") # Ex: "female, low pitch"
        
        # Vérification de sécurité
        if not text:
            return {"status": "error", "message": "Le champ 'text' est vide ou manquant."}

        print(f"Génération en cours pour : '{text[:30]}...'")

        # Paramètres dynamiques pour OmniVoice
        kwargs = {"text": text}
        if instruct:
            kwargs["instruct"] = instruct

        # Génération de l'audio
        audio_tensor = model.generate(**kwargs)
        
        # Sauvegarde de l'audio en mémoire vive (RAM)
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor[0], 24000, format="wav")
        
        # Encodage en Base64 pour le retour API
        audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {
            "status": "success", 
            "audio_base64": audio_base64,
            "message": "Audio généré avec succès."
        }

    except Exception as e:
        # En cas de crash, on renvoie l'erreur exacte pour faciliter ton "vibe coding"
        error_trace = traceback.format_exc()
        print(f"Erreur lors de la génération: {error_trace}")
        return {
            "status": "error", 
            "message": str(e), 
            "trace": error_trace
        }

# Lancement du serveur RunPod
runpod.serverless.start({"handler": handler})