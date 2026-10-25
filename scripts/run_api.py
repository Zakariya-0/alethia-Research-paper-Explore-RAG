"""
Script de Lancement du Serveur API REST FastAPI (Uvicorn).

Usage :
-------
python scripts/run_api.py [--host 127.0.0.1] [--port 8000]
"""

import sys
import argparse
from pathlib import Path

# Encodage sécurisé pour Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import uvicorn
from src.api.app import app


def main():
    parser = argparse.ArgumentParser(
        description="Lancement du serveur API REST pour la vérification scientifique."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Adresse IP d'écoute (défaut: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port d'écoute (défaut: 8000)"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🚀 DÉMARRAGE DU SERVEUR API FASTAPI — VÉRIFICATION SCIENTIFIQUE")
    print("=" * 80)
    print(f"• URL Serveur API : http://{args.host}:{args.port}")
    print(f"• Application Web : http://{args.host}:{args.port}/")
    print(f"• Documentation   : http://{args.host}:{args.port}/docs (Swagger UI)")
    print(f"• Métriques       : http://{args.host}:{args.port}/api/stats")
    print("=" * 80 + "\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
