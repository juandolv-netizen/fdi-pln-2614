from pathlib import Path

def load_corpus(path: str) -> str:
    """Carga y concatena todos los archivos .txt de un directorio."""
    directorio = Path(path)
    
    if not directorio.is_dir():
        raise NotADirectoryError(f"El directorio {path} no existe o no es válido.")

    archivos = list(directorio.glob("*.txt"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron archivos .txt en: {path}")

    return "\n".join(open(p, encoding="utf-8").read() for p in archivos)