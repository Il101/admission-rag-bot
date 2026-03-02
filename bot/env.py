import os
import urllib.parse

def setup_env():
    url = os.environ.get("DATABASE_URL")
    if url and not os.environ.get("POSTGRES_HOST"):
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        p = urllib.parse.urlparse(url)
        os.environ["POSTGRES_USER"] = p.username or ""
        os.environ["POSTGRES_PASSWORD"] = p.password or ""
        os.environ["POSTGRES_HOST"] = p.hostname or ""
        os.environ["POSTGRES_DB"] = (p.path[1:] if p.path else "")
    
    # Fallback to prevent omegaconf KeyError if variables are completely missing
    for var in ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_DB"]:
        if var not in os.environ:
            os.environ[var] = ""

setup_env()
