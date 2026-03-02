#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.

if [ -n "$DATABASE_URL" ] && [ -z "$POSTGRES_HOST" ]; then
    echo "Parsing DATABASE_URL for Postgres credentials..."
    eval $(python3 -c "
import os, urllib.parse
url = os.environ['DATABASE_URL']
if url.startswith('postgres://'): url = url.replace('postgres://', 'postgresql://', 1)
p = urllib.parse.urlparse(url)
print(f\"export POSTGRES_USER='{p.username}' POSTGRES_PASSWORD='{p.password}' POSTGRES_HOST='{p.hostname}' POSTGRES_DB='{p.path[1:]}'\")
")
fi

if [ -z "$POSTGRES_HOST" ]; then
    echo "Warning: POSTGRES_HOST is not set or empty. Skipping bot initialization gracefully. (If this is the PgVector service, this is intended)."
    sleep infinity
    exit 0
fi

python3 init_scripts/init_bot_db.py
python3 init_scripts/index_knowledge_base.py
python3 bot/app.py