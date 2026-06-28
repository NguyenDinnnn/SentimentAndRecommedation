#!/bin/sh

set -e

echo "Container starting..."

DB_FILE="/app/shopAI.db"

if [ ! -f "$DB_FILE" ]; then
    echo "Database not found. Initializing..."

    python init_db.py

    python add_stock.py

    echo "Database initialized."
fi

exec "$@"