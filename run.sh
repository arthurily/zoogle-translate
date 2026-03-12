#!/bin/bash
# Start the Zoogle Translate server and open in browser
cd "$(dirname "$0")"
echo "Starting server at http://127.0.0.1:8000"
echo "Press Ctrl+C to stop"
(sleep 2; open "http://127.0.0.1:8000" 2>/dev/null || python3 -c "import webbrowser; webbrowser.open('http://127.0.0.1:8000')") &
python3 server.py
