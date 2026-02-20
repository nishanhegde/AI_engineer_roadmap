#!/usr/bin/env python3
"""
Local dev server for AI Engineer Roadmap tracker.
Serves static files + handles profile save/load API.

Usage:
    python3 server.py
    Open http://localhost:7001
"""
from http.server import HTTPServer, SimpleHTTPRequestHandler
import json, os, urllib.parse

PROGRESS_DIR = 'progress'
PORT = 7001

os.makedirs(PROGRESS_DIR, exist_ok=True)


class Handler(SimpleHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/api/profiles':
            try:
                names = sorted(
                    f[:-5] for f in os.listdir(PROGRESS_DIR)
                    if f.endswith('.json')
                )
                self._json(200, names)
            except Exception as e:
                self._json(500, {'error': str(e)})

        elif self.path.startswith('/api/profile/'):
            name = urllib.parse.unquote(self.path[len('/api/profile/'):])
            path = os.path.join(PROGRESS_DIR, f'{name}.json')
            if os.path.isfile(path):
                with open(path, 'rb') as f:
                    self._respond(200, f.read(), 'application/json')
            else:
                self._respond(404, b'{}', 'application/json')

        else:
            super().do_GET()

    def do_POST(self):
        if self.path.startswith('/api/profile/'):
            name = urllib.parse.unquote(self.path[len('/api/profile/'):])
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            path = os.path.join(PROGRESS_DIR, f'{name}.json')
            with open(path, 'wb') as f:
                f.write(body)
            self._json(200, {'ok': True})
        else:
            self._respond(404, b'Not found')

    def do_DELETE(self):
        if self.path.startswith('/api/profile/'):
            name = urllib.parse.unquote(self.path[len('/api/profile/'):])
            path = os.path.join(PROGRESS_DIR, f'{name}.json')
            if os.path.isfile(path):
                os.remove(path)
            self._json(200, {'ok': True})
        else:
            self._respond(404, b'Not found')

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS')
        self.end_headers()

    def _json(self, code, data):
        self._respond(code, json.dumps(data).encode(), 'application/json')

    def _respond(self, code, body, ct='text/plain'):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', ct)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if not (args and args[0].startswith(('GET /api', 'POST /api', 'DELETE /api'))):
            super().log_message(fmt, *args)


if __name__ == '__main__':
    print(f'AI Engineer Roadmap → http://localhost:{PORT}')
    print(f'Profiles saved in:   {os.path.abspath(PROGRESS_DIR)}/')
    print('Press Ctrl+C to stop.\n')
    HTTPServer(('', PORT), Handler).serve_forever()
