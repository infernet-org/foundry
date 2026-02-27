import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error

# Bind to port 80 (RunPod's default LB port)
PORT = 80
LLAMA_PORT = int(os.environ.get("FOUNDRY_PORT", "8080"))
LLAMA_HOST = f"http://127.0.0.1:{LLAMA_PORT}"

class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ping":
            # RunPod health check
            try:
                req = urllib.request.Request(f"{LLAMA_HOST}/health")
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b"OK")
                        return
            except Exception:
                pass
            
            # If not 200, return 204 to indicate initializing
            self.send_response(204)
            self.end_headers()
            return
            
        # Proxy other GET requests (like /health, /v1/models)
        self.proxy_request("GET")

    def do_POST(self):
        # Proxy POST requests (like /v1/chat/completions)
        self.proxy_request("POST")
        
    def proxy_request(self, method):
        url = f"{LLAMA_HOST}{self.path}"
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None
        
        # Prepare headers for upstream
        req_headers = {}
        for key, value in self.headers.items():
            if key.lower() not in ['host', 'connection']:
                req_headers[key] = value
                
        req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
        
        try:
            with urllib.request.urlopen(req) as response:
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() != 'transfer-encoding': # Let http.server handle chunking
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(response.read())
                
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for key, value in e.headers.items():
                if key.lower() != 'transfer-encoding':
                    self.send_header(key, value)
            self.end_headers()
            self.wfile.write(e.read())
            
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def log_message(self, format, *args):
        # Only log non-ping requests
        if "/ping" not in args[0]:
            sys.stderr.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format%args))

if __name__ == "__main__":
    print(f"[foundry] RunPod unified proxy starting on port {PORT}, forwarding to {LLAMA_HOST}")
    server = HTTPServer(("0.0.0.0", PORT), ProxyHandler)
    server.serve_forever()
