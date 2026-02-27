import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler

LLAMA_HEALTH_URL = "http://localhost:8080/health"
PORT_HEALTH = 8081

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/ping":
            try:
                # Probe llama-server
                req = urllib.request.Request(LLAMA_HEALTH_URL)
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        # Server is ready
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b"OK")
                        return
            except urllib.error.HTTPError as e:
                if e.code == 503:
                    # Model still loading
                    pass 
                else:
                    # Other HTTP error
                    pass
            except Exception:
                # Connection refused (server not started yet)
                pass

            # Any state other than 200 OK means we are initializing
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
            
    def log_message(self, format, *args):
        # Suppress logging to avoid spamming the console every 10 seconds
        pass

if __name__ == "__main__":
    print(f"[foundry] RunPod health sidecar starting on port {PORT_HEALTH}")
    server = HTTPServer(("0.0.0.0", PORT_HEALTH), HealthHandler)
    server.serve_forever()
