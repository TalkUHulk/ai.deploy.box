import http.server

class MyHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        http.server.SimpleHTTPRequestHandler.end_headers(self)

http.server.SimpleHTTPRequestHandler.extensions_map['.wasm'] = 'application/wasm'
httpd = http.server.HTTPServer(('localhost', 12580), MyHttpRequestHandler)
httpd.serve_forever()
