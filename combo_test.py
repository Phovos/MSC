#!/usr/bin/env python3
"""
Production-quality JSON-RPC server with BaseModel integration.
Combines the elegance of ASGI-style routing with robust HTTP handling.
"""

import json
import asyncio
import socket
import time
import logging
import signal
import ssl
from typing import Callable, Awaitable, Dict, Any, Optional, Union, List
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from dataclasses import dataclass, fields
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# BaseModel Implementation (Simplified for focus on serving)
# ============================================================================


@dataclass(frozen=True)
class BaseModel:
    """Lightweight BaseModel for data validation and serialization."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        for field_obj in fields(self):
            value = getattr(self, field_obj.name)
            if isinstance(value, BaseModel):
                result[field_obj.name] = value.to_dict()
            elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
                result[field_obj.name] = [item.to_dict() for item in value]
            else:
                result[field_obj.name] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseModel':
        """Create from dictionary."""
        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)


@dataclass(frozen=True)
class MemoryStats(BaseModel):
    size: int
    count: int
    traceback: str
    timestamp: float
    peak_memory: int


@dataclass(frozen=True)
class FileModel(BaseModel):
    file_name: str
    file_content: str

    def save(self, path):
        full_path = path / self.file_name
        full_path.write_text(self.file_content)


# ============================================================================
# Enhanced JSON-RPC Application with Production Features
# ============================================================================


@dataclass
class ServerConfig:
    """Server configuration with production defaults."""

    host: str = "127.0.0.1"
    port: int = 8080
    max_request_size: int = 10_000_000  # 10MB
    request_timeout: float = 30.0
    max_concurrent_requests: int = 100
    enable_cors: bool = True
    debug: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None


class JSONRPCError(Exception):
    """JSON-RPC 2.0 compliant errors."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    SERVER_ERROR = -32000

    def __init__(
        self, code: int, message: str, data: Any = None, request_id: Any = None
    ):
        self.code = code
        self.message = message
        self.data = data
        self.request_id = request_id
        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        error = {"code": self.code, "message": self.message}
        if self.data is not None:
            error["data"] = self.data
        return error


class JSONRPCApp:
    """Enhanced JSON-RPC application with production features."""

    def __init__(self, config: Optional[ServerConfig] = None):
        self.config = config or ServerConfig()
        self.routes: Dict[str, Callable[[Dict], Awaitable[Dict]]] = {}
        self.middleware: List[Callable] = []
        self.request_count = 0
        self.error_count = 0
        self.active_requests = 0
        self._shutdown_event = asyncio.Event()

    def route(self, method_name: str):
        """Register a JSON-RPC method."""

        def decorator(func: Callable[[Dict], Awaitable[Dict]]):
            self.routes[method_name] = func
            logger.info(f"Registered method: {method_name}")
            return func

        return decorator

    def middleware_handler(self, func: Callable):
        """Register middleware."""
        self.middleware.append(func)
        return func

    async def _run_middleware(self, request_data: Dict[str, Any]):
        """Execute middleware chain."""
        for middleware in self.middleware:
            await middleware(request_data)

    async def _dispatch_single(
        self, request_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Dispatch a single JSON-RPC request."""
        start_time = time.time()
        request_id = request_data.get("id")

        try:
            # Validate JSON-RPC structure
            if request_data.get("jsonrpc") != "2.0":
                raise JSONRPCError(
                    JSONRPCError.INVALID_REQUEST,
                    "Invalid JSON-RPC version",
                    request_id=request_id,
                )

            method_name = request_data.get("method")
            if not method_name or not isinstance(method_name, str):
                raise JSONRPCError(
                    JSONRPCError.INVALID_REQUEST,
                    "Invalid or missing method",
                    request_id=request_id,
                )

            params = request_data.get("params", {})

            # Find handler
            handler = self.routes.get(method_name)
            if not handler:
                raise JSONRPCError(
                    JSONRPCError.METHOD_NOT_FOUND,
                    f"Method '{method_name}' not found",
                    request_id=request_id,
                )

            # Run middleware
            await self._run_middleware(request_data)

            # Execute handler with timeout
            try:
                result = await asyncio.wait_for(
                    handler(params), timeout=self.config.request_timeout
                )
            except asyncio.TimeoutError:
                raise JSONRPCError(
                    JSONRPCError.INTERNAL_ERROR,
                    "Request timeout",
                    request_id=request_id,
                )

            # Return response (skip for notifications)
            if request_id is not None:
                response = {"jsonrpc": "2.0", "id": request_id, "result": result}

                if self.config.debug:
                    response["_meta"] = {
                        "execution_time_ms": (time.time() - start_time) * 1000,
                        "method": method_name,
                    }

                return response

            return None

        except JSONRPCError as e:
            self.error_count += 1
            if request_id is not None:
                return {"jsonrpc": "2.0", "id": request_id, "error": e.to_dict()}
            return None
        except Exception as e:
            self.error_count += 1
            logger.exception(f"Unexpected error in method {method_name}: {e}")
            if request_id is not None:
                error_data = (
                    {"traceback": traceback.format_exc()} if self.config.debug else None
                )
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": JSONRPCError(
                        JSONRPCError.INTERNAL_ERROR, "Internal server error", error_data
                    ).to_dict(),
                }
            return None

    async def dispatch(
        self, request_data: Union[Dict, List]
    ) -> Optional[Union[Dict, List]]:
        """Dispatch JSON-RPC requests (single or batch)."""
        self.request_count += 1

        try:
            if isinstance(request_data, list):
                # Batch request
                if not request_data:
                    return {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": JSONRPCError(
                            JSONRPCError.INVALID_REQUEST, "Empty batch"
                        ).to_dict(),
                    }

                # Limit batch size
                if len(request_data) > 100:
                    return {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": JSONRPCError(
                            JSONRPCError.INVALID_REQUEST, "Batch too large"
                        ).to_dict(),
                    }

                results = await asyncio.gather(
                    *[self._dispatch_single(req) for req in request_data],
                    return_exceptions=False,
                )

                filtered_results = [r for r in results if r is not None]
                return filtered_results if filtered_results else None
            else:
                return await self._dispatch_single(request_data)

        except Exception as e:
            logger.exception(f"Fatal error in dispatcher: {e}")
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": JSONRPCError(
                    JSONRPCError.PARSE_ERROR, "Parse error"
                ).to_dict(),
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "methods_registered": len(self.routes),
            "requests_processed": self.request_count,
            "errors_encountered": self.error_count,
            "active_requests": self.active_requests,
            "uptime_seconds": time.time() - getattr(self, '_start_time', time.time()),
        }

    def list_methods(self) -> Dict[str, Any]:
        """List available methods."""
        return {"methods": list(self.routes.keys()), "count": len(self.routes)}


# ============================================================================
# Production HTTP Request Handler
# ============================================================================


class ProductionJSONRPCHandler(BaseHTTPRequestHandler):
    """Production-grade HTTP request handler with proper error handling."""

    def __init__(self, app: JSONRPCApp, *args, **kwargs):
        self.app = app
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """Use our logger instead of stderr."""
        logger.info(f"{self.address_string()} - {format % args}")

    def _send_cors_headers(self):
        """Send CORS headers if enabled."""
        if self.app.config.enable_cors:
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            self.send_header(
                'Access-Control-Allow-Headers', 'Content-Type, Authorization'
            )

    def _send_json_response(self, data: Any, status: int = 200):
        """Send JSON response with proper headers."""
        try:
            response_body = json.dumps(
                data, indent=2 if self.app.config.debug else None
            ).encode('utf-8')
        except Exception as e:
            logger.error(f"Failed to serialize response: {e}")
            response_body = json.dumps({"error": "Serialization error"}).encode('utf-8')
            status = 500

        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(response_body)))
        self.send_header('Server', 'JSONRPCServer/1.0')
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(response_body)

    def _send_error_response(self, message: str, status: int = 400):
        """Send error response."""
        self._send_json_response({"error": message}, status)

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handle GET requests for introspection."""
        path = urlparse(self.path).path

        if path == '/health':
            self._send_json_response(
                {
                    "status": "healthy",
                    "timestamp": time.time(),
                    "stats": self.app.get_stats(),
                }
            )
        elif path == '/methods':
            self._send_json_response(self.app.list_methods())
        elif path == '/stats':
            self._send_json_response(self.app.get_stats())
        else:
            self._send_error_response("Not Found", 404)

    def do_POST(self):
        """Handle JSON-RPC POST requests."""
        self.app.active_requests += 1

        try:
            # Check concurrent request limit
            if self.app.active_requests > self.app.config.max_concurrent_requests:
                self._send_error_response("Too Many Requests", 429)
                return

            # Validate content length
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > self.app.config.max_request_size:
                self._send_error_response("Request Entity Too Large", 413)
                return

            if content_length == 0:
                self._send_error_response("Empty Request Body", 400)
                return

            # Validate content type
            content_type = self.headers.get('Content-Type', '')
            if not content_type.startswith('application/json'):
                self._send_error_response("Invalid Content-Type", 415)
                return

            # Read request body with timeout
            try:
                raw_data = self.rfile.read(content_length)
            except socket.timeout:
                self._send_error_response("Request Timeout", 408)
                return

            # Parse JSON
            try:
                request_data = json.loads(raw_data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": JSONRPCError(
                        JSONRPCError.PARSE_ERROR, f"Parse error: {e}"
                    ).to_dict(),
                }
                self._send_json_response(error_response, 400)
                return

            # Process request asynchronously
            async def process_request():
                return await self.app.dispatch(request_data)

            try:
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                response = loop.run_until_complete(process_request())
                loop.close()
            except Exception as e:
                logger.exception(f"Error processing request: {e}")
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": JSONRPCError(
                        JSONRPCError.INTERNAL_ERROR, "Internal server error"
                    ).to_dict(),
                }
                self._send_json_response(error_response, 500)
                return

            # Send response
            if response is not None:
                self._send_json_response(response)
            else:
                # Notification - no response needed
                self.send_response(204)
                self._send_cors_headers()
                self.end_headers()

        except Exception as e:
            logger.exception(f"Unhandled error in request handler: {e}")
            self._send_error_response("Internal Server Error", 500)
        finally:
            self.app.active_requests -= 1


# ============================================================================
# Production Server Implementation
# ============================================================================


class ProductionJSONRPCServer:
    """Production-quality JSON-RPC server with robust error handling."""

    def __init__(self, app: JSONRPCApp):
        self.app = app
        self.server: Optional[ThreadingHTTPServer] = None
        self._running = False
        self._start_time = None

    def _create_handler_class(self):
        """Create handler class with app instance."""
        app = self.app

        class Handler(ProductionJSONRPCHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(app, *args, **kwargs)

        return Handler

    def _setup_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Setup SSL context if certificates are provided."""
        if not (self.app.config.ssl_cert_path and self.app.config.ssl_key_path):
            return None

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(
            self.app.config.ssl_cert_path, self.app.config.ssl_key_path
        )
        return context

    def start(self):
        """Start the server with graceful shutdown handling."""
        if self._running:
            logger.warning("Server is already running")
            return

        self._start_time = time.time()
        self.app._start_time = self._start_time

        handler_class = self._create_handler_class()

        try:
            self.server = ThreadingHTTPServer(
                (self.app.config.host, self.app.config.port), handler_class
            )

            # Configure socket options
            self.server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server.timeout = self.app.config.request_timeout

            # Setup SSL if configured
            ssl_context = self._setup_ssl_context()
            if ssl_context:
                self.server.socket = ssl_context.wrap_socket(
                    self.server.socket, server_side=True
                )
                logger.info("SSL/TLS enabled")

            self._running = True

            # Setup signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, shutting down gracefully...")
                self.stop()

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            protocol = "https" if ssl_context else "http"
            logger.info(
                f"Server starting on {protocol}://{self.app.config.host}:{self.app.config.port}"
            )
            logger.info(f"Registered methods: {list(self.app.routes.keys())}")

            # Start serving
            self.server.serve_forever()

        except OSError as e:
            if e.errno == 98:  # Address already in use
                logger.error(f"Port {self.app.config.port} is already in use")
            else:
                logger.error(f"Failed to start server: {e}")
            raise
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        finally:
            self._running = False

    def stop(self):
        """Stop the server gracefully."""
        if self.server and self._running:
            logger.info("Stopping server...")
            self.server.shutdown()
            self.server.server_close()
            self._running = False
            logger.info("Server stopped")

    def is_running(self) -> bool:
        """Check if server is running."""
        return self._running


# ============================================================================
# Example Usage with Your Routes
# ============================================================================

# Create app instance
app = JSONRPCApp(
    ServerConfig(host="127.0.0.1", port=8080, debug=True, max_concurrent_requests=50)
)


# Add middleware
@app.middleware_handler
async def logging_middleware(request_data):
    method = request_data.get('method', 'unknown')
    logger.info(f"Processing method: {method}")


# Register your routes
@app.route("get_memory_stats")
async def get_memory_stats(params):
    return MemoryStats(
        size=1024, count=8, traceback="trace", timestamp=time.time(), peak_memory=2048
    ).to_dict()


@app.route("upload_file")
async def upload_file(params):
    try:
        model = FileModel.from_dict(
            {"file_name": params["file_name"], "file_content": params["file_content"]}
        )
        # model.save(pathlib.Path("/tmp/"))  # Uncomment if you want actual saving
        return {"status": "ok", "file_name": model.file_name}
    except Exception as e:
        raise JSONRPCError(JSONRPCError.INVALID_PARAMS, f"Invalid file data: {e}")


@app.route("echo")
async def echo(params):
    """Simple echo method for testing."""
    return {"echo": params}


@app.route("add")
async def add(params):
    """Add two numbers."""
    try:
        a = params.get("a", 0)
        b = params.get("b", 0)
        return {"result": a + b}
    except Exception as e:
        raise JSONRPCError(JSONRPCError.INVALID_PARAMS, f"Invalid parameters: {e}")


# Main execution
if __name__ == '__main__':
    server = ProductionJSONRPCServer(app)

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.stop()
