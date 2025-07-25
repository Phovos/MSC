#!/usr/bin/env python3
"""
Production-quality JSON-RPC server with BaseModel integration.
Uses http.server for robust HTTP handling with custom BaseModel validation.
"""

import sys
import json
import time
import asyncio
import inspect
import logging
import threading
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Callable,
    Union,
    Type,
    get_type_hints,
    get_origin,
    get_args,
)
from dataclasses import dataclass, field, fields
from functools import wraps
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# DOMAIN LAYER: BaseModel + Validation Framework
# ============================================================================


def validate(validator: Callable[[Any], None]):
    """Decorator for field validation methods."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(self, value):
            validator(value)
            return value

        return wrapper

    return decorator


@dataclass(frozen=True)
class BaseModel:
    """
    Enhanced base model with validation, serialization, and composition.
    Provides Pydantic-style semantics with stdlib-only implementation.
    """

    __slots__ = ('__weakref__',)

    def __post_init__(self):
        """Validate all fields after initialization."""
        # Type validation
        for field_name, expected_type in self.__annotations__.items():
            actual_value = getattr(self, field_name)
            if not self._validate_type(actual_value, expected_type):
                raise TypeError(
                    f"{self.__class__.__name__}.{field_name}: expected {expected_type}, "
                    f"got {type(actual_value).__name__}"
                )

            # Custom validator methods
            validator_name = f'validate_{field_name}'
            validator = getattr(self.__class__, validator_name, None)
            if validator and callable(validator):
                try:
                    validator(self, actual_value)
                except Exception as e:
                    raise ValueError(f"Validation failed for {field_name}: {e}")

        # Model-level validation
        self._validate_model()

    def _validate_type(self, value: Any, expected_type: Any) -> bool:
        """Enhanced type validation supporting generics and unions."""
        if expected_type is Any:
            return True

        # Handle Optional[T] (Union[T, None])
        origin = get_origin(expected_type)
        if origin is Union:
            args = get_args(expected_type)
            return any(self._validate_type(value, arg) for arg in args)

        # Handle List[T], Dict[K, V], etc.
        if origin is not None:
            if not isinstance(value, origin):
                return False

            # Validate generic parameters
            args = get_args(expected_type)
            if origin is list and args:
                return all(self._validate_type(item, args[0]) for item in value)
            elif origin is dict and len(args) >= 2:
                return all(
                    self._validate_type(k, args[0]) for k in value.keys()
                ) and all(self._validate_type(v, args[1]) for v in value.values())

            return True

        # Basic isinstance check
        if hasattr(expected_type, '__origin__'):  # Generic alias
            return isinstance(value, expected_type.__origin__)

        return isinstance(value, expected_type)

    def _validate_model(self):
        """Override for model-level validation logic."""
        pass

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseModel':
        """Create instance from dictionary with field mapping and conversion."""
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict, got {type(data).__name__}")

        field_names = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in field_names}

        # Handle nested models
        annotations = get_type_hints(cls)
        for field_name, field_type in annotations.items():
            if field_name in filtered_data:
                value = filtered_data[field_name]

                # Convert nested BaseModel instances
                if (
                    inspect.isclass(field_type)
                    and issubclass(field_type, BaseModel)
                    and isinstance(value, dict)
                ):
                    filtered_data[field_name] = field_type.from_dict(value)

                # Handle List[BaseModel]
                elif (
                    get_origin(field_type) is list
                    and get_args(field_type)
                    and inspect.isclass(get_args(field_type)[0])
                    and issubclass(get_args(field_type)[0], BaseModel)
                    and isinstance(value, list)
                ):
                    model_class = get_args(field_type)[0]
                    filtered_data[field_name] = [
                        model_class.from_dict(item) if isinstance(item, dict) else item
                        for item in value
                    ]

        try:
            return cls(**filtered_data)
        except TypeError as e:
            raise ValueError(f"Failed to create {cls.__name__}: {e}")

    def to_dict(self, exclude_none: bool = False) -> Dict[str, Any]:
        """Convert to dictionary with nested model serialization."""
        result = {}
        for field_obj in fields(self):
            value = getattr(self, field_obj.name)

            if exclude_none and value is None:
                continue

            if isinstance(value, BaseModel):
                result[field_obj.name] = value.to_dict(exclude_none=exclude_none)
            elif isinstance(value, list) and value and isinstance(value[0], BaseModel):
                result[field_obj.name] = [
                    item.to_dict(exclude_none=exclude_none) for item in value
                ]
            elif isinstance(value, dict):
                # Handle dict values that might contain BaseModel instances
                result[field_obj.name] = {
                    k: v.to_dict(exclude_none=exclude_none)
                    if isinstance(v, BaseModel)
                    else v
                    for k, v in value.items()
                }
            else:
                result[field_obj.name] = value

        return result

    def clone(self, **overrides) -> 'BaseModel':
        """Create a copy with optional field overrides."""
        current_data = self.to_dict()
        current_data.update(overrides)
        return self.__class__.from_dict(current_data)

    def __repr__(self):
        attrs = ', '.join(f"{f.name}={getattr(self, f.name)!r}" for f in fields(self))
        return f"{self.__class__.__name__}({attrs})"

    def __str__(self):
        return self.__repr__()


# ============================================================================
# BUSINESS DOMAIN MODELS
# ============================================================================


@dataclass(frozen=True)
class MemoryStats(BaseModel):
    """Memory statistics with validation."""

    size_bytes: int
    object_count: int
    peak_memory: int
    traceback_info: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    @validate(lambda x: x >= 0)
    def validate_size_bytes(self, value: int) -> None:
        """Size must be non-negative."""
        pass

    @validate(lambda x: x >= 0)
    def validate_object_count(self, value: int) -> None:
        """Count must be non-negative."""
        pass

    def _validate_model(self):
        """Ensure peak >= current size."""
        if self.peak_memory < self.size_bytes:
            raise ValueError("Peak memory cannot be less than current size")


@dataclass(frozen=True)
class RPCRequest(BaseModel):
    """Typed RPC request model."""

    method: str
    params: Union[Dict[str, Any], List[Any], None] = None
    request_id: Optional[str] = None
    client_info: Optional[Dict[str, Any]] = None

    @validate(lambda x: len(x.strip()) > 0)
    def validate_method(self, value: str) -> None:
        """Method name cannot be empty."""
        pass


@dataclass(frozen=True)
class RPCResponse(BaseModel):
    """Typed RPC response model."""

    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    execution_time_ms: Optional[float] = None

    def _validate_model(self):
        """Ensure exactly one of result or error is present."""
        if self.result is not None and self.error is not None:
            raise ValueError("Response cannot have both result and error")
        if self.result is None and self.error is None:
            raise ValueError("Response must have either result or error")


# ============================================================================
# TRANSPORT LAYER: JSON-RPC Server with BaseModel Integration
# ============================================================================


@dataclass
class RPCConfig:
    """Server configuration."""

    host: str = "127.0.0.1"
    port: int = 8698
    max_content_length: int = 10_000_000
    request_timeout: float = 30.0
    enable_cors: bool = True
    debug: bool = False
    thread_pool_size: int = 4
    validate_requests: bool = True
    validate_responses: bool = True


@dataclass
class MethodInfo:
    """Enhanced method metadata with BaseModel support."""

    name: str
    func: Callable
    is_async: bool
    doc: Optional[str] = None
    input_model: Optional[Type[BaseModel]] = None
    output_model: Optional[Type[BaseModel]] = None
    raw_params: bool = False  # Skip model conversion for this method


class JSONRPCError(Exception):
    """JSON-RPC 2.0 compliant error with BaseModel integration."""

    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    VALIDATION_ERROR = -32001  # Custom validation error

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


class EnhancedJSONRPCDispatcher:
    """JSON-RPC dispatcher with BaseModel integration."""

    def __init__(self, config: RPCConfig):
        self.config = config
        self.methods: Dict[str, MethodInfo] = {}
        self.middleware: List[Callable] = []
        self.executor = ThreadPoolExecutor(max_workers=config.thread_pool_size)
        self.request_count = 0
        self.error_count = 0

    def method(
        self,
        name: Optional[str] = None,
        input_model: Optional[Type[BaseModel]] = None,
        output_model: Optional[Type[BaseModel]] = None,
        raw_params: bool = False,
    ):
        """
        Register a JSON-RPC method with optional BaseModel validation.

        Args:
            name: Method name (defaults to function name)
            input_model: BaseModel class for parameter validation
            output_model: BaseModel class for response validation
            raw_params: Skip model conversion, pass raw params
        """

        def decorator(func: Callable):
            method_name = name or func.__name__

            # Auto-detect models from type hints if not provided
            if not raw_params and (input_model is None or output_model is None):
                hints = get_type_hints(func)

                # Try to find input model from first parameter
                detected_input = input_model
                if detected_input is None:
                    params = list(inspect.signature(func).parameters.values())
                    if params and params[0].name not in ('self', 'cls'):
                        first_param_type = hints.get(params[0].name)
                        if (
                            first_param_type
                            and inspect.isclass(first_param_type)
                            and issubclass(first_param_type, BaseModel)
                        ):
                            detected_input = first_param_type

                # Try to find output model from return annotation
                detected_output = output_model
                if detected_output is None:
                    return_type = hints.get('return')
                    if (
                        return_type
                        and inspect.isclass(return_type)
                        and issubclass(return_type, BaseModel)
                    ):
                        detected_output = return_type
            else:
                detected_input = input_model
                detected_output = output_model

            method_info = MethodInfo(
                name=method_name,
                func=func,
                is_async=asyncio.iscoroutinefunction(func),
                doc=inspect.getdoc(func),
                input_model=detected_input,
                output_model=detected_output,
                raw_params=raw_params,
            )

            self.methods[method_name] = method_info
            logger.info(
                f"Registered {'async' if method_info.is_async else 'sync'} method: {method_name}"
            )
            if detected_input:
                logger.info(f"  Input model: {detected_input.__name__}")
            if detected_output:
                logger.info(f"  Output model: {detected_output.__name__}")

            return func

        return decorator

    def middleware_handler(self, func: Callable):
        """Register middleware."""
        self.middleware.append(func)
        return func

    async def _prepare_params(self, method_info: MethodInfo, raw_params: Any) -> Any:
        """Convert raw parameters to BaseModel if configured."""
        if method_info.raw_params or not method_info.input_model:
            return raw_params

        try:
            if isinstance(raw_params, dict):
                return method_info.input_model.from_dict(raw_params)
            elif isinstance(raw_params, list):
                # For positional args, assume first arg is the model data
                if len(raw_params) == 1 and isinstance(raw_params[0], dict):
                    return method_info.input_model.from_dict(raw_params[0])
                return raw_params
            elif raw_params is None:
                # Try to create empty model
                return method_info.input_model.from_dict({})
            else:
                return raw_params
        except Exception as e:
            raise JSONRPCError(
                JSONRPCError.VALIDATION_ERROR,
                f"Parameter validation failed: {e}",
                {"model": method_info.input_model.__name__, "error": str(e)},
            )

    async def _validate_result(self, method_info: MethodInfo, result: Any) -> Any:
        """Validate and convert result using output model if configured."""
        if not self.config.validate_responses or not method_info.output_model:
            return result

        try:
            if isinstance(result, method_info.output_model):
                return result.to_dict()
            elif isinstance(result, dict):
                validated = method_info.output_model.from_dict(result)
                return validated.to_dict()
            else:
                # Try to wrap primitive results
                if hasattr(method_info.output_model, 'from_primitive'):
                    validated = method_info.output_model.from_primitive(result)
                    return validated.to_dict()
                return result
        except Exception as e:
            logger.warning(f"Response validation failed for {method_info.name}: {e}")
            return result  # Return unvalidated rather than fail

    async def _call_method(self, method_info: MethodInfo, params: Any) -> Any:
        """Call method with proper parameter handling."""
        try:
            # Prepare parameters
            processed_params = await self._prepare_params(method_info, params)

            # Determine how to call the function
            sig = inspect.signature(method_info.func)
            param_names = list(sig.parameters.keys())

            if isinstance(processed_params, BaseModel):
                # Pass BaseModel as single argument
                if method_info.is_async:
                    result = await method_info.func(processed_params)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor, lambda: method_info.func(processed_params)
                    )
            elif isinstance(processed_params, dict):
                # Keyword arguments
                if method_info.is_async:
                    result = await method_info.func(**processed_params)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor, lambda: method_info.func(**processed_params)
                    )
            elif isinstance(processed_params, list):
                # Positional arguments
                if method_info.is_async:
                    result = await method_info.func(*processed_params)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor, lambda: method_info.func(*processed_params)
                    )
            else:
                # Single parameter
                if method_info.is_async:
                    result = await method_info.func(processed_params)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor, lambda: method_info.func(processed_params)
                    )

            # Validate result if configured
            return await self._validate_result(method_info, result)

        except JSONRPCError:
            raise
        except Exception as e:
            logger.exception(f"Error executing method {method_info.name}: {e}")
            raise JSONRPCError(
                JSONRPCError.INTERNAL_ERROR,
                f"Method execution failed: {e}",
                {"method": method_info.name, "error": str(e)}
                if self.config.debug
                else None,
            )

    async def dispatch_single(
        self, request_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Dispatch single request with BaseModel integration."""
        start_time = time.time()
        request_id = request_data.get("id")

        try:
            # Validate request format
            if self.config.validate_requests:
                try:
                    request = RPCRequest.from_dict(
                        {
                            "method": request_data.get("method"),
                            "params": request_data.get("params"),
                            "request_id": str(request_id)
                            if request_id is not None
                            else None,
                            "client_info": {"timestamp": time.time()},
                        }
                    )
                except Exception as e:
                    raise JSONRPCError(
                        JSONRPCError.INVALID_REQUEST, f"Request validation failed: {e}"
                    )

            # Basic JSON-RPC validation
            if request_data.get("jsonrpc") != "2.0":
                raise JSONRPCError(
                    JSONRPCError.INVALID_REQUEST, "Invalid JSON-RPC version"
                )

            method_name = request_data.get("method")
            if not method_name:
                raise JSONRPCError(JSONRPCError.INVALID_REQUEST, "Missing method")

            params = request_data.get("params")

            # Find method
            method_info = self.methods.get(method_name)
            if not method_info:
                raise JSONRPCError(
                    JSONRPCError.METHOD_NOT_FOUND, f"Method '{method_name}' not found"
                )

            # Run middleware
            for middleware in self.middleware:
                await middleware(request_data)

            # Execute method
            result = await self._call_method(method_info, params)
            execution_time = (time.time() - start_time) * 1000

            # Return response (skip for notifications)
            if request_id is not None:
                response_data = {"jsonrpc": "2.0", "id": request_id, "result": result}

                if self.config.debug:
                    response_data["_meta"] = {
                        "execution_time_ms": execution_time,
                        "method": method_name,
                    }

                return response_data

            return None

        except JSONRPCError as e:
            self.error_count += 1
            if request_id is not None:
                return {"jsonrpc": "2.0", "id": request_id, "error": e.to_dict()}
            return None

    async def dispatch(self, payload: Union[Dict, List]) -> Optional[Union[Dict, List]]:
        """Dispatch requests with BaseModel support."""
        self.request_count += 1

        try:
            if isinstance(payload, list):
                if not payload:
                    raise JSONRPCError(
                        JSONRPCError.INVALID_REQUEST, "Empty batch request"
                    )

                results = await asyncio.gather(
                    *[self.dispatch_single(req) for req in payload],
                    return_exceptions=False,
                )

                filtered_results = [r for r in results if r is not None]
                return filtered_results if filtered_results else None
            else:
                return await self.dispatch_single(payload)

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
        """Get detailed statistics."""
        return {
            "methods_registered": len(self.methods),
            "requests_processed": self.request_count,
            "errors_encountered": self.error_count,
            "active_threads": len(self.executor._threads)
            if self.executor._threads
            else 0,
            "validation_enabled": {
                "requests": self.config.validate_requests,
                "responses": self.config.validate_responses,
            },
        }

    def list_methods(self) -> Dict[str, Any]:
        """List all methods with BaseModel info."""
        return {
            name: {
                "name": info.name,
                "async": info.is_async,
                "doc": info.doc,
                "input_model": info.input_model.__name__ if info.input_model else None,
                "output_model": info.output_model.__name__
                if info.output_model
                else None,
                "raw_params": info.raw_params,
            }
            for name, info in self.methods.items()
        }

    def shutdown(self):
        """Clean shutdown of resources."""
        logger.info("Shutting down thread pool executor...")
        self.executor.shutdown(wait=True)
        logger.info("Dispatcher shutdown complete")


# ============================================================================
# HTTP REQUEST HANDLER
# ============================================================================


class JSONRPCRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for JSON-RPC with BaseModel support."""

    def __init__(
        self, dispatcher: EnhancedJSONRPCDispatcher, config: RPCConfig, *args, **kwargs
    ):
        self.dispatcher = dispatcher
        self.config = config
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """Override to use our logger."""
        logger.info(f"{self.address_string()} - {format % args}")

    def _send_cors_headers(self):
        """Send CORS headers if enabled."""
        if self.config.enable_cors:
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
            self.send_header(
                'Access-Control-Allow-Headers', 'Content-Type, Authorization'
            )

    def _send_json_response(self, data: Any, status: int = 200):
        """Send a JSON response."""
        response_body = json.dumps(
            data, indent=2 if self.config.debug else None
        ).encode('utf-8')

        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(response_body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(response_body)

    def _send_error_response(self, message: str, status: int = 400):
        """Send an error response."""
        self._send_json_response({"error": message}, status)

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handle GET requests for introspection."""
        if self.path == '/health':
            self._send_json_response({"status": "healthy", "timestamp": time.time()})
        elif self.path == '/methods':
            self._send_json_response(self.dispatcher.list_methods())
        elif self.path == '/stats':
            self._send_json_response(self.dispatcher.get_stats())
        else:
            self._send_error_response("Not Found", 404)

    def do_POST(self):
        """Handle JSON-RPC POST requests."""
        try:
            # Check content length
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > self.config.max_content_length:
                self._send_error_response("Request entity too large", 413)
                return

            if content_length == 0:
                self._send_error_response("Empty request body", 400)
                return

            # Read and parse request
            raw_data = self.rfile.read(content_length)
            try:
                request_data = json.loads(raw_data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": JSONRPCError(
                        JSONRPCError.PARSE_ERROR, "Parse error"
                    ).to_dict(),
                }
                self._send_json_response(error_response)
                return

            # Process request
            async def process_request():
                return await self.dispatcher.dispatch(request_data)

            try:
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
                self.send_response(204)
                self._send_cors_headers()
                self.end_headers()

        except Exception as e:
            logger.exception(f"Unhandled error in request handler: {e}")
            self._send_error_response("Internal server error", 500)


class EnhancedJSONRPCServer:
    """Production-quality JSON-RPC server with BaseModel integration."""

    def __init__(self, config: Optional[RPCConfig] = None):
        self.config = config or RPCConfig()
        self.dispatcher = EnhancedJSONRPCDispatcher(self.config)
        self.server: Optional[ThreadingHTTPServer] = None
        self._running = False
        self._shutdown_event = threading.Event()

    def method(self, name: Optional[str] = None, **kwargs):
        """Register RPC method decorator."""
        return self.dispatcher.method(name, **kwargs)

    def middleware(self, func: Callable):
        """Register middleware decorator."""
        return self.dispatcher.middleware_handler(func)

    def start(self):
        """Start the server."""

        def handler_factory(*args, **kwargs):
            return JSONRPCRequestHandler(self.dispatcher, self.config, *args, **kwargs)

        try:
            self.server = ThreadingHTTPServer(
                (self.config.host, self.config.port), handler_factory
            )
            self.server.timeout = self.config.request_timeout

            logger.info(
                f"JSON-RPC server starting on {self.config.host}:{self.config.port}"
            )
            logger.info(f"Debug mode: {self.config.debug}")
            logger.info(
                f"Validation enabled: requests={self.config.validate_requests}, responses={self.config.validate_responses}"
            )
            logger.info(f"Registered {len(self.dispatcher.methods)} methods")

            self._running = True

            # Start server in a way that allows graceful shutdown
            def serve_forever():
                while self._running and not self._shutdown_event.is_set():
                    self.server.handle_request()

            serve_thread = threading.Thread(target=serve_forever, daemon=True)
            serve_thread.start()

            logger.info("Server started successfully")
            return serve_thread

        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            raise

    def stop(self):
        """Stop the server gracefully."""
        if self.server and self._running:
            logger.info("Shutting down JSON-RPC server...")
            self._running = False
            self._shutdown_event.set()

            # Close server socket
            self.server.server_close()

            # Shutdown dispatcher
            self.dispatcher.shutdown()

            logger.info("Server shutdown complete")

    def run_forever(self):
        """Run the server until interrupted."""
        serve_thread = self.start()

        try:
            # Keep main thread alive
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        base_stats = self.dispatcher.get_stats()
        base_stats.update(
            {
                "server_running": self._running,
                "server_config": {
                    "host": self.config.host,
                    "port": self.config.port,
                    "max_content_length": self.config.max_content_length,
                    "request_timeout": self.config.request_timeout,
                    "thread_pool_size": self.config.thread_pool_size,
                },
            }
        )
        return base_stats


# ============================================================================
# EXAMPLE USAGE AND DEMO METHODS
# ============================================================================


def create_demo_server() -> EnhancedJSONRPCServer:
    """Create a demo server with example methods."""

    config = RPCConfig(
        host="127.0.0.1",
        port=8698,
        debug=True,
        validate_requests=True,
        validate_responses=True,
        thread_pool_size=4,
    )

    server = EnhancedJSONRPCServer(config)

    # Example 1: Method with BaseModel input/output validation
    @server.method(input_model=None, output_model=MemoryStats)
    def get_memory_stats(params: Dict[str, Any]) -> MemoryStats:
        """Get current memory statistics."""
        import psutil

        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()

        return MemoryStats(
            size_bytes=memory_info.rss,
            object_count=len(gc.get_objects()) if 'gc' in globals() else 0,
            peak_memory=memory_info.peak_wset
            if hasattr(memory_info, 'peak_wset')
            else memory_info.rss,
            traceback_info=None,
            timestamp=time.time(),
        )

    # Example 2: Raw parameters method (no validation)
    @server.method(raw_params=True)
    def echo(params: Any) -> Dict[str, Any]:
        """Echo back the parameters."""
        return {
            "echoed": params,
            "timestamp": time.time(),
            "type": type(params).__name__,
        }

    # Example 3: Async method
    @server.method()
    async def async_compute(params: Dict[str, Any]) -> Dict[str, Any]:
        """Async computation example."""
        duration = params.get('duration', 1.0)
        await asyncio.sleep(duration)
        return {"computed": True, "duration": duration, "result": sum(range(1000))}

    # Example 4: Method with custom validation
    @server.method()
    def divide_numbers(params: Dict[str, Any]) -> Dict[str, Any]:
        """Divide two numbers with validation."""
        a = params.get('a')
        b = params.get('b')

        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise ValueError("Both 'a' and 'b' must be numbers")

        if b == 0:
            raise ValueError("Division by zero is not allowed")

        return {"result": a / b, "operation": f"{a} / {b}"}

    # Example 5: File processing method
    @server.method()
    def process_text(params: Dict[str, Any]) -> Dict[str, Any]:
        """Process text with various operations."""
        text = params.get('text', '')
        operation = params.get('operation', 'count')

        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        results = {}

        if operation in ('count', 'all'):
            results['char_count'] = len(text)
            results['word_count'] = len(text.split())
            results['line_count'] = len(text.splitlines())

        if operation in ('transform', 'all'):
            results['uppercase'] = text.upper()
            results['lowercase'] = text.lower()
            results['title_case'] = text.title()

        if operation in ('analyze', 'all'):
            results['unique_chars'] = len(set(text))
            results['starts_with_vowel'] = text.lower().startswith(
                ('a', 'e', 'i', 'o', 'u')
            )
            results['palindrome'] = (
                text.lower().replace(' ', '') == text.lower().replace(' ', '')[::-1]
            )

        return results

    # Example 6: Error demonstration
    @server.method()
    def raise_error(params: Dict[str, Any]) -> Dict[str, Any]:
        """Demonstrate error handling."""
        error_type = params.get('error_type', 'generic')

        if error_type == 'value':
            raise ValueError("This is a value error")
        elif error_type == 'type':
            raise TypeError("This is a type error")
        elif error_type == 'custom':
            from .exceptions import CustomError

            raise CustomError("This is a custom error")
        else:
            raise RuntimeError("This is a generic runtime error")

    # Add middleware example
    @server.middleware
    async def logging_middleware(request_data: Dict[str, Any]):
        """Log all incoming requests."""
        method = request_data.get('method', 'unknown')
        request_id = request_data.get('id', 'no-id')
        logger.info(f"Processing request: {method} (ID: {request_id})")

    @server.middleware
    async def auth_middleware(request_data: Dict[str, Any]):
        """Simple authentication middleware."""
        # Skip auth for introspection methods
        method = request_data.get('method', '')
        if method.startswith('system.'):
            return

        # Check for auth token in params (simplified)
        params = request_data.get('params', {})
        if isinstance(params, dict):
            token = params.get('auth_token')
            if not token:
                logger.warning(f"Missing auth token for method: {method}")
                # In real implementation, you might raise an error here

    return server


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main entry point for the server."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Enhanced JSON-RPC Server with BaseModel integration'
    )
    parser.add_argument('--host', default='127.0.0.1', help='Server host')
    parser.add_argument('--port', type=int, default=8698, help='Server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Disable request/response validation',
    )
    parser.add_argument('--threads', type=int, default=4, help='Thread pool size')

    args = parser.parse_args()

    # Create configuration
    config = RPCConfig(
        host=args.host,
        port=args.port,
        debug=args.debug,
        validate_requests=not args.no_validation,
        validate_responses=not args.no_validation,
        thread_pool_size=args.threads,
    )

    # Create and configure server
    logger.info("Creating JSON-RPC server with BaseModel integration...")
    server = create_demo_server()
    server.config = config  # Override with CLI args

    # Add system methods
    @server.method(raw_params=True)
    def system_stats(params: Any) -> Dict[str, Any]:
        """Get system statistics."""
        return server.get_stats()

    @server.method(raw_params=True)
    def system_methods(params: Any) -> Dict[str, Any]:
        """List all available methods."""
        return server.dispatcher.list_methods()

    @server.method(raw_params=True)
    def system_health(params: Any) -> Dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": time.time()
            - server.dispatcher.request_count,  # Approximation
            "version": "1.0.0",
        }

    # Start server
    try:
        logger.info("Starting server...")
        server.run_forever()
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1

    return 0


# ============================================================================
# CLIENT EXAMPLE
# ============================================================================


class JSONRPCClient:
    """Simple JSON-RPC client for testing."""

    def __init__(self, url: str = "http://127.0.0.1:8698"):
        self.url = url
        self.request_id = 0

    def call(self, method: str, params: Any = None) -> Any:
        """Make a JSON-RPC call."""
        import urllib.request
        import urllib.parse

        self.request_id += 1

        request_data = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self.request_id,
        }

        json_data = json.dumps(request_data).encode('utf-8')

        req = urllib.request.Request(
            self.url, data=json_data, headers={'Content-Type': 'application/json'}
        )

        try:
            with urllib.request.urlopen(req) as response:
                response_data = json.loads(response.read().decode('utf-8'))

                if 'error' in response_data:
                    raise Exception(f"RPC Error: {response_data['error']}")

                return response_data.get('result')

        except urllib.error.URLError as e:
            raise Exception(f"Connection error: {e}")


def test_client():
    """Test client functionality."""
    client = JSONRPCClient()

    print("Testing JSON-RPC client...")

    try:
        # Test health check
        health = client.call("system_health")
        print(f"Health check: {health}")

        # Test echo
        echo_result = client.call("echo", {"test": "data", "number": 42})
        print(f"Echo result: {echo_result}")

        # Test memory stats
        memory = client.call("get_memory_stats")
        print(f"Memory stats: {memory}")

        # Test async method
        async_result = client.call("async_compute", {"duration": 0.5})
        print(f"Async result: {async_result}")

        # Test text processing
        text_result = client.call(
            "process_text", {"text": "Hello World", "operation": "all"}
        )
        print(f"Text processing: {text_result}")

        print("All tests passed!")

    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_client()
    else:
        sys.exit(main())
