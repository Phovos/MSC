#!/bin/bash

# combo_server_test.sh - Test script for JSON-RPC server

set -e  # Exit on any error

# Configuration
SERVER_FILE="jsonrpc_server.py"
SERVER_HOST="127.0.0.1"
SERVER_PORT="8698"
SERVER_URL="http://${SERVER_HOST}:${SERVER_PORT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✓${NC} $1"
}

error() {
    echo -e "${RED}✗${NC} $1"
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if server file exists
if [[ ! -f "$SERVER_FILE" ]]; then
    error "Server file '$SERVER_FILE' not found!"
    echo "Please save the server code to '$SERVER_FILE' first."
    exit 1
fi

# Function to check if server is running
check_server() {
    curl -s "${SERVER_URL}/health" > /dev/null 2>&1
    return $?
}

# Function to wait for server to start
wait_for_server() {
    local max_attempts=30
    local attempt=0
    
    log "Waiting for server to start..."
    
    while [[ $attempt -lt $max_attempts ]]; do
        if check_server; then
            success "Server is running!"
            return 0
        fi
        
        attempt=$((attempt + 1))
        sleep 1
        echo -n "."
    done
    
    echo ""
    error "Server failed to start within $max_attempts seconds"
    return 1
}

# Function to stop server
stop_server() {
    log "Stopping server..."
    
    # Find and kill the server process
    local server_pid=$(pgrep -f "$SERVER_FILE" | head -1)
    
    if [[ -n "$server_pid" ]]; then
        kill -TERM "$server_pid" 2>/dev/null || true
        
        # Wait for graceful shutdown
        local count=0
        while [[ $count -lt 10 ]] && kill -0 "$server_pid" 2>/dev/null; do
            sleep 1
            count=$((count + 1))
        done
        
        # Force kill if still running
        if kill -0 "$server_pid" 2>/dev/null; then
            warn "Server didn't stop gracefully, force killing..."
            kill -KILL "$server_pid" 2>/dev/null || true
        fi
        
        success "Server stopped (PID: $server_pid)"
    else
        warn "No server process found"
    fi
}

# Function to start server
start_server() {
    log "Starting JSON-RPC server on ${SERVER_HOST}:${SERVER_PORT}..."
    
    # Start server in background
    python3 "$SERVER_FILE" --host "$SERVER_HOST" --port "$SERVER_PORT" --debug > server.log 2>&1 &
    local server_pid=$!
    
    log "Server started with PID: $server_pid"
    
    # Wait for server to be ready
    if wait_for_server; then
        return 0
    else
        error "Failed to start server"
        log "Server log output:"
        cat server.log
        return 1
    fi
}

# Function to test server endpoints
test_endpoints() {
    log "Testing server endpoints..."
    
    # Test health endpoint
    log "Testing health endpoint..."
    if curl -s "${SERVER_URL}/health" | jq . > /dev/null 2>&1; then
        success "Health endpoint working"
        curl -s "${SERVER_URL}/health" | jq .
    else
        error "Health endpoint failed"
        return 1
    fi
    
    echo ""
    
    # Test methods endpoint
    log "Testing methods endpoint..."
    if curl -s "${SERVER_URL}/methods" | jq . > /dev/null 2>&1; then
        success "Methods endpoint working"
        echo "Available methods:"
        curl -s "${SERVER_URL}/methods" | jq 'keys[]'
    else
        error "Methods endpoint failed"
        return 1
    fi
    
    echo ""
    
    # Test stats endpoint
    log "Testing stats endpoint..."
    if curl -s "${SERVER_URL}/stats" | jq . > /dev/null 2>&1; then
        success "Stats endpoint working"
        curl -s "${SERVER_URL}/stats" | jq .
    else
        error "Stats endpoint failed"
        return 1
    fi
}

# Function to test JSON-RPC methods
test_rpc_methods() {
    log "Testing JSON-RPC methods..."
    
    # Test echo method
    log "Testing echo method..."
    local echo_response=$(curl -s -X POST "${SERVER_URL}" \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"echo","params":{"test":"hello","number":42},"id":1}')
    
    if echo "$echo_response" | jq -e '.result' > /dev/null 2>&1; then
        success "Echo method working"
        echo "$echo_response" | jq .
    else
        error "Echo method failed"
        echo "Response: $echo_response"
        return 1
    fi
    
    echo ""
    
    # Test memory stats method
    log "Testing get_memory_stats method..."
    local memory_response=$(curl -s -X POST "${SERVER_URL}" \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"get_memory_stats","params":{},"id":2}')
    
    if echo "$memory_response" | jq -e '.result.size_bytes' > /dev/null 2>&1; then
        success "Memory stats method working"
        echo "$memory_response" | jq .result
    else
        error "Memory stats method failed"
        echo "Response: $memory_response"
        return 1
    fi
    
    echo ""
    
    # Test text processing method
    log "Testing process_text method..."
    local text_response=$(curl -s -X POST "${SERVER_URL}" \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"process_text","params":{"text":"Hello World","operation":"count"},"id":3}')
    
    if echo "$text_response" | jq -e '.result.word_count' > /dev/null 2>&1; then
        success "Text processing method working"
        echo "$text_response" | jq .result
    else
        error "Text processing method failed"
        echo "Response: $text_response"
        return 1
    fi
    
    echo ""
    
    # Test async method
    log "Testing async_compute method..."
    local async_response=$(curl -s -X POST "${SERVER_URL}" \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"async_compute","params":{"duration":0.5},"id":4}')
    
    if echo "$async_response" | jq -e '.result.computed' > /dev/null 2>&1; then
        success "Async compute method working"
        echo "$async_response" | jq .result
    else
        error "Async compute method failed"
        echo "Response: $async_response"
        return 1
    fi
    
    echo ""
    
    # Test divide numbers method
    log "Testing divide_numbers method..."
    local divide_response=$(curl -s -X POST "${SERVER_URL}" \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"divide_numbers","params":{"a":10,"b":2},"id":5}')
    
    if echo "$divide_response" | jq -e '.result.result' > /dev/null 2>&1; then
        success "Divide numbers method working"
        echo "$divide_response" | jq .result
    else
        error "Divide numbers method failed"
        echo "Response: $divide_response"
        return 1
    fi
    
    echo ""
    
    # Test error handling
    log "Testing error handling..."
    local error_response=$(curl -s -X POST "${SERVER_URL}" \
        -H "Content-Type: application/json" \
        -d '{"jsonrpc":"2.0","method":"divide_numbers","params":{"a":10,"b":0},"id":6}')
    
    if echo "$error_response" | jq -e '.error' > /dev/null 2>&1; then
        success "Error handling working"
        echo "$error_response" | jq .error
    else
        error "Error handling failed"
        echo "Response: $error_response"
        return 1
    fi
}

# Function to test with Python client
test_python_client() {
    log "Testing with Python client..."
    
    python3 "$SERVER_FILE" test
}

# Function to run performance test
performance_test() {
    log "Running performance test..."
    
    # Simple performance test with multiple concurrent requests
    local num_requests=50
    local concurrency=10
    
    log "Sending $num_requests requests with concurrency $concurrency..."
    
    # Create a temporary script for parallel requests
    cat > /tmp/perf_test.sh << 'EOF'
#!/bin/bash
for i in {1..5}; do
    curl -s -X POST "http://127.0.0.1:8698" \
        -H "Content-Type: application/json" \
        -d "{\"jsonrpc\":\"2.0\",\"method\":\"echo\",\"params\":{\"test\":$i},\"id\":$i}" \
        > /dev/null 2>&1
done
EOF
    
    chmod +x /tmp/perf_test.sh
    
    local start_time=$(date +%s.%N)
    
    # Run parallel requests
    for i in $(seq 1 $concurrency); do
        /tmp/perf_test.sh &
    done
    
    wait  # Wait for all background jobs to complete
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc -l)
    local rps=$(echo "scale=2; $num_requests / $duration" | bc -l)
    
    success "Performance test completed"
    log "Total requests: $num_requests"
    log "Duration: ${duration}s"
    log "Requests per second: $rps"
    
    rm -f /tmp/perf_test.sh
}

# Cleanup function
cleanup() {
    log "Cleaning up..."
    stop_server
    rm -f server.log
}

# Set up trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    log "Starting JSON-RPC server test suite..."
    
    # Check dependencies
    if ! command -v python3 &> /dev/null; then
        error "python3 is required but not installed"
        exit 1
    fi
    
    if ! command -v curl &> /dev/null; then
        error "curl is required but not installed"
        exit 1
    fi
    
    if ! command -v jq &> /dev/null; then
        warn "jq not found, JSON output will be raw"
    fi
    
    # Stop any existing server
    stop_server
    
    # Start fresh server
    if ! start_server; then
        error "Failed to start server"
        exit 1
    fi
    
    echo ""
    
    # Run tests
    if test_endpoints; then
        success "Endpoint tests passed"
    else
        error "Endpoint tests failed"
        exit 1
    fi
    
    echo ""
    
    if test_rpc_methods; then
        success "RPC method tests passed"
    else
        error "RPC method tests failed"
        exit 1
    fi
    
    echo ""
    
    if test_python_client; then
        success "Python client tests passed"
    else
        error "Python client tests failed"
        exit 1
    fi
    
    echo ""
    
    if command -v bc &> /dev/null; then
        performance_test
    else
        warn "Skipping performance test (bc not available)"
    fi
    
    echo ""
    success "All tests completed successfully!"
    
    log "Server will continue running. Press Ctrl+C to stop."
    
    # Keep server running for manual testing
    wait
}

# Handle command line arguments
case "${1:-}" in
    "start")
        start_server
        wait
        ;;
    "stop")
        stop_server
        ;;
    "test-only")
        if check_server; then
            test_endpoints && test_rpc_methods && test_python_client
        else
            error "Server is not running. Start it first with: $0 start"
            exit 1
        fi
        ;;
    *)
        main
        ;;
esac