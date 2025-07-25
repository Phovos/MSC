# Start server
python server.py --host 127.0.0.1 --port 8698 --debug

# Test client
python combo_server.py test

# Health check
curl http://127.0.0.1:8698/health

# Call method
curl -X POST http://127.0.0.1:8698 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"echo","params":{"test":123},"id":1}'