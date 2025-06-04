#!/bin/bash
"""
Shell script with curl examples for testing vLLM Hidden States API

This script provides ready-to-use curl commands to test the hidden states functionality.

Usage:
    chmod +x test_hidden_states_curl.sh
    ./test_hidden_states_curl.sh [HOST] [PORT] [MODEL]

Examples:
    ./test_hidden_states_curl.sh
    ./test_hidden_states_curl.sh localhost 8000 meta-llama/Llama-3.2-1B-Instruct
"""

# Configuration
HOST=${1:-localhost}
PORT=${2:-8000}
MODEL=${3:-"meta-llama/Llama-3.2-1B-Instruct"}
BASE_URL="http://$HOST:$PORT"

echo "🚀 Testing vLLM Hidden States API"
echo "   Server: $BASE_URL"
echo "   Model: $MODEL"
echo "=" | sed 's/./=/g' | head -c 60; echo

# Check server health
echo "🏥 Checking server health..."
HEALTH_RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/health_response "$BASE_URL/health" 2>/dev/null)
if [ "$HEALTH_RESPONSE" = "200" ]; then
    echo "✅ Server is healthy"
else
    echo "❌ Server is not healthy (HTTP $HEALTH_RESPONSE)"
    echo "   Please start vLLM server: VLLM_USE_V1=1 vllm serve $MODEL"
    exit 1
fi
echo

# Test 1: Chat Completion without Hidden States (Baseline)
echo "🧪 Test 1: Chat Completion without Hidden States"
echo "Request:"
cat << EOF
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "Hello! How are you?"}],
  "max_tokens": 10,
  "temperature": 0.7
}
EOF
echo
echo "Response:"
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Hello! How are you?\"}],
    \"max_tokens\": 10,
    \"temperature\": 0.7
  }" | jq '.'
echo
echo "=" | sed 's/./=/g' | head -c 60; echo

# Test 2: Chat Completion with Hidden States
echo "🧪 Test 2: Chat Completion with Hidden States"
echo "Request:"
cat << EOF
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "max_tokens": 10,
  "temperature": 0.7,
  "return_hidden_states": true,
  "hidden_states_for_tokens": [-1]
}
EOF
echo
echo "Response:"
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France?\"}],
    \"max_tokens\": 10,
    \"temperature\": 0.7,
    \"return_hidden_states\": true,
    \"hidden_states_for_tokens\": [-1]
  }" | jq '.'
echo
echo "=" | sed 's/./=/g' | head -c 60; echo

# Test 3: Completion without Hidden States (Baseline)
echo "🧪 Test 3: Completion without Hidden States"
echo "Request:"
cat << EOF
{
  "model": "$MODEL",
  "prompt": "The capital of France is",
  "max_tokens": 5,
  "temperature": 0.7
}
EOF
echo
echo "Response:"
curl -s -X POST "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 5,
    \"temperature\": 0.7
  }" | jq '.'
echo
echo "=" | sed 's/./=/g' | head -c 60; echo

# Test 4: Completion with Hidden States
echo "🧪 Test 4: Completion with Hidden States"
echo "Request:"
cat << EOF
{
  "model": "$MODEL",
  "prompt": "The capital of France is",
  "max_tokens": 5,
  "temperature": 0.7,
  "return_hidden_states": true,
  "hidden_states_for_tokens": [-1]
}
EOF
echo
echo "Response:"
curl -s -X POST "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"The capital of France is\",
    \"max_tokens\": 5,
    \"temperature\": 0.7,
    \"return_hidden_states\": true,
    \"hidden_states_for_tokens\": [-1]
  }" | jq '.'
echo
echo "=" | sed 's/./=/g' | head -c 60; echo

# Test 5: Streaming Chat Completion with Hidden States
echo "🧪 Test 5: Streaming Chat Completion with Hidden States"
echo "Request:"
cat << EOF
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "Write a short story."}],
  "max_tokens": 20,
  "temperature": 0.7,
  "stream": true,
  "return_hidden_states": true,
  "hidden_states_for_tokens": [-1]
}
EOF
echo
echo "Response (streaming):"
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Write a short story.\"}],
    \"max_tokens\": 20,
    \"temperature\": 0.7,
    \"stream\": true,
    \"return_hidden_states\": true,
    \"hidden_states_for_tokens\": [-1]
  }"
echo
echo "=" | sed 's/./=/g' | head -c 60; echo

# Test 6: Multiple Token Positions
echo "🧪 Test 6: Hidden States for Multiple Token Positions"
echo "Request:"
cat << EOF
{
  "model": "$MODEL",
  "prompt": "The quick brown fox jumps over the lazy dog",
  "max_tokens": 5,
  "temperature": 0.7,
  "return_hidden_states": true,
  "hidden_states_for_tokens": [0, 5, -1]
}
EOF
echo
echo "Response:"
curl -s -X POST "$BASE_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"prompt\": \"The quick brown fox jumps over the lazy dog\",
    \"max_tokens\": 5,
    \"temperature\": 0.7,
    \"return_hidden_states\": true,
    \"hidden_states_for_tokens\": [0, 5, -1]
  }" | jq '.'
echo
echo "=" | sed 's/./=/g' | head -c 60; echo

echo "🎉 All tests completed!"
echo
echo "📝 Notes:"
echo "   - Hidden states should appear in the 'hidden_states' field of choices"
echo "   - Hidden states are extracted for the final token by default (position -1)"
echo "   - Multiple token positions can be specified in 'hidden_states_for_tokens'"
echo "   - Baseline tests should NOT include 'hidden_states' field"
echo "   - Server must be started with VLLM_USE_V1=1 for hidden states support"