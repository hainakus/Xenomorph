#!/bin/bash
# Script to verify coordinator keypair consistency

COORDINATOR_PRIVKEY_FILE="/tmp/genetics-l2-nih2.db.key"
COORDINATOR_URL="http://localhost:8091"

echo "=== Coordinator Keypair Check ==="
echo ""

# Check if private key file exists
if [ ! -f "$COORDINATOR_PRIVKEY_FILE" ]; then
    echo "❌ Private key file not found: $COORDINATOR_PRIVKEY_FILE"
    echo "   Coordinator needs to be started first to generate keypair"
    exit 1
fi

# Read private key
PRIVKEY=$(cat "$COORDINATOR_PRIVKEY_FILE")
echo "✓ Private key file exists"
echo "  Length: ${#PRIVKEY} chars"
echo "  First 16 chars: ${PRIVKEY:0:16}..."
echo ""

# Fetch public key from coordinator API
echo "Fetching public key from coordinator API..."
PUBKEY_RESPONSE=$(curl -s "$COORDINATOR_URL/pubkey")

if [ $? -ne 0 ]; then
    echo "❌ Failed to fetch public key from $COORDINATOR_URL/pubkey"
    echo "   Is the coordinator running?"
    exit 1
fi

PUBKEY=$(echo "$PUBKEY_RESPONSE" | jq -r '.pubkey' 2>/dev/null)

if [ -z "$PUBKEY" ] || [ "$PUBKEY" = "null" ]; then
    echo "❌ Invalid response from coordinator:"
    echo "   $PUBKEY_RESPONSE"
    exit 1
fi

echo "✓ Public key fetched from API"
echo "  Length: ${#PUBKEY} chars"
echo "  Pubkey: $PUBKEY"
echo ""

# Verify keypair consistency using Python
echo "Verifying keypair consistency..."
python3 -c "
import sys
try:
    from secp256k1 import PrivateKey
    
    privkey_hex = '$PRIVKEY'
    expected_pubkey = '$PUBKEY'
    
    # Derive public key from private key
    privkey_bytes = bytes.fromhex(privkey_hex)
    privkey = PrivateKey(privkey_bytes, raw=True)
    pubkey = privkey.pubkey.serialize().hex()
    
    if pubkey == expected_pubkey:
        print('✓ Keypair is consistent!')
        print(f'  Private key correctly derives to public key')
        sys.exit(0)
    else:
        print('❌ Keypair mismatch!')
        print(f'  Derived pubkey:  {pubkey}')
        print(f'  Expected pubkey: {expected_pubkey}')
        sys.exit(1)
except ImportError:
    print('⚠ Cannot verify (secp256k1 Python module not installed)')
    print('  Install with: pip install secp256k1')
    sys.exit(0)
except Exception as e:
    print(f'❌ Error verifying keypair: {e}')
    sys.exit(1)
"

echo ""
echo "=== Summary ==="
echo "Private key file: $COORDINATOR_PRIVKEY_FILE"
echo "Coordinator API:  $COORDINATOR_URL/pubkey"
echo ""
echo "If decryption is failing, the issue may be:"
echo "1. Results were encrypted before coordinator generated its keypair"
echo "2. Database was deleted but old results still exist"
echo "3. Miner cached an old public key"
echo ""
echo "Solution: Delete old database and restart all services"
echo "  rm /tmp/genetics-l2-nih2*.db*"
echo "  ./script.sh"
