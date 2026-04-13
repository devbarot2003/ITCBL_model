import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from security.session_tracker import SessionTracker

tracker = SessionTracker()

session_id = "user_123"

tracker.store_request(session_id, "username=admin'--")
tracker.store_request(session_id, "GET /profile?id=10")

if tracker.detect_second_order(session_id):
    print("⚠ Possible second-order SQL injection detected")
else:
    print("No attack detected")