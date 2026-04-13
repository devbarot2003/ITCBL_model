from collections import OrderedDict
import re


class SessionTracker:

    def __init__(self, max_sessions=1000):

        self.sessions = OrderedDict()
        self.max_sessions = max_sessions

        # simple suspicious SQL patterns
        self.sqli_patterns = [
            r"or\s+1=1",
            r"union\s+select",
            r"drop\s+table",
            r"--",
            r"'"
        ]


    def store_request(self, session_id, request_text):

        if session_id not in self.sessions:
            self.sessions[session_id] = []

        self.sessions[session_id].append(request_text)

        # maintain limited memory
        if len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)


    def detect_second_order(self, session_id):

        if session_id not in self.sessions:
            return False

        history = " ".join(self.sessions[session_id]).lower()

        for pattern in self.sqli_patterns:
            if re.search(pattern, history):
                return True

        return False