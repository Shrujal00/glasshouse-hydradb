"""Test-wide defaults.

The answer cache exists so that the same question over the same evidence
returns the answer it returned the first time -- the hosted model does not
guarantee that on a long prompt. Under test it has to be off: the streaming
tests drive a stubbed client through the same prompt with different scripted
responses, and a cache would replay the first one for the second.
"""

import os

os.environ.setdefault("GLASSHOUSE_ANSWER_CACHE", "0")
