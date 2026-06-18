# Make the repository root importable so tests can `import deepfaune_batch`
# without installing the package.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
