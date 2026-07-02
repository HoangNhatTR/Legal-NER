"""Cho phép `import api.judgment_rag` khi chạy pytest từ bất kỳ đâu.

legal_ner là package độc lập (uvicorn chạy từ trong legal_ner/), nên thêm thư mục
legal_ner/ vào sys.path để 'api.*' phân giải được.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # legal_ner/
