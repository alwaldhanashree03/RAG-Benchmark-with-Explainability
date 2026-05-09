"""Tests for input validators.

Industry best practice: Test all validation logic.
"""

import pytest
from src.utils.validators import QueryRequest, validate_file_upload
from src.utils.exceptions import ValidationError


def test_query_request_valid():
    """Test valid query request."""
    request = QueryRequest(
        query="What is machine learning?",
        model="baseline",
        top_k=3,
        apply_guardrails=True
    )
    assert request.query == "What is machine learning?"
    assert request.model == "baseline"
    assert request.top_k == 3


def test_query_request_empty():
    """Test empty query is rejected."""
    from pydantic import ValidationError as PydanticValidationError
    with pytest.raises(PydanticValidationError):
        QueryRequest(query="", model="baseline")


def test_query_request_sql_injection():
    """Test SQL injection is blocked."""
    with pytest.raises(ValidationError):
        QueryRequest(query="DROP TABLE users;", model="baseline")


def test_query_request_invalid_model():
    """Test invalid model name is rejected."""
    with pytest.raises(ValidationError):
        QueryRequest(query="test", model="invalid_model")


def test_query_request_top_k_bounds():
    """Test top_k is bounded."""
    with pytest.raises(ValueError):
        QueryRequest(query="test", top_k=0)
    
    with pytest.raises(ValueError):
        QueryRequest(query="test", top_k=100)


def test_file_upload_valid():
    """Test valid file upload."""
    validate_file_upload("document.pdf")
    validate_file_upload("data.json")
    validate_file_upload("text.txt")


def test_file_upload_invalid_extension():
    """Test invalid file extension."""
    with pytest.raises(ValidationError):
        validate_file_upload("malicious.exe")


def test_file_upload_path_traversal():
    """Test path traversal is blocked."""
    with pytest.raises(ValidationError):
        validate_file_upload("../../../etc/passwd")
    
    with pytest.raises(ValidationError):
        validate_file_upload("path/to/file.txt")
