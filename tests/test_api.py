"""
Test suite for Loan Eligibility API
Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app import app

client = TestClient(app)


class TestHealthEndpoint:
    """
    Tests for health check endpoint
    """
    
    def test_health_check_returns_200(self):
        """Test that health endpoint returns 200 OK"""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_check_response_format(self):
        """Test health check response structure"""
        response = client.get("/health")
        data = response.json()
        
        assert "status" in data
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "plaid_configured" in data
        assert "timestamp" in data
    
    def test_root_endpoint(self):
        """Test root endpoint returns API info"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data


class TestPredictionEndpoint:
    """
    Tests for loan prediction endpoint
    """
    
    @pytest.fixture
    def valid_applicant_data(self):
        """Sample valid applicant data"""
        return {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 5000,
            "CoapplicantIncome": 2000,
            "LoanAmount": 150,
            "Loan_Amount_Term": 360,
            "Credit_History": 1.0,
            "Property_Area": "Urban"
        }
    
    def test_predict_with_valid_data(self, valid_applicant_data):
        """Test prediction with valid applicant data"""
        response = client.post("/predict", json=valid_applicant_data)
        
        # May return 503 if model not loaded, that's acceptable in test
        if response.status_code == 503:
            pytest.skip("Model not loaded - run training script first")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "probability" in data
        assert "decision" in data
        assert "confidence" in data
        assert "explanation" in data
        
        # Check data types
        assert isinstance(data["probability"], float)
        assert data["decision"] in ["Eligible", "Not Eligible"]
        assert data["confidence"] in ["High", "Medium", "Low"]
        assert 0 <= data["probability"] <= 1
    
    def test_predict_with_missing_fields(self):
        """Test prediction with missing required fields"""
        incomplete_data = {
            "Gender": "Male",
            "ApplicantIncome": 5000
        }
        
        response = client.post("/predict", json=incomplete_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_with_invalid_data_types(self, valid_applicant_data):
        """Test prediction with invalid data types"""
        invalid_data = valid_applicant_data.copy()
        invalid_data["ApplicantIncome"] = "not_a_number"
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422
    
    def test_predict_with_transaction_data(self, valid_applicant_data):
        """Test prediction with optional transaction aggregates"""
        data_with_transactions = valid_applicant_data.copy()
        data_with_transactions.update({
            "avg_monthly_inflow": 6000.0,
            "avg_monthly_outflow": 3000.0,
            "nsf_count": 0
        })
        
        response = client.post("/predict", json=data_with_transactions)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200


class TestPlaidEndpoints:
    """
    Tests for Plaid integration endpoints
    """
    
    def test_plaid_exchange_token(self):
        """Test Plaid public token exchange"""
        request_data = {
            "public_token": "public-sandbox-test-token-123"
        }
        
        response = client.post("/plaid/exchange", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "access_token" in data
        assert "item_id" in data
        assert "is_simulated" in data
    
    def test_plaid_get_transactions(self):
        """Test fetching transactions from Plaid"""
        request_data = {
            "access_token": "access-sandbox-test-token",
            "start_date": "2024-01-01",
            "end_date": "2024-03-31"
        }
        
        response = client.post("/plaid/transactions", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "transactions" in data
        assert "aggregates" in data
        assert "is_simulated" in data
        
        # Check aggregates structure
        aggregates = data["aggregates"]
        assert "avg_monthly_inflow" in aggregates
        assert "avg_monthly_outflow" in aggregates
        assert "nsf_count" in aggregates
        assert "total_transactions" in aggregates
        
        # Check transactions structure
        if data["transactions"]:
            transaction = data["transactions"][0]
            assert "transaction_id" in transaction
            assert "date" in transaction
            assert "amount" in transaction
    
    def test_plaid_transactions_with_invalid_dates(self):
        """Test transactions endpoint with invalid date format"""
        request_data = {
            "access_token": "access-test",
            "start_date": "invalid-date",
            "end_date": "2024-03-31"
        }
        
        response = client.post("/plaid/transactions", json=request_data)
        # Should either return 422 (validation) or 400 (bad request)
        assert response.status_code in [400, 422]


class TestEdgeCases:
    """
    Tests for edge cases and error handling
    """
    
    def test_predict_with_zero_income(self):
        """Test prediction with zero income"""
        data = {
            "Gender": "Male",
            "Married": "No",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 0,
            "CoapplicantIncome": 0,
            "LoanAmount": 100,
            "Loan_Amount_Term": 360,
            "Credit_History": 1.0,
            "Property_Area": "Urban"
        }
        
        response = client.post("/predict", json=data)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        # Should still return a prediction (likely rejection)
        assert response.status_code == 200
        result = response.json()
        assert result["probability"] < 0.5  # Likely rejected
    
    def test_predict_with_very_high_loan_amount(self):
        """Test prediction with extremely high loan amount"""
        data = {
            "Gender": "Male",
            "Married": "Yes",
            "Dependents": "0",
            "Education": "Graduate",
            "Self_Employed": "No",
            "ApplicantIncome": 5000,
            "CoapplicantIncome": 2000,
            "LoanAmount": 10000,  # Very high
            "Loan_Amount_Term": 360,
            "Credit_History": 1.0,
            "Property_Area": "Urban"
        }
        
        response = client.post("/predict", json=data)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert response.status_code == 200


# Run with: pytest tests/test_api.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])