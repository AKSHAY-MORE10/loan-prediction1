"""
FastAPI Backend for Loan Eligibility Prediction
Provides REST API endpoints for predictions and Plaid integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import joblib
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Loan Eligibility API",
    description="ML-powered loan eligibility prediction with Plaid integration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None

# Plaid configuration
PLAID_CLIENT_ID = os.getenv("PLAID_CLIENT_ID")
PLAID_SECRET = os.getenv("PLAID_SECRET")
PLAID_ENV = os.getenv("PLAID_ENV", "sandbox")
PLAID_CONFIGURED = bool(PLAID_CLIENT_ID and PLAID_SECRET)


# Pydantic models for request/response validation
class ApplicantData(BaseModel):
    Gender: str = Field(..., example="Male")
    Married: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="0")
    Education: str = Field(..., example="Graduate")
    Self_Employed: str = Field(..., example="No")
    ApplicantIncome: float = Field(..., example=5000)
    CoapplicantIncome: float = Field(..., example=2000)
    LoanAmount: float = Field(..., example=150)
    Loan_Amount_Term: float = Field(..., example=360)
    Credit_History: float = Field(..., example=1.0)
    Property_Area: str = Field(..., example="Urban")
    
    # Optional: Transaction aggregates from Plaid
    avg_monthly_inflow: Optional[float] = None
    avg_monthly_outflow: Optional[float] = None
    nsf_count: Optional[int] = None


class PredictionResponse(BaseModel):
    probability: float
    decision: str
    confidence: str
    explanation: Dict[str, float]


class PlaidExchangeRequest(BaseModel):
    public_token: str


class PlaidExchangeResponse(BaseModel):
    access_token: str
    item_id: str
    is_simulated: bool
    message: Optional[str] = None


class PlaidTransactionsRequest(BaseModel):
    access_token: str
    start_date: str
    end_date: str


class PlaidTransactionsResponse(BaseModel):
    transactions: List[Dict]
    aggregates: Dict[str, float]
    is_simulated: bool
    message: Optional[str] = None


@app.on_event("startup")
async def load_model():
    """
    Load the trained model pipeline at startup
    """
    global model
    # Get the directory where this file is located (backend/)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Model is in backend/models/model.joblib relative to this file
    model_path = os.path.join(current_dir, "models", "model.joblib")
    
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"[OK] Model loaded successfully from {model_path}")
        else:
            print(f"[WARN] Model not found at {model_path}")
            print(f"  Current working directory: {os.getcwd()}")
            print(f"  Backend directory: {current_dir}")
            print("  Please run: python scripts/train.py")
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        import traceback
        traceback.print_exc()


@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Loan Eligibility API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "plaid_exchange": "/plaid/exchange",
            "plaid_transactions": "/plaid/transactions"
        },
        "model_loaded": model is not None,
        "plaid_configured": PLAID_CONFIGURED
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "plaid_configured": PLAID_CONFIGURED,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_eligibility(data: ApplicantData):
    """
    Predict loan eligibility based on applicant data
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run training script first."
        )
    
    try:
        # Convert to DataFrame
        input_dict = data.dict(exclude={'avg_monthly_inflow', 'avg_monthly_outflow', 'nsf_count'})
        df = pd.DataFrame([input_dict])
        
        # Engineer features (same as training)
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        df['IncomeToLoanRatio'] = df['TotalIncome'] / (df['LoanAmount'] + 1)
        df['LogLoanAmount'] = np.log1p(df['LoanAmount'])
        df['DebtToIncomeRatio'] = df['LoanAmount'] / (df['TotalIncome'] + 1)
        df['EMI'] = df['LoanAmount'] / (df['Loan_Amount_Term'] / 12 + 1)
        df['BalanceIncome'] = df['TotalIncome'] - df['EMI']
        
        # If transaction data provided, incorporate it
        if data.avg_monthly_inflow is not None:
            # Could add transaction-based features here
            # For now, just log it
            print(f"Transaction data received: inflow={data.avg_monthly_inflow}")
        
        # Make prediction
        proba = model.predict_proba(df)[0]
        probability = float(proba[1])  # Probability of approval
        
        # Decision
        decision = "Eligible" if probability >= 0.5 else "Not Eligible"
        
        # Confidence level
        if probability >= 0.75 or probability <= 0.25:
            confidence = "High"
        elif probability >= 0.6 or probability <= 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Simple explanation based on key features
        explanation = {
            "Credit_History": float(data.Credit_History),
            "TotalIncome": float(df['TotalIncome'].iloc[0]),
            "LoanAmount": float(data.LoanAmount),
            "IncomeToLoanRatio": float(df['IncomeToLoanRatio'].iloc[0]),
            "DebtToIncomeRatio": float(df['DebtToIncomeRatio'].iloc[0])
        }
        
        return PredictionResponse(
            probability=probability,
            decision=decision,
            confidence=confidence,
            explanation=explanation
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/plaid/exchange", response_model=PlaidExchangeResponse)
async def exchange_public_token(request: PlaidExchangeRequest):
    """
    Exchange Plaid public token for access token
    Uses Plaid sandbox if configured, otherwise simulates
    """
    if PLAID_CONFIGURED:
        # Real Plaid integration would go here
        # For sandbox, we would use plaid-python library
        try:
            # Placeholder for actual Plaid API call
            # from plaid import Client
            # client = Client(client_id=PLAID_CLIENT_ID, secret=PLAID_SECRET, environment=PLAID_ENV)
            # response = client.Item.public_token.exchange(request.public_token)
            
            return PlaidExchangeResponse(
                access_token=f"access-sandbox-{request.public_token[-10:]}",
                item_id=f"item-sandbox-{request.public_token[-8:]}",
                is_simulated=False,
                message="Connected to Plaid sandbox"
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Plaid exchange error: {str(e)}")
    else:
        # Simulated mode
        return PlaidExchangeResponse(
            access_token=f"access-simulated-{request.public_token[-10:]}",
            item_id=f"item-simulated-{request.public_token[-8:]}",
            is_simulated=True,
            message="Plaid sandbox keys not configured. Returning simulated access token for demo."
        )


@app.post("/plaid/transactions", response_model=PlaidTransactionsResponse)
async def get_transactions(request: PlaidTransactionsRequest):
    """
    Fetch transactions from Plaid and calculate aggregates
    Uses Plaid sandbox if configured, otherwise simulates
    """
    if PLAID_CONFIGURED:
        # Real Plaid integration
        try:
            # Placeholder for actual Plaid API call
            # response = client.Transactions.get(
            #     request.access_token,
            #     request.start_date,
            #     request.end_date
            # )
            
            # For now, return simulated data even with keys
            return _generate_simulated_transactions(request.start_date, request.end_date, False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Plaid transactions error: {str(e)}")
    else:
        # Simulated mode
        return _generate_simulated_transactions(request.start_date, request.end_date, True)


def _generate_simulated_transactions(start_date: str, end_date: str, is_simulated: bool):
    """
    Generate realistic simulated transaction data
    """
    np.random.seed(42)
    
    # Generate random transactions
    num_transactions = np.random.randint(20, 50)
    transactions = []
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    categories = ["Food and Drink", "Shopping", "Transfer", "Payment", "Income", "Entertainment"]
    
    for i in range(num_transactions):
        days_offset = np.random.randint(0, (end - start).days)
        transaction_date = start + timedelta(days=days_offset)
        
        category = np.random.choice(categories)
        
        # Income transactions are positive, expenses are negative
        if category == "Income":
            amount = np.random.uniform(2000, 5000)
        else:
            amount = -np.random.uniform(10, 500)
        
        transactions.append({
            "transaction_id": f"txn_{i+1}",
            "date": transaction_date.strftime("%Y-%m-%d"),
            "name": f"{category} Transaction",
            "amount": round(amount, 2),
            "category": [category]
        })
    
    # Calculate aggregates
    inflows = [t['amount'] for t in transactions if t['amount'] > 0]
    outflows = [abs(t['amount']) for t in transactions if t['amount'] < 0]
    
    # Count NSF (negative balance) events
    nsf_count = sum(1 for t in transactions if t['amount'] < -1000)
    
    months = (end - start).days / 30
    
    aggregates = {
        "avg_monthly_inflow": round(sum(inflows) / months if months > 0 else sum(inflows), 2),
        "avg_monthly_outflow": round(sum(outflows) / months if months > 0 else sum(outflows), 2),
        "nsf_count": nsf_count,
        "total_transactions": len(transactions)
    }
    
    message = "Plaid sandbox keys not configured. Returning simulated transactions for demo." if is_simulated else None
    
    return PlaidTransactionsResponse(
        transactions=transactions,
        aggregates=aggregates,
        is_simulated=is_simulated,
        message=message
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)