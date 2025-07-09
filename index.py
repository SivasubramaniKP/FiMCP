"""
Personal Finance AI Agent using Google Gemini and LangChain
Created for Fi Money MCP Hackathon

Required installations:
pip install langchain langchain-google-genai pandas requests python-dotenv

Set your Google API key:
export GOOGLE_API_KEY="your-gemini-api-key"
or create a .env file with GOOGLE_API_KEY=your-api-key
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import requests
from io import StringIO
import math
import matplotlib.pyplot as plt
import os.path

# Set up environment
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# LangChain imports
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Mock data structures
@dataclass
class Asset:
    name: str
    type: str
    current_value: float
    purchase_date: str
    purchase_value: float
    
@dataclass
class Liability:
    name: str
    type: str
    current_amount: float
    interest_rate: float
    emi: float
    tenure_months: int
    
@dataclass
class SIPInvestment:
    name: str
    monthly_amount: float
    current_value: float
    start_date: str
    returns: float
    category: str

@dataclass
class Transaction:
    date: str
    amount: float
    category: str
    description: str
    type: str  # 'income' or 'expense'

@dataclass
class LoanDetail:
    name: str
    principal: float
    outstanding: float
    interest_rate: float
    emi: float
    tenure_months: int
    start_date: str
    loan_type: str  # e.g., 'home', 'car', 'personal', 'mortgage'
    property: Optional[str] = None  # For mortgage loans

@dataclass
class CreditCardDetail:
    name: str
    card_number: str
    limit: float
    outstanding: float
    due_date: str
    min_due: float
    last_payment: float
    last_payment_date: str
    annual_fee: float
    rewards_points: int

@dataclass
class FixedDeposit:
    name: str
    principal: float
    current_value: float
    interest_rate: float
    start_date: str
    maturity_date: str
    bank: str

@dataclass
class RecurringDeposit:
    name: str
    monthly_deposit: float
    current_value: float
    interest_rate: float
    start_date: str
    maturity_date: str
    bank: str

# Mock Financial Data Generator
class MockFinancialDataGenerator:
    def __init__(self):
        self.user_profile = {
            "name": "Arjun Sharma",
            "age": 28,
            "monthly_income": 125000,
            "city": "Bangalore",
            "profession": "Software Engineer",
            "risk_tolerance": "moderate"
        }
        
    def generate_assets(self) -> List[Asset]:
        return [
            Asset("Savings Account - SBI", "cash", 350000, "2020-01-15", 100000),
            Asset("Fixed Deposit - HDFC", "fixed_deposit", 500000, "2023-03-10", 500000),
            Asset("Equity Portfolio", "stocks", 850000, "2021-06-01", 600000),
            Asset("Mutual Fund Portfolio", "mutual_funds", 1200000, "2020-08-15", 800000),
            Asset("PPF Account", "retirement", 180000, "2019-04-01", 120000),
            Asset("Gold ETF", "commodities", 75000, "2022-01-20", 65000),
            Asset("Apartment - Koramangala", "real_estate", 4500000, "2023-01-15", 4200000),
        ]
    
    def generate_liabilities(self) -> List[Liability]:
        return [
            Liability("Home Loan - HDFC", "home_loan", 3200000, 8.5, 35000, 240),
            Liability("Car Loan - ICICI", "vehicle_loan", 450000, 9.2, 18000, 36),
            Liability("Credit Card - SBI", "credit_card", 25000, 18.0, 5000, 12),
            Liability("Personal Loan", "personal_loan", 150000, 12.5, 8000, 24),
        ]
    
    def generate_sip_investments(self) -> List[SIPInvestment]:
        return [
            SIPInvestment("SBI Blue Chip Fund", 10000, 280000, "2021-01-15", 12.8, "Large Cap"),
            SIPInvestment("HDFC Mid-Cap Opportunities", 8000, 185000, "2021-06-01", 15.2, "Mid Cap"),
            SIPInvestment("Axis Small Cap Fund", 5000, 95000, "2022-01-01", 8.5, "Small Cap"),
            SIPInvestment("UTI Nifty Index Fund", 7000, 145000, "2020-08-15", 11.8, "Index Fund"),
            SIPInvestment("ICICI Prudential Balanced", 6000, 125000, "2021-03-10", 10.5, "Hybrid"),
        ]
    
    def generate_transactions(self) -> List[Transaction]:
        transactions = []
        base_date = datetime.now() - timedelta(days=90)
        
        # Generate income transactions
        for i in range(3):
            date = base_date + timedelta(days=i*30)
            transactions.append(Transaction(
                date.strftime("%Y-%m-%d"),
                125000,
                "salary",
                "Monthly Salary",
                "income"
            ))
        
        # Generate expense transactions
        expense_categories = [
            ("groceries", 8000, 12000),
            ("utilities", 3000, 5000),
            ("entertainment", 5000, 10000),
            ("dining", 4000, 8000),
            ("transport", 2000, 4000),
            ("shopping", 6000, 15000),
            ("healthcare", 2000, 8000),
        ]
        
        for i in range(90):
            date = base_date + timedelta(days=i)
            if random.random() < 0.3:  # 30% chance of transaction per day
                category, min_amt, max_amt = random.choice(expense_categories)
                amount = random.uniform(min_amt, max_amt)
                transactions.append(Transaction(
                    date.strftime("%Y-%m-%d"),
                    -amount,
                    category,
                    f"{category.title()} expense",
                    "expense"
                ))
        
        return transactions
    
    def get_epf_balance(self) -> float:
        return 485000
    
    def get_credit_score(self) -> int:
        return 758

    def generate_loan_details(self) -> List[LoanDetail]:
        return [
            LoanDetail(
                name="Home Loan - HDFC",
                principal=4000000,
                outstanding=3200000,
                interest_rate=8.5,
                emi=35000,
                tenure_months=240,
                start_date="2023-01-15",
                loan_type="home",
                property="Apartment - Koramangala"
            ),
            LoanDetail(
                name="Car Loan - ICICI",
                principal=600000,
                outstanding=450000,
                interest_rate=9.2,
                emi=18000,
                tenure_months=36,
                start_date="2022-06-01",
                loan_type="car"
            ),
            LoanDetail(
                name="Personal Loan - Axis",
                principal=200000,
                outstanding=150000,
                interest_rate=12.5,
                emi=8000,
                tenure_months=24,
                start_date="2023-05-01",
                loan_type="personal"
            ),
            LoanDetail(
                name="Mortgage - SBI",
                principal=2500000,
                outstanding=2100000,
                interest_rate=7.9,
                emi=22000,
                tenure_months=180,
                start_date="2021-09-01",
                loan_type="mortgage",
                property="Plot - Whitefield"
            ),
        ]

    def generate_credit_card_details(self) -> List[CreditCardDetail]:
        return [
            CreditCardDetail(
                name="SBI Platinum Card",
                card_number="XXXX-XXXX-XXXX-1234",
                limit=200000,
                outstanding=25000,
                due_date=(datetime.now() + timedelta(days=10)).strftime("%Y-%m-%d"),
                min_due=2500,
                last_payment=5000,
                last_payment_date=(datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d"),
                annual_fee=1500,
                rewards_points=3200
            ),
            CreditCardDetail(
                name="HDFC Regalia",
                card_number="XXXX-XXXX-XXXX-5678",
                limit=300000,
                outstanding=18000,
                due_date=(datetime.now() + timedelta(days=5)).strftime("%Y-%m-%d"),
                min_due=1800,
                last_payment=2000,
                last_payment_date=(datetime.now() - timedelta(days=15)).strftime("%Y-%m-%d"),
                annual_fee=2500,
                rewards_points=5400
            ),
        ]

    def generate_fixed_deposits(self) -> List[FixedDeposit]:
        return [
            FixedDeposit(
                name="FD - HDFC",
                principal=300000,
                current_value=330000,
                interest_rate=6.5,
                start_date="2022-04-01",
                maturity_date="2025-04-01",
                bank="HDFC"
            ),
            FixedDeposit(
                name="FD - SBI",
                principal=200000,
                current_value=215000,
                interest_rate=6.2,
                start_date="2021-10-01",
                maturity_date="2024-10-01",
                bank="SBI"
            ),
        ]

    def generate_recurring_deposits(self) -> List[RecurringDeposit]:
        return [
            RecurringDeposit(
                name="RD - ICICI",
                monthly_deposit=5000,
                current_value=65000,
                interest_rate=6.0,
                start_date="2022-01-01",
                maturity_date="2025-01-01",
                bank="ICICI"
            ),
            RecurringDeposit(
                name="RD - Axis",
                monthly_deposit=3000,
                current_value=38000,
                interest_rate=6.1,
                start_date="2021-07-01",
                maturity_date="2024-07-01",
                bank="Axis"
            ),
        ]

# Financial Analysis Tools
class FinancialAnalyzer:
    def __init__(self):
        self.data_generator = MockFinancialDataGenerator()
        self.assets = self.data_generator.generate_assets()
        self.liabilities = self.data_generator.generate_liabilities()
        self.sips = self.data_generator.generate_sip_investments()
        self.transactions = self.data_generator.generate_transactions()
        self.epf_balance = self.data_generator.get_epf_balance()
        self.credit_score = self.data_generator.get_credit_score()
        # New data
        self.loan_details = self.data_generator.generate_loan_details()
        self.credit_card_details = self.data_generator.generate_credit_card_details()
        self.fixed_deposits = self.data_generator.generate_fixed_deposits()
        self.recurring_deposits = self.data_generator.generate_recurring_deposits()
        
    def calculate_net_worth(self) -> Dict[str, Any]:
        # Assets
        total_assets = (
            sum(asset.current_value for asset in self.assets)
            + self.epf_balance
            + sum(fd.current_value for fd in self.fixed_deposits)
            + sum(rd.current_value for rd in self.recurring_deposits)
        )
        # Liabilities
        total_liabilities = (
            sum(liability.current_amount for liability in self.liabilities)
            + sum(loan.outstanding for loan in self.loan_details)
            + sum(card.outstanding for card in self.credit_card_details)
        )
        net_worth = total_assets - total_liabilities
        return {
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "net_worth": net_worth,
            "epf_balance": self.epf_balance,
            "asset_breakdown": {**{asset.name: asset.current_value for asset in self.assets},
                                 **{fd.name: fd.current_value for fd in self.fixed_deposits},
                                 **{rd.name: rd.current_value for rd in self.recurring_deposits}},
            "liability_breakdown": {**{liability.name: liability.current_amount for liability in self.liabilities},
                                     **{loan.name: loan.outstanding for loan in self.loan_details},
                                     **{card.name: card.outstanding for card in self.credit_card_details}},
            "fixed_deposits": [{"name": fd.name, "current_value": fd.current_value} for fd in self.fixed_deposits],
            "recurring_deposits": [{"name": rd.name, "current_value": rd.current_value} for rd in self.recurring_deposits],
            "loans": [{"name": loan.name, "outstanding": loan.outstanding} for loan in self.loan_details],
            "credit_cards": [{"name": card.name, "outstanding": card.outstanding} for card in self.credit_card_details],
        }
    
    def calculate_financial_health_score(self) -> Dict[str, Any]:
        net_worth_data = self.calculate_net_worth()
        monthly_income = self.data_generator.user_profile["monthly_income"]
        # Calculate various metrics
        monthly_liabilities = (
            sum(liability.emi for liability in self.liabilities)
            + sum(loan.emi for loan in self.loan_details)
            + sum(card.min_due for card in self.credit_card_details)
        )
        debt_to_income = monthly_liabilities / monthly_income
        # Calculate savings rate
        monthly_expenses = (
            sum(t.amount for t in self.transactions if t.type == "expense" and 
                datetime.strptime(t.date, "%Y-%m-%d") >= datetime.now() - timedelta(days=30)) / -1
            + sum(card.last_payment for card in self.credit_card_details)
        )
        savings_rate = (monthly_income - monthly_expenses) / monthly_income
        # Scoring components (out of 100)
        credit_score_component = min(self.credit_score / 850 * 25, 25)
        debt_component = max(0, 25 - (debt_to_income * 100))
        savings_component = min(savings_rate * 100 * 0.25, 25)
        net_worth_component = min(net_worth_data["net_worth"] / 5000000 * 25, 25)
        total_score = credit_score_component + debt_component + savings_component + net_worth_component
        return {
            "total_score": round(total_score, 1),
            "credit_score_component": round(credit_score_component, 1),
            "debt_component": round(debt_component, 1),
            "savings_component": round(savings_component, 1),
            "net_worth_component": round(net_worth_component, 1),
            "recommendations": self._get_health_recommendations(total_score),
            "monthly_liabilities": monthly_liabilities,
            "monthly_expenses": monthly_expenses,
        }
    
    def _get_health_recommendations(self, score: float) -> List[str]:
        recommendations = []
        if score < 60:
            recommendations.append("Consider reducing high-interest debt")
            recommendations.append("Increase your emergency fund")
            recommendations.append("Review and optimize your monthly expenses")
        elif score < 80:
            recommendations.append("Great progress! Consider increasing SIP investments")
            recommendations.append("Diversify your investment portfolio")
        else:
            recommendations.append("Excellent financial health! Consider wealth optimization strategies")
        return recommendations
    
    def analyze_sip_performance(self) -> Dict[str, Any]:
        # Mock market benchmark data
        market_benchmarks = {
            "Large Cap": 11.5,
            "Mid Cap": 13.2,
            "Small Cap": 12.8,
            "Index Fund": 11.2,
            "Hybrid": 9.8
        }
        
        performance_analysis = []
        for sip in self.sips:
            benchmark = market_benchmarks.get(sip.category, 10.0)
            performance = "outperforming" if sip.returns > benchmark else "underperforming"
            
            performance_analysis.append({
                "fund_name": sip.name,
                "returns": sip.returns,
                "benchmark": benchmark,
                "performance": performance,
                "current_value": sip.current_value,
                "monthly_investment": sip.monthly_amount,
                "category": sip.category
            })
        
        return {
            "total_sip_value": sum(sip.current_value for sip in self.sips),
            "monthly_sip_amount": sum(sip.monthly_amount for sip in self.sips),
            "performance_analysis": performance_analysis
        }

# Initialize the analyzer
analyzer = FinancialAnalyzer()

# Define LangChain tools
@tool
def get_net_worth_summary() -> str:
    """Get a comprehensive net worth summary including assets, liabilities, and trends."""
    net_worth_data = analyzer.calculate_net_worth()
    
    summary = f"""
    **Net Worth Summary:**
    - Total Assets: ₹{net_worth_data['total_assets']:,.2f}
    - Total Liabilities: ₹{net_worth_data['total_liabilities']:,.2f}
    - Net Worth: ₹{net_worth_data['net_worth']:,.2f}
    - EPF Balance: ₹{net_worth_data['epf_balance']:,.2f}
    
    **Asset Breakdown:**
    {chr(10).join([f"- {name}: ₹{value:,.2f}" for name, value in net_worth_data['asset_breakdown'].items()])}
    
    **Liability Breakdown:**
    {chr(10).join([f"- {name}: ₹{value:,.2f}" for name, value in net_worth_data['liability_breakdown'].items()])}
    """
    return summary

@tool
def calculate_loan_affordability(loan_amount: float, loan_type: str = "home") -> str:
    """Calculate if user can afford a specific loan amount."""
    monthly_income = analyzer.data_generator.user_profile["monthly_income"]
    current_emis = sum(liability.emi for liability in analyzer.liabilities)
    
    # Standard loan parameters
    interest_rates = {"home": 8.5, "car": 9.2, "personal": 12.5}
    tenures = {"home": 240, "car": 60, "personal": 60}  # months
    
    rate = interest_rates.get(loan_type, 10.0) / 100 / 12
    tenure = tenures.get(loan_type, 60)
    
    # Calculate EMI
    emi = (loan_amount * rate * (1 + rate)**tenure) / ((1 + rate)**tenure - 1)
    
    # Check affordability (total EMI should not exceed 40% of income)
    total_emi = current_emis + emi
    emi_to_income_ratio = total_emi / monthly_income
    
    affordability = "Yes" if emi_to_income_ratio <= 0.4 else "No"
    
    return f"""
    **Loan Affordability Analysis:**
    - Loan Amount: ₹{loan_amount:,.2f}
    - Estimated EMI: ₹{emi:,.2f}
    - Current EMIs: ₹{current_emis:,.2f}
    - Total EMIs: ₹{total_emi:,.2f}
    - EMI to Income Ratio: {emi_to_income_ratio:.1%}
    - Affordability: {affordability}
    
    **Recommendation:** {"You can afford this loan comfortably" if affordability == "Yes" else "This loan may strain your finances. Consider a lower amount or longer tenure."}
    """

@tool
def get_sip_performance_analysis() -> str:
    """Analyze SIP performance against market benchmarks."""
    performance_data = analyzer.analyze_sip_performance()
    
    analysis = f"""
    **SIP Performance Analysis:**
    - Total SIP Portfolio Value: ₹{performance_data['total_sip_value']:,.2f}
    - Monthly SIP Amount: ₹{performance_data['monthly_sip_amount']:,.2f}
    
    **Individual Fund Performance:**
    """
    
    for fund in performance_data['performance_analysis']:
        status_emoji = "📈" if fund['performance'] == "outperforming" else "📉"
        analysis += f"""
    {status_emoji} {fund['fund_name']} ({fund['category']})
    - Returns: {fund['returns']:.1f}% vs Benchmark: {fund['benchmark']:.1f}%
    - Current Value: ₹{fund['current_value']:,.2f}
    - Monthly Investment: ₹{fund['monthly_investment']:,.2f}
    """
    
    return analysis

@tool
def calculate_financial_health_score() -> str:
    """Calculate a comprehensive financial health score out of 100."""
    health_data = analyzer.calculate_financial_health_score()
    
    score_interpretation = ""
    if health_data['total_score'] >= 80:
        score_interpretation = "Excellent! 🌟"
    elif health_data['total_score'] >= 60:
        score_interpretation = "Good 👍"
    else:
        score_interpretation = "Needs Improvement ⚠️"
    
    return f"""
    **Financial Health Score: {health_data['total_score']}/100 - {score_interpretation}**
    
    **Score Breakdown:**
    - Credit Score Component: {health_data['credit_score_component']}/25
    - Debt Management: {health_data['debt_component']}/25
    - Savings Rate: {health_data['savings_component']}/25
    - Net Worth Growth: {health_data['net_worth_component']}/25
    
    **Recommendations:**
    {chr(10).join([f"• {rec}" for rec in health_data['recommendations']])}
    """

@tool
def simulate_retirement_planning(target_age: int = 60) -> str:
    """Simulate retirement planning and calculate required corpus."""
    current_age = analyzer.data_generator.user_profile["age"]
    years_to_retirement = target_age - current_age
    
    current_monthly_expenses = 80000  # Assumed
    inflation_rate = 0.06
    expected_return = 0.12
    
    # Calculate required corpus
    future_monthly_expenses = current_monthly_expenses * (1 + inflation_rate)**years_to_retirement
    required_corpus = future_monthly_expenses * 12 * 25  # 25x annual expenses
    
    # Calculate current retirement savings
    current_retirement_savings = analyzer.epf_balance
    for asset in analyzer.assets:
        if asset.type in ["retirement", "mutual_funds"]:
            current_retirement_savings += asset.current_value
    
    # Calculate future value of current savings
    future_value_current_savings = current_retirement_savings * (1 + expected_return)**years_to_retirement
    
    # Calculate monthly SIP required
    shortfall = required_corpus - future_value_current_savings
    if shortfall > 0:
        monthly_sip_required = shortfall / (((1 + expected_return/12)**(years_to_retirement*12) - 1) / (expected_return/12))
    else:
        monthly_sip_required = 0
    
    return f"""
    **Retirement Planning Analysis (Target Age: {target_age}):**
    - Years to Retirement: {years_to_retirement}
    - Current Monthly Expenses: ₹{current_monthly_expenses:,.2f}
    - Projected Monthly Expenses at Retirement: ₹{future_monthly_expenses:,.2f}
    - Required Retirement Corpus: ₹{required_corpus:,.2f}
    - Current Retirement Savings: ₹{current_retirement_savings:,.2f}
    - Future Value of Current Savings: ₹{future_value_current_savings:,.2f}
    - Additional Monthly SIP Required: ₹{monthly_sip_required:,.2f}
    
    **Status:** {"On track! 🎯" if shortfall <= 0 else "Needs additional investment 📈"}
    """

@tool
def get_monthly_spending_analysis() -> str:
    """Analyze monthly spending patterns and trends."""
    recent_transactions = [t for t in analyzer.transactions if 
                         datetime.strptime(t.date, "%Y-%m-%d") >= datetime.now() - timedelta(days=30)]
    expense_by_category = {}
    for transaction in recent_transactions:
        if transaction.type == "expense":
            category = transaction.category
            if category not in expense_by_category:
                expense_by_category[category] = 0
            expense_by_category[category] += abs(transaction.amount)
    # Add credit card payments as a category
    total_credit_card_payments = sum(card.last_payment for card in analyzer.credit_card_details)
    if total_credit_card_payments > 0:
        expense_by_category["credit_card_payments"] = total_credit_card_payments
    # Add loan EMIs as a category
    total_loan_emis = sum(loan.emi for loan in analyzer.loan_details)
    if total_loan_emis > 0:
        expense_by_category["loan_emis"] = total_loan_emis
    total_expenses = sum(expense_by_category.values())
    analysis = f"""
    **Monthly Spending Analysis:**
    - Total Monthly Expenses: ₹{total_expenses:,.2f}
    - Monthly Income: ₹{analyzer.data_generator.user_profile['monthly_income']:,.2f}
    - Savings Rate: {((analyzer.data_generator.user_profile['monthly_income'] - total_expenses) / analyzer.data_generator.user_profile['monthly_income']) * 100:.1f}%
    
    **Expense Breakdown:**
    """
    for category, amount in sorted(expense_by_category.items(), key=lambda x: x[1], reverse=True):
        percentage = (amount / total_expenses) * 100
        analysis += f"- {category.replace('_', ' ').title()}: ₹{amount:,.2f} ({percentage:.1f}%)\n"
    return analysis

@tool
def generate_financial_report() -> str:
    """Generate a comprehensive financial report for export."""
    net_worth = analyzer.calculate_net_worth()
    health_score = analyzer.calculate_financial_health_score()
    sip_performance = analyzer.analyze_sip_performance()
    # New breakdowns
    loans = net_worth["loans"]
    credit_cards = net_worth["credit_cards"]
    fixed_deposits = net_worth["fixed_deposits"]
    recurring_deposits = net_worth["recurring_deposits"]
    report = f"""
    **COMPREHENSIVE FINANCIAL REPORT**
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    **PERSONAL INFORMATION:**
    Name: {analyzer.data_generator.user_profile['name']}
    Age: {analyzer.data_generator.user_profile['age']}
    Monthly Income: ₹{analyzer.data_generator.user_profile['monthly_income']:,.2f}
    Credit Score: {analyzer.credit_score}
    
    **NET WORTH SUMMARY:**
    Total Assets: ₹{net_worth['total_assets']:,.2f}
    Total Liabilities: ₹{net_worth['total_liabilities']:,.2f}
    Net Worth: ₹{net_worth['net_worth']:,.2f}
    
    **LOANS & MORTGAGES:**
    {chr(10).join([f"- {loan['name']}: Outstanding ₹{loan['outstanding']:,.2f}" for loan in loans])}
    
    **CREDIT CARDS:**
    {chr(10).join([f"- {card['name']}: Outstanding ₹{card['outstanding']:,.2f}" for card in credit_cards])}
    
    **FIXED DEPOSITS:**
    {chr(10).join([f"- {fd['name']}: ₹{fd['current_value']:,.2f}" for fd in fixed_deposits])}
    
    **RECURRING DEPOSITS:**
    {chr(10).join([f"- {rd['name']}: ₹{rd['current_value']:,.2f}" for rd in recurring_deposits])}
    
    **FINANCIAL HEALTH SCORE:** {health_score['total_score']}/100
    
    **INVESTMENT PORTFOLIO:**
    Total SIP Value: ₹{sip_performance['total_sip_value']:,.2f}
    Monthly SIP Amount: ₹{sip_performance['monthly_sip_amount']:,.2f}
    
    **KEY RECOMMENDATIONS:**
    {chr(10).join([f"• {rec}" for rec in health_score['recommendations']])}
    
    This report can be exported to PDF or CSV for tax filing or CA consultation.
    """
    return report

@tool
def get_credit_card_summary() -> str:
    """Get a detailed summary of all credit cards, including limits, outstanding, due dates, and rewards."""
    cards = analyzer.credit_card_details
    if not cards:
        return "No credit card data available."
    summary = "**Credit Card Summary:**\n"
    for card in cards:
        summary += f"\n- {card.name} ({card.card_number})\n  - Limit: ₹{card.limit:,.2f}\n  - Outstanding: ₹{card.outstanding:,.2f}\n  - Due Date: {card.due_date}\n  - Minimum Due: ₹{card.min_due:,.2f}\n  - Last Payment: ₹{card.last_payment:,.2f} on {card.last_payment_date}\n  - Annual Fee: ₹{card.annual_fee:,.2f}\n  - Rewards Points: {card.rewards_points}\n"
    return summary

@tool
def get_loan_mortgage_summary() -> str:
    """Get a detailed summary of all loans and mortgages, including property, outstanding, EMI, and tenure."""
    loans = analyzer.loan_details
    if not loans:
        return "No loan or mortgage data available."
    summary = "**Loan & Mortgage Summary:**\n"
    for loan in loans:
        summary += f"\n- {loan.name} ({loan.loan_type.title()})\n  - Principal: ₹{loan.principal:,.2f}\n  - Outstanding: ₹{loan.outstanding:,.2f}\n  - Interest Rate: {loan.interest_rate:.2f}%\n  - EMI: ₹{loan.emi:,.2f}\n  - Tenure: {loan.tenure_months} months\n  - Start Date: {loan.start_date}\n"
        if loan.property:
            summary += f"  - Property: {loan.property}\n"
    return summary

@tool
def get_deposit_summary() -> str:
    """Get a detailed summary of all fixed and recurring deposits."""
    fds = analyzer.fixed_deposits
    rds = analyzer.recurring_deposits
    summary = "**Deposit Summary:**\n\n**Fixed Deposits:**\n"
    if not fds:
        summary += "- None\n"
    else:
        for fd in fds:
            summary += f"- {fd.name} ({fd.bank})\n  - Principal: ₹{fd.principal:,.2f}\n  - Current Value: ₹{fd.current_value:,.2f}\n  - Interest Rate: {fd.interest_rate:.2f}%\n  - Start Date: {fd.start_date}\n  - Maturity Date: {fd.maturity_date}\n"
    summary += "\n**Recurring Deposits:**\n"
    if not rds:
        summary += "- None\n"
    else:
        for rd in rds:
            summary += f"- {rd.name} ({rd.bank})\n  - Monthly Deposit: ₹{rd.monthly_deposit:,.2f}\n  - Current Value: ₹{rd.current_value:,.2f}\n  - Interest Rate: {rd.interest_rate:.2f}%\n  - Start Date: {rd.start_date}\n  - Maturity Date: {rd.maturity_date}\n"
    return summary

@tool
def plot_monthly_credit_card_expenses() -> str:
    """Plot monthly credit card expenses (total and per card) and save the graph to the file system."""
    ensure_plots_dir()
    # Aggregate by card and by month
    cards = analyzer.credit_card_details
    transactions = analyzer.transactions
    # Only consider last 6 months
    now = datetime.now()
    months = [(now - timedelta(days=30*i)).strftime('%Y-%m') for i in reversed(range(6))]
    card_names = [card.name for card in cards]
    # Initialize data
    data = {card: {month: 0.0 for month in months} for card in card_names}
    # Aggregate expenses by card and month
    for card in cards:
        for t in transactions:
            if t.type == 'expense' and card.name.lower().split()[0] in t.description.lower():
                t_month = t.date[:7]
                if t_month in months:
                    data[card.name][t_month] += abs(t.amount)
    # Plot
    plt.figure(figsize=(10,6))
    for card in card_names:
        plt.plot(months, [data[card][m] for m in months], marker='o', label=card)
    plt.title('Monthly Credit Card Expenses (Last 6 Months)')
    plt.xlabel('Month')
    plt.ylabel('Amount (₹)')
    plt.legend()
    plt.tight_layout()
    filename = f"plots/credit_card_expenses_{now.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    plt.close()
    return f"Credit card expenses plot saved to: {filename}"

@tool
def plot_net_worth_over_time() -> str:
    """Plot net worth over the last 12 months (mocked) and save the graph to the file system."""
    ensure_plots_dir()
    now = datetime.now()
    months = [(now - timedelta(days=30*i)).strftime('%Y-%m') for i in reversed(range(12))]
    # Mock net worth history: add some random walk to current net worth
    current = analyzer.calculate_net_worth()['net_worth']
    net_worths = [current - (11-i)*random.uniform(20000, 80000) for i in range(12)]
    plt.figure(figsize=(10,6))
    plt.plot(months, net_worths, marker='o', color='tab:blue')
    plt.title('Net Worth Over Time (Last 12 Months)')
    plt.xlabel('Month')
    plt.ylabel('Net Worth (₹)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = f"plots/net_worth_over_time_{now.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    plt.close()
    return f"Net worth over time plot saved to: {filename}"

@tool
def plot_monthly_spending_by_category() -> str:
    """Plot monthly spending by category for the last 6 months and save the graph to the file system."""
    ensure_plots_dir()
    now = datetime.now()
    months = [(now - timedelta(days=30*i)).strftime('%Y-%m') for i in reversed(range(6))]
    transactions = analyzer.transactions
    categories = set(t.category for t in transactions if t.type == 'expense')
    data = {cat: [0.0 for _ in months] for cat in categories}
    for idx, month in enumerate(months):
        for t in transactions:
            if t.type == 'expense' and t.date[:7] == month:
                data[t.category][idx] += abs(t.amount)
    plt.figure(figsize=(12,7))
    for cat in categories:
        plt.plot(months, data[cat], marker='o', label=cat)
    plt.title('Monthly Spending by Category (Last 6 Months)')
    plt.xlabel('Month')
    plt.ylabel('Amount (₹)')
    plt.legend()
    plt.tight_layout()
    filename = f"plots/monthly_spending_by_category_{now.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    plt.close()
    return f"Monthly spending by category plot saved to: {filename}"

@tool
def plot_loan_outstanding_over_time() -> str:
    """Plot total loan outstanding over the last 12 months (mocked) and save the graph to the file system."""
    ensure_plots_dir()
    now = datetime.now()
    months = [(now - timedelta(days=30*i)).strftime('%Y-%m') for i in reversed(range(12))]
    # Mock: assume each month, some EMI is paid off
    loans = analyzer.loan_details
    total_outstanding = sum(loan.outstanding for loan in loans)
    outstanding_history = [max(total_outstanding - (11-i)*sum(loan.emi for loan in loans), 0) for i in range(12)]
    plt.figure(figsize=(10,6))
    plt.plot(months, outstanding_history, marker='o', color='tab:red')
    plt.title('Loan Outstanding Over Time (Last 12 Months)')
    plt.xlabel('Month')
    plt.ylabel('Outstanding Amount (₹)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    filename = f"plots/loan_outstanding_over_time_{now.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    plt.close()
    return f"Loan outstanding over time plot saved to: {filename}"

@tool
def plot_sip_investment_growth() -> str:
    """Plot SIP investment growth (mocked) for each fund and save the graph to the file system. Always saves a plot, even if no SIPs."""
    ensure_plots_dir()
    now = datetime.now()
    months = [(now - timedelta(days=30*i)).strftime('%Y-%m') for i in reversed(range(12))]
    sips = analyzer.sips
    print(f"[DEBUG] Number of SIPs: {len(sips)}")
    plt.figure(figsize=(12,7))
    if not sips:
        plt.text(0.5, 0.5, 'No SIP data available', fontsize=18, ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('SIP Investment Growth (Last 12 Months)')
        plt.axis('off')
    else:
        for sip in sips:
            # Mock: assume linear growth from 0 to current_value
            values = [sip.current_value * (i/11) for i in range(12)]
            plt.plot(months, values, marker='o', label=sip.name)
        plt.title('SIP Investment Growth (Last 12 Months)')
        plt.xlabel('Month')
        plt.ylabel('Value (₹)')
        plt.legend()
        plt.tight_layout()
    filename = f"plots/sip_investment_growth_{now.strftime('%Y%m%d_%H%M%S')}.png"
    try:
        plt.savefig(filename)
        plt.close()
        print(f"[DEBUG] SIP plot saved to: {filename}")
        return f"SIP investment growth plot saved to: {filename}"
    except Exception as e:
        print(f"[ERROR] Failed to save SIP plot: {e}")
        raise

@tool
def plot_deposit_growth() -> str:
    """Plot Fixed and Recurring Deposit growth (mocked) over the last 12 months and save the graph to the file system."""
    ensure_plots_dir()
    now = datetime.now()
    months = [(now - timedelta(days=30*i)).strftime('%Y-%m') for i in reversed(range(12))]
    fds = analyzer.fixed_deposits
    rds = analyzer.recurring_deposits
    plt.figure(figsize=(12,7))
    for fd in fds:
        # Mock: linear growth from principal to current_value
        values = [fd.principal + (fd.current_value - fd.principal) * (i/11) for i in range(12)]
        plt.plot(months, values, marker='o', label=f"FD: {fd.name}")
    for rd in rds:
        # Mock: linear growth from 0 to current_value
        values = [rd.current_value * (i/11) for i in range(12)]
        plt.plot(months, values, marker='o', label=f"RD: {rd.name}")
    plt.title('Deposit Growth (FD & RD, Last 12 Months)')
    plt.xlabel('Month')
    plt.ylabel('Value (₹)')
    plt.legend()
    plt.tight_layout()
    filename = f"plots/deposit_growth_{now.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(filename)
    plt.close()
    return f"Deposit growth plot saved to: {filename}"

# Initialize Gemini model
def initialize_gemini_model():
    """Initialize Gemini model with proper configuration."""
    # Make sure to set your Google API key
    os.environ["GOOGLE_API_KEY"] = "AIzaSyCthTT_cy1WMTBe_PBiYtxDjDgA8glRPt4"
    
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            temperature=0.1,
            max_tokens=4096,
            timeout=30,
            max_retries=2,
        )
        return model
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        print("Please make sure you have set your GOOGLE_API_KEY environment variable")
        return None

# Create the agent
def create_financial_agent():
    """Create the financial planning agent with all tools."""
    model = initialize_gemini_model()
    if not model:
        return None
    
    tools = [
        get_net_worth_summary,
        calculate_loan_affordability,
        get_sip_performance_analysis,
        calculate_financial_health_score,
        simulate_retirement_planning,
        get_monthly_spending_analysis,
        generate_financial_report,
        get_credit_card_summary,
        get_loan_mortgage_summary,
        get_deposit_summary,
        plot_monthly_credit_card_expenses,
        plot_net_worth_over_time,
        plot_monthly_spending_by_category,
        plot_loan_outstanding_over_time,
        plot_sip_investment_growth,
        plot_deposit_growth
    ]
    
    # Create system prompt template
    system_prompt = f"""
    You are a highly intelligent personal finance AI assistant powered by Google's Gemini model. 
    You help users with comprehensive financial planning, investment analysis, and goal-based financial decisions.
    
    Your capabilities include:
    - Analyzing net worth and financial health
    - Evaluating loan affordability
    - Monitoring SIP and investment performance
    - Providing retirement planning guidance
    - Generating detailed financial reports
    
    User Profile:
    - Name: {analyzer.data_generator.user_profile['name']}
    - Age: {analyzer.data_generator.user_profile['age']}
    - Monthly Income: ₹{analyzer.data_generator.user_profile['monthly_income']:,}
    - Profession: {analyzer.data_generator.user_profile['profession']}
    - Location: {analyzer.data_generator.user_profile['city']}
    
    Always provide personalized, talk in Indian financing Terms, actionable advice based on the user's actual financial data.
    Be conversational, empathetic, and encouraging while maintaining professional accuracy.
    Use Indian currency (₹) and financial context throughout your responses.
    
    You have access to the following tools to analyze the user's financial data:
    - get_net_worth_summary: Get comprehensive net worth analysis
    - calculate_loan_affordability: Check if user can afford specific loan amounts
    - get_sip_performance_analysis: Analyze SIP performance vs benchmarks
    - calculate_financial_health_score: Get overall financial health score
    - simulate_retirement_planning: Plan for retirement goals
    - get_monthly_spending_analysis: Analyze spending patterns
    - generate_financial_report: Create comprehensive financial reports
    
    Use these tools whenever the user asks questions related to their finances.
    """
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    # Create agent
    agent = create_tool_calling_agent(model, tools, prompt)
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,
        early_stopping_method="generate"
    )
    
    return agent_executor

# Chat interface
class FinancialChatbot:
    def __init__(self):
        self.agent = create_financial_agent()
        self.chat_history = []
        
    def chat(self, user_input: str) -> str:
        """Process user input and return AI response."""
        if not self.agent:
            return "Sorry, I'm having trouble connecting to the AI model. Please make sure your Google API key is set."
        
        try:
            # Prepare input for agent
            response = self.agent.invoke({
                "input": user_input,
                "chat_history": self.chat_history
            })
            
            # Add to chat history
            self.chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=response["output"])
            ])
            
            return response["output"]
            
        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}. Please try rephrasing your question."
            return error_msg

# Utility to ensure plots directory exists
def ensure_plots_dir():
    if not os.path.exists('plots'):
        os.makedirs('plots')

# Main chatbot interface
def main():
    """Main chatbot interface."""
    print("🏦 Welcome to your Personal Finance AI Assistant powered by Gemini!")
    print("I can help you with financial planning, investment analysis, and goal tracking.")
    print("Type 'quit' to exit.\n")
    
    # Initialize chatbot
    chatbot = FinancialChatbot()
    
    # Sample questions to get started
    sample_questions = [
        "What's my current net worth?",
        "Can I afford a ₹50L home loan?",
        "How are my SIPs performing?",
        "What's my financial health score?",
        "How much money will I have at retirement?",
        "Analyze my monthly spending",
        "Generate a financial report"
    ]
    
    print("Sample questions you can ask:")
    for i, question in enumerate(sample_questions, 1):
        print(f"{i}. {question}")
    print()
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Thank you for using your Personal Finance AI Assistant! Take care! 👋")
            break
        
        if not user_input:
            continue
        
        print("\nAI Assistant: ", end="")
        response = chatbot.chat(user_input)
        print(response)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()