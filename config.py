import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # Google AI API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = "gemini-2.0-flash"
    
    # Streamlit Configuration
    STREAMLIT_CONFIG = {
        "page_title": "Fi Money AI Agent",
        "page_icon": "💰",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
    
    # Financial Data Configuration
    MOCK_DATA_CONFIG = {
        "user_name": "Rahul Sharma",
        "base_income": 85000,
        "base_expenses": 55000,
        "credit_score_range": (650, 850),
        "market_benchmark_return": 11.5
    }
    
    # Agent Configuration
    AGENT_CONFIG = {
        "temperature": 0.3,
        "max_tokens": 2048,
        "system_message": """
        You are a personal financial advisor AI powered by Fi Money's MCP server and Google's Gemini.
        You have access to comprehensive financial data and tools to provide personalized insights.
        
        Your capabilities include:
        1. Net worth analysis and trends
        2. Goal-based financial planning
        3. SIP performance monitoring
        4. Financial health scoring
        5. Spending pattern analysis
        6. Financial report generation
        
        Always provide actionable, personalized advice based on the user's actual financial data.
        Be conversational, helpful, and explain complex financial concepts in simple terms.
        Use emojis and formatting to make responses engaging.
        """
    }
    
    # Tool Descriptions
    TOOL_DESCRIPTIONS = {
        "net_worth": "Comprehensive net worth analysis with trends and insights",
        "goal_planning": "Analyze financial goal feasibility and create actionable plans",
        "sip_performance": "Monitor SIP performance against market benchmarks",
        "health_score": "Calculate dynamic financial health score out of 100",
        "spending_analysis": "Detailed spending pattern analysis and recommendations",
        "report_generation": "Generate comprehensive financial reports for export"
    }

# Environment setup script
def setup_environment():
    """Setup the environment for the Fi Money AI Agent"""
    
    # Check if required API keys are available
    if not Config.GOOGLE_API_KEY:
        print("⚠️  Warning: GOOGLE_API_KEY not found in environment variables")
        print("Please set your Google AI API key in a .env file:")
        print("GOOGLE_API_KEY=your_api_key_here")
        return False
    
    # Create necessary directories
    directories = ["data", "exports", "logs"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
    
    print("🚀 Environment setup complete!")
    return True

if __name__ == "__main__":
    setup_environment()