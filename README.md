**⚡ BrainSpark — AI Quiz Generator**
BrainSpark is a professional-grade Streamlit application that leverages Groq LLaMA‑3.3‑70B to generate unique, adaptive multiple‑choice quizzes. Designed for both math and non‑math topics, it provides structured textbook‑style solutions for calculation problems and integrates curated learning resources to support deeper understanding.

**📌 Features**
20 Unique MCQs per quiz — no repetition, always fresh content.
Adaptive difficulty levels: Easy, Medium, Hard with tailored rules.
Math mode: Step‑by‑step solutions with formulas, given values, and calculations.
Resource integration: Fetches 3 web links + 1 YouTube tutorial per question via SerpAPI & DuckDuckGo.
Performance tracking: Score percentage, grade assignment, and detailed review of answers.
Interactive UI: Streamlit interface with clear stages — Home → Quiz → Results.
Robust fallback mechanisms for resource fetching and error handling.

**🛠️ Tech Stack**
Python 3.9+
Streamlit — interactive web app framework
Groq LLaMA‑3.3‑70B — quiz generation engine
NLTK — topic detection & text processing
Scikit‑learn (TF‑IDF) — query ranking
Requests + BeautifulSoup — resource fetching
SerpAPI & DuckDuckGo — search integration
SymPy — math formatting support

**🚀 Getting Started**
1. Clone the repository
   
bash
git clone https://github.com/your-username/BrainSpark.git
cd BrainSpark

2. Install dependencies

bash
pip install -r requirements.txt

3. Configure environment variables
Set your API keys:

bash
export GROQ_API_KEY="your_groq_api_key"
export SERPAPI_API_KEY="your_serpapi_api_key"

4. Launch the app

bash
streamlit run BrainSpark.py

**📊 Usage Flow**
Home Stage
Enter quiz topic
Select difficulty (Easy / Medium / Hard)
Generate quiz

Quiz Stage
Answer 20 MCQs
Submit quiz

Results Stage
View score, grade, and performance feedback
Review detailed explanations
Access curated learning resources
Retake quiz or start a new one

**🎯 Example**
Topic: Algebra
Difficulty: Medium
Output: 20 unique MCQs with step‑by‑step solutions and resource links.

**📌 Roadmap**
User progress tracking & analytics
Export results to PDF/CSV
Collaborative quiz sessions
Integration with additional search APIs
