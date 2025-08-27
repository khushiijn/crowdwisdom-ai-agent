# CrowdWisdomTrading AI Agent

## 📌 Overview
This project is an AI-powered pipeline built using [CrewAI](https://github.com/crewAIInc/crewAI).  
It provides **daily short summaries of US financial markets** after market close, and posts them to a Telegram channel.

## ⚙️ Tech Stack
- **Python 3.12**
- **CrewAI** (multi-agent orchestration)
- **LiteLLM** (for LLM model calls)
- **Tavily / Serper API** (web search)
- **Telegram Bot API** (distribution)

## 🚀 Project Flow
1. **Search Agent**  
   - Queries Tavily/Serper for financial news from the last hour.  
   - Optionally uses Groq API for additional summaries.  

2. **Summary Agent**  
   - Creates a concise (<500 words) summary of today’s financial market activity.  
   - Checks for finance-related keywords.  

3. **Formatting Agent**  
   - Finds 2 related charts/images.  
   - Embeds them into the summary in Markdown.  

4. **Translation Agent**  
   - Translates the summary into **English, Arabic, Hindi, Hebrew**.  

5. **Send Agent**  
   - Posts the final multi-language summary + images to a Telegram channel.  

## 🛡️ Guardrails
Each task includes guardrails to ensure:
- Search → must return results  
- Summary → under 500 words & finance-related  
- Format → exactly 2 images included  
- Translate → must contain all 4 required languages  


## 🔑 Setup
1. Clone the repo:
   ```bash
   git clone https://github.com/khushiijn/crowdwisdom-ai-agent.git
   cd crowdwisdom-ai-agent

2. Install dependencies:
   ```bash
    pip install -r requirements.txt
   
3. Configure environment variables:
   ```bash
   cp .env.env

4. Run
   ```bash
   python crowdwisdom_full_pipeline.py --post true


## ✅ Example Output
<img width="1570" height="810" alt="Screenshot 2025-08-27 194652" src="https://github.com/user-attachments/assets/2d78fceb-3acc-4948-a386-cf9a3ff3cb3a" />
<img width="1590" height="834" alt="Screenshot 2025-08-27 194703" src="https://github.com/user-attachments/assets/5585f3a2-cb6f-4179-b171-d87c4cf1198e" />
<img width="1538" height="835" alt="Screenshot 2025-08-27 194715" src="https://github.com/user-attachments/assets/38c4e4cf-be27-4560-a19f-a7add0079472" />
<img width="1576" height="654" alt="Screenshot 2025-08-27 194723" src="https://github.com/user-attachments/assets/345e5fd6-1b87-49d9-b480-8c9e2416f9f0" />
<img width="1195" height="870" alt="Screenshot 2025-08-28 002757" src="https://github.com/user-attachments/assets/4cf7bb08-1a5a-4723-b01f-d4afca99c9db" />


## 🎥 Demo Video
A short demo video showing the pipeline running and posting to Telegram.  
👉 [Watch the demo here]
https://drive.google.com/file/d/1B1qyRRf8R_KBYAqwU6EAn6-sOTLQtNae/view?usp=sharing
