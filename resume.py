"""
====================================================================
 Generative AI Resume Screening & Skill Matching System
 Internship Project 4 — Industry-Oriented AI/HR-Tech
====================================================================
 Tools: Python | OpenAI API | Prompt Engineering
 Author: [Your Name]
 Date: June 2026
====================================================================
"""
import os
import json
import re
import pandas as pd
from openai import OpenAI

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "your-api-key-here"))
MODEL = "gpt-3.5-turbo" # Change to "gpt-4" for better results
SHORTLIST_THRESHOLD = 70 # Score >= 70 → Shortlisted


# ─────────────────────────────────────────
# SAMPLE DATA
# ─────────────────────────────────────────
JOB_DESCRIPTION = """
Job Title: AI/ML Engineer
Company: TechCorp Solutions

Required Skills:
- Python programming (3+ years)
- Machine Learning frameworks (TensorFlow or PyTorch)
- Natural Language Processing (NLP)
- REST API development
- SQL and data handling
- Experience with cloud platforms (AWS/GCP/Azure)

Nice to Have:
- Experience with LLMs or generative AI
- Knowledge of Docker/Kubernetes
- Familiarity with prompt engineering

Experience: 2–4 years in software/AI development
Education: B.Tech/B.E. in Computer Science or related field
"""

RESUMES = {
    "Arjun Sharma": """
    B.Tech Computer Science — IIT Delhi (2022) | CGPA: 8.7
    Skills: Python, TensorFlow, PyTorch, NLP, BERT, OpenAI API, FastAPI, SQL, AWS, Docker
    Experience: ML Engineer at DataMinds (2 years)
    - Built NLP pipelines, deployed ML models on AWS SageMaker
    - Developed REST APIs, worked with prompt engineering for GPT chatbots
    """,
    "Priya Nair": """
    B.E. Computer Engineering — VIT University (2021) | CGPA: 7.9
    Skills: Python, Flask, HTML, CSS, JavaScript, MySQL, Git
    Experience: Junior Web Developer at StartupXYZ (3 years)
    - Built full-stack web apps, managed MySQL databases, integrated REST APIs
    """,
    "Rahul Mehta": """
    B.Tech Data Science — BITS Pilani (2023) | CGPA: 9.1
    Skills: Python, PyTorch, NLP, Hugging Face, LangChain, OpenAI API,
            Prompt Engineering, GCP, Docker, Kubernetes, SQL, PySpark
    Experience: Data Science Intern at Google India (6 months)
    - Fine-tuned LLMs, built generative AI pipelines with LangChain
    - Deployed models on GCP with Docker and Kubernetes
    """
}


# ─────────────────────────────────────────
# PROMPT FUNCTIONS
# ─────────────────────────────────────────
def extract_resume_info(resume_text: str) -> dict:
    """Extract structured information from a resume using LLM prompt."""
    prompt = f"""
You are an expert HR analyst. Extract structured info from this resume.
Return ONLY valid JSON with these keys:
{{
  "name": "candidate name",
  "education": "degree and institution",
  "years_experience": number,
  "technical_skills": ["skill1", "skill2"],
  "domains": ["domain1", "domain2"],
  "notable_achievements": "1-2 line summary"
}}

Resume:
{resume_text}

Return only the JSON. No explanation. No markdown.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```json|^```|```$', '', raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def score_candidate(resume_text: str, jd_text: str) -> dict:
    """Score a candidate against a job description using a bias-aware prompt."""
    prompt = f"""
You are a fair, unbiased AI recruitment assistant.
Evaluate the candidate's resume against the job description.

BIAS-AWARE: Evaluate ONLY on skills, experience, qualifications.
Do NOT factor in name, gender, age, or location.

Return ONLY valid JSON:
{{
  "overall_score": 0-100,
  "skills_match_score": 0-100,
  "experience_score": 0-100,
  "education_score": 0-100,
  "matched_skills": ["skill1", "skill2"],
  "missing_skills": ["gap1", "gap2"],
  "strengths": "2-3 sentence summary",
  "gaps": "1-2 sentence summary",
  "recommendation": "Shortlist" or "Maybe" or "Reject",
  "reason": "1 sentence reason"
}}

Job Description:
{jd_text}

Resume:
{resume_text}

Return only the JSON. No explanation. No markdown.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```json|^```|```$', '', raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────
def run_screening():
    print("\n" + "="*65)
    print(" GENERATIVE AI RESUME SCREENING SYSTEM")
    print("="*65 + "\n")

    extracted_data = {}
    scores = {}

    # Step 1: Extract info from each resume
    print("🔍 Step 1: Extracting resume information...\n")
    for name, resume_text in RESUMES.items():
        print(f" Processing: {name}...")
        info = extract_resume_info(resume_text)
        extracted_data[name] = info
        print(f" ✅ Skills found: {len(info.get('technical_skills', []))}")

    # Step 2: Score each candidate
    print("\n📊 Step 2: Scoring candidates...\n")
    for name, resume_text in RESUMES.items():
        print(f" Scoring: {name}...")
        result = score_candidate(resume_text, JOB_DESCRIPTION)
        scores[name] = result
        print(f" ✅ Score: {result.get('overall_score', 'N/A')}/100 — {result.get('recommendation', 'N/A')}")

    # Step 3: Shortlisting
    print(f"\n🏅 Step 3: Shortlisting (threshold: {SHORTLIST_THRESHOLD}/100)\n")
    print("-" * 50)
    shortlisted = []
    for name, score in scores.items():
        overall = score.get("overall_score", 0)
        rec = score.get("recommendation", "N/A")
        if overall >= SHORTLIST_THRESHOLD or rec == "Shortlist":
            status = "✅ SHORTLISTED"
            shortlisted.append(name)
        elif overall >= 50 or rec == "Maybe":
            status = "🟡 MAYBE"
        else:
            status = "❌ REJECTED"
        print(f"{status} — {name} ({overall}/100)")
        print(f" {score.get('reason', '')}\n")

    print(f"Total shortlisted: {len(shortlisted)}/{len(RESUMES)}")

    # Step 4: Export to CSV
    rows = []
    for name in RESUMES:
        s = scores.get(name, {})
        e = extracted_data.get(name, {})
        rows.append({
            "Candidate": name,
            "Overall Score": s.get("overall_score", 0),
            "Skills Match": s.get("skills_match_score", 0),
            "Experience Score": s.get("experience_score", 0),
            "Education Score": s.get("education_score", 0),
            "Matched Skills": ", ".join(s.get("matched_skills", [])),
            "Missing Skills": ", ".join(s.get("missing_skills", [])),
            "Recommendation": s.get("recommendation", "N/A"),
            "Reason": s.get("reason", ""),
        })

    df = pd.DataFrame(rows).sort_values("Overall Score", ascending=False)
    df.to_csv("screening_results.csv", index=False)
    print("\n💾 Results saved to: screening_results.csv")
    print("\n" + "="*65)
    print(" SCREENING COMPLETE")
    print("="*65 + "\n")
    return df


if __name__ == "__main__":
    run_screening()

