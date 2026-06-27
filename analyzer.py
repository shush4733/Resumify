"""
analyzer.py
Core logic: section detection, rule-based scoring, and JD matching.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# 1. SECTION DETECTION
# ----------------------------

SECTION_HEADERS = {
    "experience": [
        "experience", "work experience", "employment history",
        "professional experience", "internship", "internships"
    ],
    "education": [
        "education", "academic background", "qualifications", "academics"
    ],
    "skills": [
        "skills", "technical skills", "core competencies", "key skills"
    ],
    "projects": [
        "projects", "academic projects", "personal projects"
    ],
    "certifications": [
        "certifications", "certificates", "licenses"
    ],
}


def detect_sections(text):
    """
    Splits resume text into sections based on common header keywords.
    Returns a dict: {section_name: section_text}
    """
    lines = text.split("\n")
    sections = {key: "" for key in SECTION_HEADERS}
    sections["other"] = ""

    current_section = "other"

    for line in lines:
        clean_line = line.strip().lower()
        matched = False

        # Check if this line is a section header
        for section, keywords in SECTION_HEADERS.items():
            # Header lines are usually short and match a keyword closely
            if len(clean_line) < 40:
                for kw in keywords:
                    if clean_line == kw or clean_line.startswith(kw):
                        current_section = section
                        matched = True
                        break
            if matched:
                break

        if not matched:
            sections[current_section] += line + "\n"

    return sections


# ----------------------------
# 2. CONTACT INFO CHECK
# ----------------------------

def find_contact_info(text):
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    phone_pattern = r"(\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"
    linkedin_pattern = r"linkedin\.com/in/[A-Za-z0-9\-_/]+"

    email = re.search(email_pattern, text)
    phone = re.search(phone_pattern, text)
    linkedin = re.search(linkedin_pattern, text, re.IGNORECASE)

    return {
        "email_found": bool(email),
        "email": email.group() if email else None,
        "phone_found": bool(phone),
        "phone": phone.group() if phone else None,
        "linkedin_found": bool(linkedin),
        "linkedin": linkedin.group() if linkedin else None,
    }


# ----------------------------
# 2b. RESUME VALIDATION
# ----------------------------

# Words/phrases that strongly suggest a document IS a resume
RESUME_SIGNAL_KEYWORDS = [
    "experience", "education", "skills", "objective", "summary",
    "certification", "certifications", "internship", "projects",
    "qualification", "qualifications", "achievements", "references",
    "employment", "career", "resume", "curriculum vitae", "cv"
]


def looks_like_resume(text):
    """
    Heuristic check to catch obviously non-resume documents
    (essays, random PDFs, etc.) before running full analysis.

    A real resume almost always has:
      - contact info (email or phone), AND
      - at least one recognizable resume-related keyword/section

    If a document has neither, it's very unlikely to be a resume.
    Returns: (is_resume: bool, reason: str or None)
    """
    if not text or len(text.strip()) < 30:
        return False, "The document appears to be empty or too short to be a resume."

    contact = find_contact_info(text)
    has_contact = contact["email_found"] or contact["phone_found"]

    text_lower = text.lower()
    # Use word-boundary matching so "educational" doesn't falsely match "education"
    keyword_hits = sum(
        1 for kw in RESUME_SIGNAL_KEYWORDS
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)
    )

    word_count = len(text.split())

    # Reject if it has no contact info AND fewer than 2 distinct resume keywords —
    # a single incidental keyword mention (e.g. "my experience was great") isn't enough
    if not has_contact and keyword_hits < 2:
        return False, (
            "This file doesn't look like a resume — no contact details or resume-related "
            "sections (Experience, Education, Skills, etc.) were found. Please upload a resume file."
        )

    # Reject very long documents with weak resume signal — likely an essay/report
    if word_count > 1000 and keyword_hits <= 2 and not has_contact:
        return False, (
            "This file looks like a general document rather than a resume. "
            "Please upload a resume file."
        )

    return True, None


# ----------------------------
# 3. ACTION VERBS & QUANTIFICATION
# ----------------------------

ACTION_VERBS = [
    "led", "built", "created", "designed", "developed", "implemented",
    "managed", "improved", "increased", "reduced", "achieved", "launched",
    "optimized", "automated", "streamlined", "delivered", "spearheaded",
    "coordinated", "analyzed", "engineered", "architected", "established",
    "executed", "generated", "organized", "resolved", "trained"
]

WEAK_PHRASES = [
    "responsible for", "duties included", "worked on", "helped with",
    "involved in", "assisted with"
]


def count_action_verbs(text):
    text_lower = text.lower()
    found = [verb for verb in ACTION_VERBS if re.search(r"\b" + verb + r"\b", text_lower)]
    return found


def count_weak_phrases(text):
    text_lower = text.lower()
    found = [phrase for phrase in WEAK_PHRASES if phrase in text_lower]
    return found


def count_quantified_bullets(text):
    """
    Looks for numbers/percentages near bullet-like lines —
    a sign of measurable achievements ('increased sales by 20%').
    """
    lines = text.split("\n")
    quantified = 0
    for line in lines:
        if re.search(r"\d+%|\$\d+|\d+\s*(users|customers|hours|days|projects|team|members|x\b)", line.lower()):
            quantified += 1
    return quantified


# ----------------------------
# 4. MAIN SCORING ENGINE
# ----------------------------

def analyze_resume(text):
    word_count = len(text.split())
    sections = detect_sections(text)
    contact = find_contact_info(text)
    action_verbs_found = count_action_verbs(text)
    weak_phrases_found = count_weak_phrases(text)
    quantified_count = count_quantified_bullets(text)

    score = 0
    max_score = 100
    feedback = []
    strengths = []

    # --- Check 1: Word count (15 pts) ---
    if 300 <= word_count <= 800:
        score += 15
        strengths.append(f"Good resume length ({word_count} words).")
    elif word_count < 300:
        score += 7
        feedback.append(f"Your resume seems short ({word_count} words). Consider adding more detail about your experience and projects.")
    else:
        score += 8
        feedback.append(f"Your resume seems long ({word_count} words). Try to make it more concise — aim for 1 page if you have under 3 years of experience.")

    # --- Check 2: Contact info (15 pts) ---
    contact_score = 0
    if contact["email_found"]:
        contact_score += 7
        strengths.append("Email address found.")
    else:
        feedback.append("No email address detected. Make sure your email is clearly visible at the top.")

    if contact["phone_found"]:
        contact_score += 5
        strengths.append("Phone number found.")
    else:
        feedback.append("No phone number detected.")

    if contact["linkedin_found"]:
        contact_score += 3
        strengths.append("LinkedIn profile found.")
    else:
        feedback.append("Consider adding your LinkedIn profile URL.")

    score += contact_score

    # --- Check 3: Key sections present (25 pts, ~5 each) ---
    section_score = 0
    for sec in ["experience", "education", "skills", "projects"]:
        if sections[sec].strip():
            section_score += 6
            strengths.append(f"'{sec.capitalize()}' section detected.")
        else:
            if sec == "experience":
                feedback.append(
                    "Experience section not found. If you don't have formal work experience yet, "
                    "that's normal — make sure your Projects and Internships are clearly labeled instead, "
                    "since they can carry similar weight for entry-level roles."
                )
            else:
                feedback.append(f"'{sec.capitalize()}' section not found. Make sure it has a clear heading so it can be detected.")
    score += min(section_score, 25)

    # --- Check 4: Action verbs (20 pts) ---
    verb_count = len(action_verbs_found)
    if verb_count >= 8:
        score += 20
        strengths.append(f"Strong use of action verbs ({verb_count} found: {', '.join(action_verbs_found[:6])}...).")
    elif verb_count >= 4:
        score += 12
        feedback.append(f"You're using some action verbs ({verb_count} found), but adding more (e.g. 'led', 'built', 'optimized') would strengthen your bullet points.")
    else:
        score += 4
        feedback.append("Few strong action verbs detected. Start bullet points with words like 'developed', 'led', 'built' instead of describing tasks passively.")

    # --- Check 5: Weak phrases penalty info (10 pts) ---
    if len(weak_phrases_found) == 0:
        score += 10
        strengths.append("No weak/passive phrases detected.")
    else:
        score += max(10 - len(weak_phrases_found) * 3, 0)
        feedback.append(f"Avoid passive phrases like {', '.join(set(weak_phrases_found))}. Replace them with direct action verbs.")

    # --- Check 6: Quantified achievements (15 pts) ---
    if quantified_count >= 3:
        score += 15
        strengths.append(f"Good use of numbers/metrics ({quantified_count} quantified points found).")
    elif quantified_count >= 1:
        score += 8
        feedback.append("You have some quantified achievements — try to add numbers/percentages to more bullet points (e.g. 'reduced load time by 30%').")
    else:
        feedback.append("No quantified achievements found. Wherever possible, add numbers — team size, % improvement, time saved, etc.")

    score = min(score, max_score)

    return {
        "score": score,
        "word_count": word_count,
        "sections_detected": {k: bool(v.strip()) for k, v in sections.items() if k != "other"},
        "contact": contact,
        "action_verbs_found": action_verbs_found,
        "weak_phrases_found": weak_phrases_found,
        "quantified_count": quantified_count,
        "strengths": strengths,
        "feedback": feedback,
    }


# ----------------------------
# 5. JOB DESCRIPTION MATCHING (TF-IDF + Cosine Similarity)
# ----------------------------

def match_with_jd(resume_text, jd_text):
    """
    Computes similarity between resume and job description using TF-IDF.
    Also extracts JD keywords missing from the resume.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    match_percent = round(similarity * 100, 2)

    # Find important JD keywords missing from resume
    jd_words = set(re.findall(r"\b[a-zA-Z][a-zA-Z+#.]{2,}\b", jd_text.lower()))
    resume_words = set(re.findall(r"\b[a-zA-Z][a-zA-Z+#.]{2,}\b", resume_text.lower()))

    # crude stopword filter
    common_stopwords = {
        "the", "and", "for", "with", "you", "are", "this", "that", "will",
        "your", "have", "from", "our", "all", "any", "can", "able", "who",
        "job", "role", "work", "team", "experience", "skills", "years"
    }
    jd_keywords = {w for w in jd_words if w not in common_stopwords}
    missing_keywords = sorted(jd_keywords - resume_words)

    return {
        "match_percent": match_percent,
        "missing_keywords": missing_keywords[:15],  # cap for readability
    }


# ----------------------------
# 6. GEMINI AI FEEDBACK
# ----------------------------

def get_ai_feedback(resume_text, score):
    """
    Sends resume text to Groq (Llama 3.1) and gets personalized AI feedback.
    Returns a string with 3 specific suggestions, or None if the call fails.
    Retries once on temporary server errors before giving up silently.
    """
    import time

    def _call_groq():
        import os
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None

        client = Groq(api_key=api_key)

        prompt = f"""You are an expert resume coach reviewing a student's resume.
The resume scored {score}/100 on an automated analysis.

Resume content:
{resume_text[:3000]}

Give exactly 3 specific, actionable suggestions to improve this resume.
Rules:
- Start directly with "1." — no intro sentence, no preamble
- Each suggestion: Title: explanation (1-2 sentences max)
- Format strictly as:
1. Title: explanation
2. Title: explanation
3. Title: explanation
- Be direct. Focus on what's weak or missing."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

    # First attempt
    try:
        return _call_groq()
    except Exception:
        pass

    # Retry once after 3 seconds
    try:
        time.sleep(3)
        return _call_groq()
    except Exception:
        return None
