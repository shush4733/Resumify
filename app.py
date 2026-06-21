"""
app.py
Main Flask application — handles routes for upload and results pages.
"""

import os
import tempfile
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from parser import extract_text
from analyzer import analyze_resume, match_with_jd, looks_like_resume

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-this")

ALLOWED_EXTENSIONS = {"pdf", "docx"}
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB limit


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # --- Validate file upload ---
    if "resume" not in request.files:
        flash("No file selected.")
        return redirect(url_for("index"))

    file = request.files["resume"]

    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Please upload a PDF or DOCX file.")
        return redirect(url_for("index"))

    # --- Save file temporarily (works on any hosting platform, auto-cleaned) ---
    filename = secure_filename(file.filename)
    suffix = os.path.splitext(filename)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        filepath = tmp.name

    try:
        # --- Extract text ---
        text, error = extract_text(filepath)
    finally:
        # --- Always clean up the temp file ---
        if os.path.exists(filepath):
            os.remove(filepath)

    if error:
        flash(error)
        return redirect(url_for("index"))

    # --- Validate that this is actually a resume, not some other document ---
    is_resume, reason = looks_like_resume(text)
    if not is_resume:
        flash(reason)
        return redirect(url_for("index"))

    # --- Run analysis ---
    result = analyze_resume(text)

    # --- Optional JD matching ---
    jd_text = request.form.get("job_description", "").strip()
    jd_result = None
    if jd_text:
        jd_result = match_with_jd(text, jd_text)

    return render_template(
        "results.html",
        result=result,
        jd_result=jd_result,
        filename=filename
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
