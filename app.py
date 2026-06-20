"""
app.py
Main Flask application — handles routes for upload and results pages.
"""

import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from parser import extract_text
from analyzer import analyze_resume, match_with_jd

app = Flask(__name__)
app.secret_key = "dev-secret-key-change-this"  # needed for flash messages

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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

    # --- Save file ---
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # --- Extract text ---
    text, error = extract_text(filepath)
    if error:
        flash(error)
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
    app.run(debug=True, host="0.0.0.0", port=5000)
