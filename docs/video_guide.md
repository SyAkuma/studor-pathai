# Video Recording Guide

Practical instructions for recording the 10-minute walkthrough.

---

## Setup

**Screen layout before you hit record:**
1. Browser open to `https://github.com/SyAkuma/studor-pathai` — repo root visible
2. JupyterLab open in a second tab with `task1/task1v2.ipynb` loaded
3. Have these files ready to switch to quickly:
   - `task1/PLOTS/archetype_trajectories.png`
   - `task2/task2v2_executed.ipynb` (scroll to confusion matrix / AUROC section)
   - `task2/PLOTS/shap_top10.png`
   - `Task3/PLOTS/evaluation_comparison.png`
   - `Task3/OUTPUTS/recommendations.csv`

**Recording tool:** OBS, Loom, or Windows built-in (Win + G). Loom is easiest — one click, auto-uploads.

**Resolution:** 1920×1080 minimum. If on a 4K screen, set display scaling to 150% so text is readable in the recording.

**Audio:** Use a headset or earphones with a mic — laptop mic picks up keyboard and fan noise. Do a 10-second test recording and listen back before the full take.

**Camera:** Front-facing camera on, picture-in-picture bottom-right corner. Keep it on the whole time — it makes the presentation feel like a person talking, not a screencast.

---

## Before You Record

- Close Slack, email, notifications — anything that could pop up
- Silence your phone
- Do one full dry run out loud with a timer — not reading, just talking. You'll find the parts that are too slow
- Have `video_script.md` open on your phone or a second monitor for reference — don't read it, glance at it

---

## The Flow — What to Have on Screen When

| Time | What you're saying | What's on screen |
|------|-------------------|-----------------|
| 0:00–0:45 | Cold open, the problem | GitHub repo README |
| 0:45–1:15 | Introducing Task 1, 4 domains | task1v2.ipynb — feature table cell |
| 1:15–2:15 | Archetypes, trajectories | archetype_trajectories.png (open full screen) |
| 2:15–3:00 | Limitation, transition to T2 | Back to notebook briefly |
| 3:00–4:00 | Task 2 intro, temporal split, results | task2v2_executed.ipynb — AUROC/confusion matrix |
| 4:00–5:00 | SHAP top features | shap_top10.png (full screen) |
| 5:00–6:00 | Alert design, archetype-conditioned action | student_alerts_v2.csv or alert cell |
| 6:00–7:15 | Task 3, three signals | evaluation_comparison.png |
| 7:15–8:15 | Grid search result, cold-start | recommendations.csv briefly |
| 8:15–9:15 | Key decision — temporal split | task2/model_v2.py — temporal_split() function |
| 9:15–10:00 | Would do differently, close | Back to GitHub README |

---

## Tone and Delivery Tips

**Do:**
- Speak at 80% of your natural pace — you'll speed up when nervous
- Say the honest numbers directly: "we miss 22 in every 100 at-risk students" — don't soften it
- When you show a plot, give it 3 seconds of silence before speaking — let it register
- Name the limitation before they ask: it shows you understand the work, not just the output

**Don't:**
- Read from the script — it sounds robotic. Know the 6 section beats, fill in the words naturally
- Apologise for anything ("sorry this is a bit rough") — just present it
- Over-explain the code — show it briefly, then come back to what it means
- Go over 10 minutes — cut the Task 3 section short if needed, it's worth the least points

---

## If You Stumble

Loom and OBS let you re-record. Don't try to record a perfect single take — record in sections:
1. Cold open + Task 1 (3 min)
2. Task 2 (3 min)
3. Task 3 + close (4 min)

Edit together in any video editor (even Photos on Windows). Cuts are fine.

---

## File and Submission

- Export as MP4, 1080p
- File name: `studor_pathai_walkthrough.mp4`
- Upload to Google Drive or YouTube (unlisted) and share the link
- Do not upload directly to GitHub — video files are too large
