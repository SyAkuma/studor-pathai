# 10-Minute Walkthrough Video Script
# Studor PathAI Engine — DS Screening

---

## FORMAT NOTES
- Total time: 10 minutes hard cap
- Tone: confident, direct, like you're presenting to a founding team — not defending a thesis
- Screen share: have the GitHub repo open, then switch to notebooks as needed
- Camera on the whole time if possible — it's a product pitch, not a lecture

---

## [0:00 – 0:45] COLD OPEN — The Problem

> "University staff find out a student is at risk when they stop showing up to exams.
> By then it's too late. The grade is already failing, the withdrawal is already logged.
>
> The question I was given is: can we detect disengagement in behaviour — in how students
> move through a learning platform — weeks before it shows up in a grade?
>
> I built three systems on the Open University Learning Analytics Dataset — 32,000 students,
> 7 modules, 10 million click events — to answer that question."

*[Show repo README on screen]*

---

## [0:45 – 3:00] TASK 1 — Engagement Score

> "Task 1 is the foundation. Every student gets a score from 0 to 100 each week, built
> entirely from behaviour — not grades.
>
> I engineered 10 features across 4 domains."

*[Show task1/task1v2.ipynb — the feature domain table or archetype trajectory plot]*

> "The domains are:
> VLE Presence — are they showing up? How often, how consistently?
> VLE Quality — when they log in, what are they doing? Studying course content,
>   or just refreshing the homepage?
> Trajectory — is the trend going up or down? A student at 80 who's declining is
>   more at risk than one at 40 who's recovering.
> Submission behaviour — are they submitting assignments? How late?
>
> The submission domain gets the highest weight — 30%. That was a deliberate choice,
> backed by the SHAP analysis in Task 2, which independently confirmed that assessment
> completion is the second strongest predictor of withdrawal.
>
> The score produces 9 student archetypes — here are three trajectories:"

*[Show archetype_trajectories.png from task1/PLOTS/]*

> "Ghost — never really showed up. High withdrawal rate, obvious in week 1.
> Steady Engager — consistent, rising, low risk.
> Recovering — dipped mid-semester but came back. These students are recoverable
>   if you catch them during the dip, not after.
>
> One honest limitation: the activity diversity feature counts distinct activity types
> rather than using Shannon entropy — so a student clicking one resource 500 times
> looks as diverse as one genuinely spread across 10 types. That's the first thing
> I'd fix with more time."

---

## [3:00 – 6:00] TASK 2 — Disengagement Prediction

> "Task 2 uses the week-6 snapshot of Task 1's output to make a single binary prediction:
> will this student withdraw or fail?
>
> The hardest constraint was leakage. Any feature using data after week 6 is cheating —
> the model would know the future. I explicitly excluded the withdrawal date, exam scores,
> and the full-semester archetype label."

*[Show task2/task2v2_executed.ipynb — the confusion matrix or AUROC plot]*

> "I trained three models. XGBoost won — AUROC 0.849, Recall 0.777 on a held-out 2014
> test set. The train/test split is temporal — 2013 presentations train the model,
> 2014 presentations test it. That's how it would work in deployment: you always predict
> future cohorts from past ones. A random split inflates AUROC by 2–4 points and doesn't
> reflect reality."

*[Show shap_top10.png from task2/PLOTS/]*

> "The top 3 features from SHAP:
> Number one — assessment score by week 6. Students below cohort average on their first
>   two assignments very rarely recover without intervention.
> Number two — completion rate. Missing even one submission by week 4 roughly doubles
>   withdrawal probability.
> Number three — module identity. GGG has a 67% dropout rate; CCC has 38%. The model
>   has to control for structural module risk before comparing individual students.
>
> I optimised for Recall, not accuracy. Missing an at-risk student who then withdraws
> destroys trust in the system permanently. A false alarm costs one 10-minute conversation.
> At the operating threshold, we catch 78 in every 100 at-risk students — 22 are missed.
> That's the honest number."

*[Show student_alerts_v2.csv briefly or the alert design slide]*

> "Each flag comes with three plain-English reasons and an archetype-conditioned action —
> not just a probability. A Ghost student needs outreach to re-establish contact.
> A Willing-but-Struggling student is already engaged — they need tutoring, not a
> motivational call. Same risk score, completely different intervention."

---

## [6:00 – 8:15] TASK 3 — Recommendation Engine

> "Task 3 recommends the three most suitable next modules for each student.
> The primary user is staff — advisors designing progression pathways — not students
> self-selecting. That framing matters because staff need explainable recommendations
> they can defend."

*[Show Task3/PLOTS/evaluation_comparison.png]*

> "I implemented three signals and combined them:
>
> Content-based — course metadata matched to student profile via cosine similarity.
> Collaborative filtering — find 20 students with similar VLE engagement patterns,
>   see what modules they took next.
> Markov chain — the key insight here: condition on outcome.
>   A student who *failed* module AAA should get different next-module recommendations
>   than one who *passed* it. They're at different points academically.
>
> The grid search found the best weights: Markov 0.9, CF 0.1, Content 0.0.
> Content got zero weight because OULAD's course metadata is too sparse to differentiate
> modules. In a real system with syllabi and topic tags, content-based would carry real
> weight. The methodology is right — the data is the constraint.
>
> Precision@3 on the 2014 holdout: 0.259 versus 0.215 for a simple popularity baseline.
> One honest caveat: the Markov chain was trained on data that partially overlaps the
> evaluation set, so P@3 is slightly inflated. Training on 2013-only data is the fix."

*[Show recommendations.csv briefly]*

> "For cold-start — brand new students with no history — the system recommends the
> most-enrolled modules, filtered to exclude high-withdrawal courses for students
> already flagged as high-risk by Task 2. Honest, product-safe, and doesn't pretend
> to personalise when there's no signal to work with."

---

## [8:15 – 9:15] ONE KEY DECISION

> "The key decision I'd highlight is the temporal train/test split in Task 2.
>
> The naive approach — random stratified split — mixes 2013 and 2014 students in both
> train and test. The model sees some 2014 students during training. That's not how
> deployment works. In deployment you always predict future cohorts from past cohorts.
>
> Switching to a temporal split dropped my AUROC from 0.89 to 0.85. That 4-point drop
> is real — it's the gap between what looks good in a notebook and what you'd actually
> see in production.
>
> I'd rather report 0.85 that holds in the real world than 0.89 that doesn't."

---

## [9:15 – 10:00] ONE THING I'D DO DIFFERENTLY

> "If I had more time, the single highest-value fix is separating the calibration step
> from the threshold selection step in Task 2.
>
> Right now, I fit the Platt calibrator on the validation set, then also select the
> operating threshold on the same validation set. These two steps share the same data.
> The calibrator has already seen the answers when the threshold is tuned — which makes
> the Brier score, the calibration quality measure, look slightly better than it is.
>
> The fix is simple: a third held-out set, or cross-calibration across folds.
> I know exactly what's wrong and exactly how to fix it. I ran out of time to re-run.
>
> More broadly — I'd get this in front of actual university advisors within 90 days.
> The archetype names, the alert format, the threshold — all of that should be validated
> by the people who'll use it, not just optimised against a dataset.
>
> Thanks."

---

## TIMING GUIDE

| Section | Target | Hard limit |
|---------|--------|-----------|
| Cold open | 45s | 60s |
| Task 1 | 2:15 | 2:30 |
| Task 2 | 3:00 | 3:15 |
| Task 3 | 2:15 | 2:30 |
| Key decision | 60s | 75s |
| Would do differently | 45s | 60s |
| **Total** | **~10:00** | **10:00** |
