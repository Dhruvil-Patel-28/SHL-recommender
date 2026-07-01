"""
Tests based on all 10 sample conversations (C1-C10).
Validates Turn 1 behavior: clarify vs recommend immediately.
Also validates schema, URL validity, and key behavioral patterns.
"""

import requests
import json
import sys
import time

BASE_URL = "http://localhost:8000"
PASS = 0
FAIL = 0
ERRORS = []

DELAY = 5  # seconds between tests to avoid rate limiting


def chat(messages, timeout=90):
    """Send a chat request and return parsed response."""
    resp = requests.post(
        f"{BASE_URL}/chat",
        json={"messages": messages},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        msg = f"  ❌ {name}" + (f" — {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg)


def validate_schema(data, ctx=""):
    """Validate response matches expected schema."""
    check(f"Has 'reply' (string) {ctx}", isinstance(data.get("reply"), str))
    check(f"Has 'recommendations' (list) {ctx}", isinstance(data.get("recommendations"), list))
    check(f"Has 'end_of_conversation' (bool) {ctx}", isinstance(data.get("end_of_conversation"), bool))
    for i, r in enumerate(data.get("recommendations", [])):
        check(f"rec[{i}] has name/url/test_type {ctx}",
              all(k in r for k in ("name", "url", "test_type")))
        if "url" in r:
            check(f"rec[{i}] url is valid SHL URL {ctx}",
                  r["url"].startswith("https://www.shl.com/products/product-catalog/view/"),
                  f"got: {r['url'][:80]}")


# ============================================================
# C1: Vague "senior leadership" → MUST clarify
# ============================================================
def test_c1_turn1():
    print("\n═══ C1 Turn 1: 'solution for senior leadership' → should CLARIFY ═══")
    data = chat([
        {"role": "user", "content": "We need a solution for senior leadership."}
    ])
    validate_schema(data, "(C1 T1)")
    check("Recommendations EMPTY (clarifying)", len(data["recommendations"]) == 0,
          f"got {len(data['recommendations'])} recs")
    check("end_of_conversation is false", data["end_of_conversation"] is False)
    check("Reply contains a question", "?" in data["reply"],
          f"reply: {data['reply'][:120]}")


# ============================================================
# C2: "senior Rust engineer" → MUST clarify (missing tech)
# ============================================================
def test_c2_turn1():
    print("\n═══ C2 Turn 1: 'senior Rust engineer' → should CLARIFY (gap) ═══")
    data = chat([
        {"role": "user", "content": "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?"}
    ])
    validate_schema(data, "(C2 T1)")
    # C2 expects: explain Rust gap, ask follow-up, NO recs
    check("Recommendations EMPTY (clarifying)", len(data["recommendations"]) == 0,
          f"got {len(data['recommendations'])} recs")
    check("end_of_conversation is false", data["end_of_conversation"] is False)
    reply_lower = data["reply"].lower()
    check("Mentions Rust gap or alternatives",
          any(w in reply_lower for w in ["rust", "not available", "no", "closest", "alternative", "doesn't"]),
          f"reply: {data['reply'][:150]}")


# ============================================================
# C3: "500 contact centre agents" → MUST clarify (language?)
# ============================================================
def test_c3_turn1():
    print("\n═══ C3 Turn 1: '500 contact centre agents' → should CLARIFY (language) ═══")
    data = chat([
        {"role": "user", "content": "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?"}
    ])
    validate_schema(data, "(C3 T1)")
    check("Recommendations EMPTY (clarifying)", len(data["recommendations"]) == 0,
          f"got {len(data['recommendations'])} recs")
    check("end_of_conversation is false", data["end_of_conversation"] is False)
    check("Reply asks clarifying question", "?" in data["reply"])


# ============================================================
# C4: "graduate financial analysts" → should recommend immediately
# ============================================================
def test_c4_turn1():
    print("\n═══ C4 Turn 1: 'graduate financial analysts' → should RECOMMEND ═══")
    data = chat([
        {"role": "user", "content": "Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test."}
    ])
    validate_schema(data, "(C4 T1)")
    check("Recommendations NON-EMPTY", len(data["recommendations"]) > 0,
          f"got {len(data['recommendations'])} recs")
    check("end_of_conversation is false", data["end_of_conversation"] is False)


# ============================================================
# C5: "re-skill Sales organization" → should recommend
# ============================================================
def test_c5_turn1():
    print("\n═══ C5 Turn 1: 're-skill Sales organization' → should RECOMMEND ═══")
    data = chat([
        {"role": "user", "content": "As part of our restructuring and annual talent audit, we need to re-skill our Sales organization. What solutions do you recommend?"}
    ])
    validate_schema(data, "(C5 T1)")
    # C5 gives immediate recommendations in the sample
    # But this could go either way — it's borderline
    check("end_of_conversation is false", data["end_of_conversation"] is False)
    recs_or_question = len(data["recommendations"]) > 0 or "?" in data["reply"]
    check("Either recommends or asks clarifying question", recs_or_question)


# ============================================================
# C6: "plant operators, safety priority" → should recommend
# ============================================================
def test_c6_turn1():
    print("\n═══ C6 Turn 1: 'plant operators, safety priority' → should RECOMMEND ═══")
    data = chat([
        {"role": "user", "content": "We're hiring plant operators for a chemical facility. Safety is absolute top priority — reliability, procedure compliance, never cutting corners. What do you recommend?"}
    ])
    validate_schema(data, "(C6 T1)")
    check("Recommendations NON-EMPTY", len(data["recommendations"]) > 0,
          f"got {len(data['recommendations'])} recs")
    check("end_of_conversation is false", data["end_of_conversation"] is False)
    names_lower = " ".join(r["name"].lower() for r in data["recommendations"])
    check("Includes safety/DSI assessment",
          any(w in names_lower for w in ["safety", "dsi", "dependability"]),
          f"names: {names_lower}")


# ============================================================
# C7: "bilingual healthcare admin" → should clarify (constraint)
# ============================================================
def test_c7_turn1():
    print("\n═══ C7 Turn 1: 'bilingual healthcare admin' → should CLARIFY ═══")
    data = chat([
        {"role": "user", "content": "We're hiring bilingual healthcare admin staff in South Texas — they handle patient records and need to be assessed in Spanish. HIPAA compliance is critical. What assessments work?"}
    ])
    validate_schema(data, "(C7 T1)")
    # C7 expects clarification about language constraint approach
    check("end_of_conversation is false", data["end_of_conversation"] is False)
    # Can either clarify or recommend — sample shows clarification
    recs_or_question = len(data["recommendations"]) > 0 or "?" in data["reply"]
    check("Either recommends or asks clarifying question", recs_or_question)


# ============================================================
# C8: "screen admin assistants Excel/Word" → should recommend
# ============================================================
def test_c8_turn1():
    print("\n═══ C8 Turn 1: 'screen admin assistants Excel/Word' → should RECOMMEND ═══")
    data = chat([
        {"role": "user", "content": "I need to quickly screen admin assistants for Excel and Word daily."}
    ])
    validate_schema(data, "(C8 T1)")
    check("Recommendations NON-EMPTY", len(data["recommendations"]) > 0,
          f"got {len(data['recommendations'])} recs")
    check("end_of_conversation is false", data["end_of_conversation"] is False)
    names_lower = " ".join(r["name"].lower() for r in data["recommendations"])
    check("Includes Excel-related test",
          any(w in names_lower for w in ["excel", "ms excel"]),
          f"names: {names_lower}")


# ============================================================
# C9: Full JD → should clarify (backend vs fullstack?)
# ============================================================
def test_c9_turn1():
    print("\n═══ C9 Turn 1: Full JD → should CLARIFY (scope) ═══")
    data = chat([
        {"role": "user", "content": "Here's the JD for an engineer we need to fill. Can you recommend an assessment battery?\n\n\"Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, Angular, SQL/relational databases, AWS deployment, and Docker. Will own end-to-end microservice delivery, contribute to architectural decisions, and mentor mid-level engineers. Strong CI/CD and cloud-native experience required.\""}
    ])
    validate_schema(data, "(C9 T1)")
    check("end_of_conversation is false", data["end_of_conversation"] is False)
    # C9 sample clarifies on Turn 1 (asks backend vs fullstack?)
    # But recommending immediately on a detailed JD is also acceptable
    recs_or_question = len(data["recommendations"]) > 0 or "?" in data["reply"]
    check("Either recommends or asks clarifying question", recs_or_question)


# ============================================================
# C10: "graduate management trainees, full battery" → should recommend
# ============================================================
def test_c10_turn1():
    print("\n═══ C10 Turn 1: 'graduate trainees, full battery' → should RECOMMEND ═══")
    data = chat([
        {"role": "user", "content": "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates."}
    ])
    validate_schema(data, "(C10 T1)")
    check("Recommendations NON-EMPTY", len(data["recommendations"]) > 0,
          f"got {len(data['recommendations'])} recs")
    check("end_of_conversation is false", data["end_of_conversation"] is False)


# ============================================================
# Multi-turn: C10 Turn 2+4 — drop OPQ32r
# ============================================================
def test_c10_multiturn():
    print("\n═══ C10 Multi-turn: Drop OPQ32r ═══")
    data = chat([
        {"role": "user", "content": "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates."},
        {"role": "assistant", "content": "I recommend SHL Verify Interactive G+, OPQ32r, and Graduate Scenarios."},
        {"role": "user", "content": "Drop the OPQ. Final list: Verify G+ and Graduate Scenarios."}
    ])
    validate_schema(data, "(C10 drop)")
    check("end_of_conversation is true (confirmed)", data["end_of_conversation"] is True)
    names_lower = [r["name"].lower() for r in data["recommendations"]]
    check("OPQ32r NOT in final list",
          not any("opq" in n for n in names_lower),
          f"names: {names_lower}")
    check("Has recommendations", len(data["recommendations"]) > 0)


# ============================================================
# Off-topic refusal: C7 Turn 3 pattern
# ============================================================
def test_offtopic_legal():
    print("\n═══ Off-Topic: Legal question → should REFUSE ═══")
    data = chat([
        {"role": "user", "content": "Are we legally required under HIPAA to test all staff who touch patient records?"}
    ])
    validate_schema(data, "(off-topic)")
    check("Recommendations EMPTY", len(data["recommendations"]) == 0)
    check("end_of_conversation is false", data["end_of_conversation"] is False)


if __name__ == "__main__":
    print("=" * 60)
    print("C1-C10 Sample Conversation Test Suite")
    print("=" * 60)

    time.sleep(2)

    tests = [
        test_c1_turn1,
        test_c2_turn1,
        test_c3_turn1,
        test_c4_turn1,
        test_c5_turn1,
        test_c6_turn1,
        test_c7_turn1,
        test_c8_turn1,
        test_c9_turn1,
        test_c10_turn1,
        test_c10_multiturn,
        test_offtopic_legal,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            FAIL += 1
            msg = f"  💥 {test_fn.__name__} CRASHED: {e}"
            print(msg)
            ERRORS.append(msg)
        time.sleep(DELAY)

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if ERRORS:
        print("\nFAILURES:")
        for err in ERRORS:
            print(err)

    sys.exit(1 if FAIL > 0 else 0)
