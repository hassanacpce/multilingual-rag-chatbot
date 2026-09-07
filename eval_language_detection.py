"""
eval_language_detection.py

Hand-labeled accuracy test for language_detection.py.
"""
from language_detection import detect_language, _detect_raw

# (text, expected_code, conversation_language_context)
LABELED_SET = [
    ("What are your business hours?", "en", None),
    ("Can I get help with my order please?", "en", None),
    ("Cuáles son los horarios de atención?", "es", None),
    ("Necesito ayuda con mi pedido, por favor", "es", None),
    ("Quels sont vos horaires d ouverture?", "fr", None),
    ("J ai besoin d aide avec ma commande", "fr", None),
    ("आपके कार्य करने के घंटे क्या हैं?", "hi", None),
    ("मुझे अपने ऑर्डर में मदद चाहिए", "hi", None),
    ("तुमचे कामाचे तास काय आहेत?", "mr", None),
    ("मला माझ्या ऑर्डरसाठी मदत हवी आहे", "mr", None),
    ("thanks", "en", "en"),
    ("merci", "fr", "fr"),
    ("gracias", "es", "es"),
    ("धन्यवाद", "hi", "hi"),
    ("धन्यवाद", "mr", "mr"),
    ("ok great", "en", "en"),
    # Regression cases from real usage: short, unambiguous, SHOULD NOT
    # be overridden by unrelated prior conversation context.
    ("i have sinus", "en", "fr"),
    ("i have sinus", "en", "es"),
    ("i have sinus", "en", None),
    ("thanks", "en", "fr"),     # even with a DIFFERENT established context, "thanks" is unambiguous
    ("merci", "fr", "en"),      # same -- "merci" shouldn't be dragged to English context
]

if __name__ == "__main__":
    raw_correct = 0
    ctx_correct = 0
    print(f"{'Expected':10s} {'Ctx':6s} {'Raw (no ctx)':14s} {'System':16s}  Text")
    print("-" * 90)
    for text, expected, ctx in LABELED_SET:
        raw = _detect_raw(text)
        raw_code = raw[0][0] if raw else "?"
        raw_ok = raw_code == expected
        raw_correct += raw_ok

        system_result = detect_language(text, conversation_language=ctx)
        sys_ok = system_result.code == expected
        ctx_correct += sys_ok

        ctx_str = ctx or "-"
        print(f"{expected:10s} {ctx_str:6s} {raw_code:14s} {system_result.code:16s}  [{'OK' if sys_ok else 'MISS'}] {text}")

    n = len(LABELED_SET)
    print("-" * 90)
    print(f"Raw per-message accuracy (no context):     {raw_correct}/{n} = {raw_correct/n*100:.1f}%")
    print(f"System accuracy (dictionary + context):     {ctx_correct}/{n} = {ctx_correct/n*100:.1f}%")
