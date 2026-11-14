from deepeval.test_case import LLMTestCase

from test_rag import rag_quality

# We test the JUDGE here, not the bot: we write the answers ourselves,
# so we already know which ones deserve to pass.
QUESTION = "¿Cuánto cuesta la hamburguesa en combo?"
EXPECTED = "La hamburguesa en combo cuesta $35.000 e incluye papas y gaseosa."
MENU = (
    "Hamburguesa\nCarne de hamburguesa, proteína a elección, pan artesanal, queso, "
    "guacamole y pico de gallo.\nSolo $26.000\nCombo $35.000\nCombo: papas y gaseosa."
)

answers = [
    ("correct, same words", True, EXPECTED),
    ("correct, other words", True, "Sale en 35 mil pesos, con papitas y gaseosa 🍔"),
    ("wrong price", False, "La hamburguesa en combo cuesta $30.000 e incluye papas y gaseosa."),
    ("made-up extra", False, "Cuesta $35.000 e incluye papas, gaseosa y un postre gratis."),
    ("long, friendly, wrong", False,
     "¡Excelente elección! Nuestra hamburguesa es una de las favoritas de la casa, preparada con "
     "pan artesanal, queso derretido, guacamole fresco y pico de gallo. En combo la puedes "
     "disfrutar por solo $30.000, acompañada de papas y una gaseosa bien fría. ¡Te va a encantar!"),
    ("dodges the question", False, "¡Gran elección! Nuestras hamburguesas son deliciosas."),
    ("correct, but very rude", False, "$35.000, ¿no sabes leer el menú o que?"),
]

agree = 0
for name, should_pass, answer in answers:
    test_case = LLMTestCase(
        input=QUESTION,
        actual_output=answer,
        expected_output=EXPECTED,
        retrieval_context=[MENU],
    )
    score = rag_quality.measure(test_case)
    judge_pass = score >= rag_quality.threshold
    agree += judge_pass == should_pass

    mark = "✅ agrees" if judge_pass == should_pass else "❌ DISAGREES"
    print(f"{name:<24} score {score:.2f}  judge: {'pass' if judge_pass else 'fail'}  "
          f"we say: {'pass' if should_pass else 'fail'}  {mark}")

print(f"\nJudge agreed with us on {agree} of {len(answers)} answers.")
