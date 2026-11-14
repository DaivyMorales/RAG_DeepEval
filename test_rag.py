import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import JevEval
from deepeval.metrics.jev_eval import Noul
from deepeval.test_case import LLMTestCase, SingleTurnParams

import re

from rag import query_engine


def load_goldens(path):
    dataset = EvaluationDataset()
    dataset.add_goldens_from_json_file(path)
    return dataset.goldens


MENU_GOLDENS = load_goldens("goldens.json")
EDGE_GOLDENS = load_goldens(
    "edge_cases.json"
)

# Jev answers each yes/no question with a probability; the score is their average.
rag_quality = JevEval(
    name="RAG quality",
    evaluation_params=[
        SingleTurnParams.INPUT,
        SingleTurnParams.ACTUAL_OUTPUT,
        SingleTurnParams.EXPECTED_OUTPUT,
        SingleTurnParams.RETRIEVAL_CONTEXT,
    ],
    strict_mode=True,
    questions=[
        # 1. Direct answer
        Noul(
            "Does `test_case.actual_output` directly answer the customer's question "
            "in `test_case.input`?",
            true="the response answers what the customer asked",
            false="the response is off-topic, evasive, or answers a different question",
        ),
        # 2. Facts in the menu
        Noul(
            "Is every fact in `test_case.actual_output` (items, prices, ingredients) "
            "stated in the menu passages in `test_case.retrieval_context`?",
            true="every fact in the response appears in the menu passages",
            false="the response states at least one fact the menu passages do not",
        ),
        # 3. Price agreement
        Noul(
            "Does `test_case.actual_output` agree with the correct answer in "
            "`test_case.expected_output` on the key facts, especially prices?",
            true="the response gives the same key facts and prices as the correct answer",
            false="the response gives a different price or contradicts the correct answer",
        ),
        # 4. Politeness
        Noul(
            "Is `test_case.actual_output` polite and friendly towards the customer?",
            true="the response is polite and friendly towards the customer",
            false="the response is rude or disrespectful towards the customer",
        )
    ],
    threshold=0.7,  
)


def check_golden(golden):
    response = query_engine.query(golden.input)

    test_case = LLMTestCase(
        input=golden.input,
        actual_output=str(response),
        expected_output=golden.expected_output,
        # The menu chunks the RAG found; the second question checks the answer against these.
        retrieval_context=[node.get_content() for node in response.source_nodes],
    )

    assert_test(test_case, [rag_quality])

def prices(text):
    return re.findall(r'\$[\d.]+\d', text)

def exact_match(actual, expected):
    bot_prices = prices(actual)
    # An answer with no price must fail, or it would pass with nothing checked.
    return len(bot_prices) > 0 and all(p in prices(expected) for p in bot_prices)



@pytest.mark.parametrize("golden", MENU_GOLDENS, ids=lambda g: g.input)
def test_menu_question(golden):
    check_golden(golden)


@pytest.mark.parametrize("golden", EDGE_GOLDENS, ids=lambda g: g.input)
def test_edge_case(golden):
    check_golden(golden)


@pytest.mark.parametrize("golden", MENU_GOLDENS, ids=lambda g: g.input)
def test_exact_match(golden):
    if not prices(golden.expected_output):
        pytest.skip("no price in the expected answer")
    response = query_engine.query(golden.input)
    assert exact_match(str(response), golden.expected_output)
