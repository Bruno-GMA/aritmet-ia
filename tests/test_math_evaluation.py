from pipeline.eval.math_evaluation import aggregate_math_evaluations, evaluate_analogous_answer


def test_correct_analogous_example_is_not_compared_to_original_answer():
    result = evaluate_analogous_answer("Exemplo: 3 + 4 = 7")
    assert result["status"] == "correct"
    assert result["correct"] == 1


def test_incorrect_and_missing_equations_are_distinct():
    incorrect = evaluate_analogous_answer("2 * 3 = 5")
    missing = evaluate_analogous_answer("Use uma soma e explique cada passo.")
    assert incorrect["status"] == "incorrect"
    assert missing["status"] == "not_evaluable"
    aggregate = aggregate_math_evaluations([incorrect, missing])
    assert aggregate["accuracy"] == 0
    assert aggregate["coverage"] == 1
