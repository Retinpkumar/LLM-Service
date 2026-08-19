from llm_service import main


def test_main_prints_greeting(capsys):
    main()
    captured = capsys.readouterr()
    assert "Hello from llm-service!" in captured.out

