import unittest

from langchain.chat_models import AzureChatOpenAI

from src.main.lab import agent_executor_no_memory, agent_executor_with_memory

"""
This file will contain test cases for the automatic evaluation of your
solution in main/lab.py. You should not modify the code in this file. You should
also manually test your solution by running app.py.
"""

class TestLLMResponse(unittest.TestCase):
    """
    This test will verify that the connection to an external LLM is made. If it does not
    work, this may be because the API key is invalid, or the service may be down.
    If that is the case, this lab may not be completable.
    """
    def test_llm_sanity_check(self):
        llm = AzureChatOpenAI(model_name="gpt-35-turbo")

    """
    This test will verify that the agent without memory works, but does not remember facts about the conversation
    """
    def test_agent_with_no_memory(self):

        agent_executor_no_memory.invoke(
            {"input": "Hi, how are you? My name is Developer"},
        )

        self.assertNotIn("Developer", agent_executor_no_memory("Do you know my name?"))

    """
    This test will verify that the agent with memory is able to remember facts about the conversation.
    """
    def test_agent_remembers_conversation(self):
        agent_executor_with_memory.invoke(
            {"input": "Hi, how are you? My name is Developer"},
        )

        response = agent_executor_with_memory.invoke(
            {"input": "Do you know my name?"},
        )

        self.assertIn("Developer", response["output"])

    """
    This test will verify that the agent produces the correct answer, even if the user hasn't stated it.
    This correct output will be based on the previously mentioned historical figure (Trajan in this case)
    """
    def test_agent_gets_correct_answer(self):

        agent_executor_with_memory.invoke(
            {"input": "What can you tell me about Trajan?"},
        )

        response = agent_executor_with_memory.invoke(
            {"input": "Who was the historical figure married to?"},
        )

        self.assertIn("Pompeia" or "Plotina", response["output"])
