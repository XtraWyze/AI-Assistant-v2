"""
Unit tests for the internal messages[] representation.

Tests the message construction, flattening, and behavior compatibility
with the existing prompt format.

Architecture note:
- messages[] is internal only; transport to Ollama remains a single prompt string.
- This enables future streaming-to-TTS without changing routing.
"""
import unittest
import sys
import os

# Add wyzer to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from wyzer.brain.messages import (
    Message,
    msg_system,
    msg_user,
    msg_assistant,
    flatten_messages,
    MessageBuilder
)


class TestMessageConstruction(unittest.TestCase):
    """Test message construction helpers."""
    
    def test_msg_system_creates_system_role(self):
        """Test that msg_system creates a message with role='system'."""
        msg = msg_system("You are Wyzer.")
        
        self.assertEqual(msg["role"], "system")
        self.assertEqual(msg["content"], "You are Wyzer.")
    
    def test_msg_user_creates_user_role(self):
        """Test that msg_user creates a message with role='user'."""
        msg = msg_user("What time is it?")
        
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["content"], "What time is it?")
    
    def test_msg_assistant_creates_assistant_role(self):
        """Test that msg_assistant creates a message with role='assistant'."""
        msg = msg_assistant("It is 3 PM.")
        
        self.assertEqual(msg["role"], "assistant")
        self.assertEqual(msg["content"], "It is 3 PM.")
    
    def test_message_is_typed_dict(self):
        """Test that messages are plain dicts with expected keys."""
        msg = msg_system("test")
        
        self.assertIsInstance(msg, dict)
        self.assertIn("role", msg)
        self.assertIn("content", msg)


class TestFlattenMessages(unittest.TestCase):
    """Test flatten_messages function."""
    
    def test_flatten_empty_list_returns_empty_string(self):
        """Test that flattening empty list returns empty string."""
        result = flatten_messages([])
        
        self.assertEqual(result, "")
    
    def test_flatten_single_message(self):
        """Test flattening a single message."""
        messages = [msg_system("You are Wyzer.")]
        
        result = flatten_messages(messages)
        
        self.assertEqual(result, "You are Wyzer.")
    
    def test_flatten_multiple_messages_preserves_order(self):
        """Test that flattening preserves message order."""
        messages = [
            msg_system("System prompt."),
            msg_user("User input."),
            msg_assistant("Assistant reply.")
        ]
        
        result = flatten_messages(messages)
        
        # Check order is preserved
        system_pos = result.find("System prompt.")
        user_pos = result.find("User input.")
        assistant_pos = result.find("Assistant reply.")
        
        self.assertLess(system_pos, user_pos)
        self.assertLess(user_pos, assistant_pos)
    
    def test_flatten_uses_block_separator(self):
        """Test that messages are separated by block separator."""
        messages = [
            msg_system("First"),
            msg_user("Second")
        ]
        
        result = flatten_messages(messages, block_separator="\n\n")
        
        self.assertEqual(result, "First\n\nSecond")
    
    def test_flatten_custom_separator(self):
        """Test custom block separator."""
        messages = [
            msg_system("A"),
            msg_user("B")
        ]
        
        result = flatten_messages(messages, block_separator="---")
        
        self.assertEqual(result, "A---B")
    
    def test_flatten_without_role_headers_default(self):
        """Test that role headers are NOT included by default."""
        messages = [
            msg_system("System text"),
            msg_user("User text")
        ]
        
        result = flatten_messages(messages)
        
        # Should NOT contain role headers
        self.assertNotIn("System:", result)
        self.assertNotIn("User:", result)
        self.assertIn("System text", result)
        self.assertIn("User text", result)
    
    def test_flatten_with_role_headers(self):
        """Test that role headers ARE included when requested."""
        messages = [
            msg_system("System text"),
            msg_user("User text")
        ]
        
        result = flatten_messages(messages, include_role_headers=True)
        
        # Should contain role headers
        self.assertIn("System:", result)
        self.assertIn("User:", result)
    
    def test_flatten_skips_empty_messages(self):
        """Test that empty messages are skipped."""
        messages = [
            msg_system("Valid"),
            msg_user(""),
            msg_assistant("   "),  # whitespace only
            msg_user("Also valid")
        ]
        
        result = flatten_messages(messages)
        
        self.assertIn("Valid", result)
        self.assertIn("Also valid", result)
        # Should only have one separator (between the two valid messages)
        self.assertEqual(result, "Valid\n\nAlso valid")
    
    def test_flatten_strips_content_whitespace(self):
        """Test that message content is stripped."""
        messages = [
            msg_system("  trimmed  ")
        ]
        
        result = flatten_messages(messages)
        
        self.assertEqual(result, "trimmed")


class TestMessageBuilder(unittest.TestCase):
    """Test MessageBuilder convenience class."""
    
    def test_builder_starts_empty(self):
        """Test that builder starts with no messages."""
        builder = MessageBuilder()
        
        self.assertEqual(len(builder), 0)
        self.assertFalse(builder)
    
    def test_builder_add_system(self):
        """Test adding system message."""
        builder = MessageBuilder()
        builder.system("System prompt")
        
        messages = builder.build()
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "System prompt")
    
    def test_builder_add_user(self):
        """Test adding user message."""
        builder = MessageBuilder()
        builder.user("User input")
        
        messages = builder.build()
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
    
    def test_builder_add_assistant(self):
        """Test adding assistant message."""
        builder = MessageBuilder()
        builder.assistant("Reply")
        
        messages = builder.build()
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "assistant")
    
    def test_builder_chaining(self):
        """Test method chaining."""
        builder = MessageBuilder()
        
        result = builder.system("A").user("B").assistant("C")
        
        self.assertIs(result, builder)
        self.assertEqual(len(builder), 3)
    
    def test_builder_skips_empty_content(self):
        """Test that empty content is skipped."""
        builder = MessageBuilder()
        builder.system("Valid")
        builder.user("")  # Should be skipped
        builder.assistant("   ")  # Should be skipped
        
        messages = builder.build()
        
        self.assertEqual(len(messages), 1)
    
    def test_builder_flatten(self):
        """Test builder flatten method."""
        builder = MessageBuilder()
        builder.system("System").user("User")
        
        result = builder.flatten()
        
        self.assertEqual(result, "System\n\nUser")
    
    def test_builder_clear(self):
        """Test builder clear method."""
        builder = MessageBuilder()
        builder.system("A").user("B")
        
        builder.clear()
        
        self.assertEqual(len(builder), 0)
    
    def test_builder_add_message(self):
        """Test adding pre-constructed message."""
        builder = MessageBuilder()
        msg = msg_system("Pre-built")
        
        builder.add(msg)
        
        messages = builder.build()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["content"], "Pre-built")
    
    def test_builder_build_returns_copy(self):
        """Test that build() returns a copy, not the internal list."""
        builder = MessageBuilder()
        builder.system("Original")
        
        messages = builder.build()
        messages.append(msg_user("Added"))
        
        # Original builder should be unchanged
        self.assertEqual(len(builder), 1)


class TestBehaviorCompatibility(unittest.TestCase):
    """Test that the messages system produces compatible output."""
    
    def test_system_user_order_preserved(self):
        """Test that system prompt comes before user input."""
        messages = [
            msg_system("You are Wyzer."),
            msg_user("Hello")
        ]
        
        result = flatten_messages(messages)
        
        # System should come first
        self.assertTrue(result.startswith("You are Wyzer."))
        self.assertIn("Hello", result)
    
    def test_multiple_system_blocks_concatenate(self):
        """Test that multiple system blocks are properly concatenated."""
        messages = [
            msg_system("Base prompt."),
            msg_system("Session context."),
            msg_system("Memory block."),
            msg_user("Query")
        ]
        
        result = flatten_messages(messages)
        
        # All blocks should appear in order
        base_pos = result.find("Base prompt.")
        session_pos = result.find("Session context.")
        memory_pos = result.find("Memory block.")
        query_pos = result.find("Query")
        
        self.assertLess(base_pos, session_pos)
        self.assertLess(session_pos, memory_pos)
        self.assertLess(memory_pos, query_pos)
    
    def test_memory_block_optional(self):
        """Test that memory block can be optionally included."""
        # Without memory
        messages_no_memory = [
            msg_system("System"),
            msg_user("Query")
        ]
        
        # With memory
        messages_with_memory = [
            msg_system("System"),
            msg_system("Memory: User's name is Bob"),
            msg_user("Query")
        ]
        
        result_no = flatten_messages(messages_no_memory)
        result_with = flatten_messages(messages_with_memory)
        
        # Memory block should only appear when included
        self.assertNotIn("Memory:", result_no)
        self.assertIn("Memory:", result_with)
    
    def test_no_tool_syntax_introduced(self):
        """Test that no tool/function syntax is introduced in output."""
        messages = [
            msg_system("You are Wyzer."),
            msg_user("Open chrome"),
            msg_assistant("Opening Chrome.")
        ]
        
        result = flatten_messages(messages)
        
        # Should not contain any function/tool syntax
        self.assertNotIn("function_call", result)
        self.assertNotIn("tool_call", result)
        self.assertNotIn("[TOOL]", result)
        self.assertNotIn("<tool>", result)


class TestLLMEngineMessageBuilding(unittest.TestCase):
    """Test that LLMEngine properly builds messages internally."""
    
    def test_engine_imports_successfully(self):
        """Test that LLMEngine can be imported with new message support."""
        # This tests that the imports work correctly
        from wyzer.brain.llm_engine import LLMEngine
        
        # Engine should have the new methods
        self.assertTrue(hasattr(LLMEngine, '_build_messages'))
        self.assertTrue(hasattr(LLMEngine, '_flatten_to_prompt'))


class TestPromptFormatCompatibility(unittest.TestCase):
    """Test that prompt.py message functions work correctly."""
    
    def test_build_prompt_messages_exists(self):
        """Test that build_prompt_messages function exists."""
        from wyzer.brain.prompt import build_prompt_messages
        
        # Should be callable
        self.assertTrue(callable(build_prompt_messages))
    
    def test_format_prompt_from_messages_exists(self):
        """Test that format_prompt_from_messages function exists."""
        from wyzer.brain.prompt import format_prompt_from_messages
        
        # Should be callable
        self.assertTrue(callable(format_prompt_from_messages))


if __name__ == "__main__":
    unittest.main()
