"""Response Generator - Generate natural language responses."""

from __future__ import annotations

import random
from typing import Any

from prism.memory.user_profile import UserProfile
from prism.memory.conversation_context import ConversationContext


class ResponseGenerator:
    """Generate natural, context-aware responses.
    
    Uses templates and context to produce human-like responses.
    Personalizes responses when user info is known.
    
    Example:
        >>> gen = ResponseGenerator(profile, context)
        >>> gen.greeting()  # "Hello Raphael! How can I help?"
        >>> gen.learned("cat is an animal")  # "Got it - cats are animals."
    """
    
    # Response templates by type
    TEMPLATES = {
        "greeting": [
            "Hello{name}! How can I help you today?",
            "Hi{name}! What would you like to know?",
            "Hey{name}! Ready to chat.",
        ],
        "greeting_return": [
            "Welcome back{name}!",
            "Good to see you again{name}!",
        ],
        "farewell": [
            "Goodbye{name}! See you next time.",
            "Bye{name}! Take care.",
            "Until next time{name}!",
        ],
        "learned_fact": [
            "Got it - {fact}.",
            "I'll remember that: {fact}.",
            "Noted! {fact}.",
            "Understood. {fact}.",
        ],
        "learned_name": [
            "Nice to meet you, {name}!",
            "Hello {name}! I'll remember that.",
            "Great to meet you, {name}!",
        ],
        "learned_preference": [
            "Got it - you {sentiment} {item}.",
            "Noted! You {sentiment} {item}.",
            "I'll remember that you {sentiment} {item}.",
        ],
        "no_answer": [
            "I don't have information about that.",
            "I'm not sure about that.",
            "I don't know the answer to that.",
        ],
        "clarify": [
            "Could you be more specific?",
            "I'm not sure I understand. Can you rephrase?",
            "What exactly would you like to know?",
        ],
        "context_continuation": [
            "Regarding {topic}: {response}",
            "About {topic} - {response}",
            "On {topic}: {response}",
        ],
    }
    
    def __init__(
        self,
        user_profile: UserProfile,
        context: ConversationContext,
    ) -> None:
        """Initialize generator."""
        self.profile = user_profile
        self.context = context
    
    def _name_part(self) -> str:
        """Get name part for templates."""
        if self.profile.user_name:
            return f", {self.profile.user_name}"
        return ""
    
    def _pick(self, template_type: str) -> str:
        """Pick a random template."""
        return random.choice(self.TEMPLATES[template_type])
    
    def greeting(self, returning: bool = False) -> str:
        """Generate a greeting."""
        template_type = "greeting_return" if returning else "greeting"
        return self._pick(template_type).format(name=self._name_part())
    
    def farewell(self) -> str:
        """Generate a farewell."""
        return self._pick("farewell").format(name=self._name_part())
    
    def learned_fact(self, fact: str) -> str:
        """Acknowledge learning a fact."""
        return self._pick("learned_fact").format(fact=fact)
    
    def learned_name(self, name: str) -> str:
        """Acknowledge learning user's name."""
        return self._pick("learned_name").format(name=name)
    
    def learned_preference(self, sentiment: str, item: str) -> str:
        """Acknowledge learning a preference."""
        sentiment_word = "like" if sentiment in ("like", "positive") else "don't like"
        return self._pick("learned_preference").format(sentiment=sentiment_word, item=item)
    
    def no_answer(self) -> str:
        """Generate a 'no answer' response."""
        return self._pick("no_answer")
    
    def clarify(self) -> str:
        """Ask for clarification."""
        return self._pick("clarify")
    
    def with_context(self, response: str) -> str:
        """Wrap response with context if applicable."""
        if self.context.current_topic and len(response) < 100:
            return self._pick("context_continuation").format(
                topic=self.context.current_topic,
                response=response.lower(),
            )
        return response
    
    def format_query_result(
        self,
        results: list[tuple[str, float]],
        query_type: str = "result",
    ) -> str:
        """Format query results as natural language."""
        if not results:
            return self.no_answer()
        
        # Single confident result
        if len(results) == 1 or results[0][1] > 0.8:
            return results[0][0]
        
        # Multiple results
        items = [w for w, s in results[:3] if s > 0.1]
        if len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} and {items[1]}"
        else:
            return f"{', '.join(items[:-1])}, and {items[-1]}"
    
    def format_similar(self, word: str, results: list[tuple[str, float]]) -> str:
        """Format similarity results."""
        if not results:
            return f"I don't have enough information about '{word}'."
        
        lines = [f"Similar to '{word}':"]
        for w, score in results[:5]:
            lines.append(f"  • {w} [{score:.2f}]")
        return "\n".join(lines)
    
    def format_analogy(
        self,
        a: str, b: str, c: str,
        results: list[tuple[str, float]],
    ) -> str:
        """Format analogy results."""
        header = f"{a} : {b} :: {c} : ?"
        
        if not results:
            return f"{header}\n  (no confident results)"
        
        lines = [header]
        for w, score in results[:3]:
            lines.append(f"  → {w} [{score:.2f}]")
        return "\n".join(lines)
    
    def personalize(self, response: str) -> str:
        """Add personalization to response."""
        if self.profile.user_name and random.random() < 0.3:
            # Sometimes add name
            if not response.endswith("?"):
                response = f"{response}, {self.profile.user_name}."
        return response
