"""User Statement Parser - Parse personal statements."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from prism.core.lexicon import Lexicon
from prism.memory.user_profile import UserProfile


@dataclass
class ParseResult:
    """Result of parsing a user statement."""
    
    type: str  # "identity", "preference", "fact", "question"
    data: dict
    original: str = ""
    handled: bool = False


class UserStatementParser:
    """Parse user-specific statements.
    
    Recognizes patterns like:
    - "My name is X" → identity
    - "I like X" / "I hate X" → preference
    - "I am X" / "I have X" / "I can X" → fact
    - "What do I like?" → preference query
    
    Example:
        >>> parser = UserStatementParser(lexicon, profile)
        >>> result = parser.parse("My name is Raphael")
        >>> result.type  # "identity"
        >>> result.data  # {"name": "Raphael"}
    """
    
    def __init__(
        self,
        lexicon: Lexicon,
        user_profile: UserProfile,
    ) -> None:
        """Initialize parser."""
        self.lexicon = lexicon
        self.profile = user_profile
        
        # Pattern rules: (regex, type, handler)
        # NOTE: Order matters! More specific patterns first.
        self._patterns: list[tuple[re.Pattern, str, Callable]] = [
            # Preference patterns - positive (check before "I am")
            (re.compile(r"i (?:also )?(?:really )?(?:like|love|enjoy|adore) (.+)", re.I),
             "preference", lambda m: self._handle_preference(m, "like")),
            
            # Preference patterns - negative
            (re.compile(r"i (?:also )?(?:really )?(?:hate|dislike|don'?t like|can'?t stand) (.+)", re.I),
             "preference", lambda m: self._handle_preference(m, "dislike")),
            
            # Identity patterns (specific name phrases)
            (re.compile(r"(?:my name is|i'?m called|call me) (\w+)", re.I), 
             "identity", self._handle_name),
            
            # Fact patterns
            (re.compile(r"i (?:am|'m) (?:a |an )(.+)", re.I),
             "fact", lambda m: self._handle_fact(m, "am")),
            (re.compile(r"i (?:have|'ve|got) (?:a |an )?(.+)", re.I),
             "fact", lambda m: self._handle_fact(m, "have")),
            (re.compile(r"i (?:can|could) (.+)", re.I),
             "fact", lambda m: self._handle_fact(m, "can")),
            (re.compile(r"i (?:work|live|study) (?:at|in|as) (.+)", re.I),
             "fact", lambda m: self._handle_fact(m, "work at")),
            
            # Preference queries
            (re.compile(r"what (?:do i|did i|have i) (?:like|love|enjoy)", re.I),
             "query", lambda m: self._handle_query("likes")),
            (re.compile(r"what (?:do i|did i|have i) (?:hate|dislike)", re.I),
             "query", lambda m: self._handle_query("dislikes")),
            
            # User property queries
            (re.compile(r"what(?:'s| is) my name", re.I),
             "query", lambda m: self._handle_query("identity")),
            (re.compile(r"what(?:'s| is) my (\w+)", re.I),
             "query", lambda m: self._handle_user_property_query(m)),
            
            # Preference counting/listing
            (re.compile(r"(?:count|how many) (?:my |of my )?(?:preference|like|dislike)s?", re.I),
             "query", lambda m: self._handle_preference_count()),
            (re.compile(r"(?:list|show) (?:all )?(?:my |of my )?(?:preference|like|dislike)s?", re.I),
             "query", lambda m: self._handle_preference_list()),
            
            # Knowledge queries about user
            (re.compile(r"what do you (?:know|remember) about me", re.I),
             "query", lambda m: self._handle_query("all")),
            (re.compile(r"who am i", re.I),
             "query", lambda m: self._handle_query("identity")),
            (re.compile(r"do i like (.+)", re.I),
             "query", lambda m: self._handle_preference_check(m)),
            
            # Knowledge queries about concepts
            (re.compile(r"(?:tell me about|describe|explain) (?:a |an |the )?(.+)", re.I),
             "query", lambda m: self._handle_concept_query(m)),
            (re.compile(r"what do you know about (?:a |an |the )?(.+)", re.I),
             "query", lambda m: self._handle_concept_query(m)),
        ]
    
    def parse(self, text: str) -> ParseResult | None:
        """Parse a user statement.
        
        Args:
            text: User input text
            
        Returns:
            ParseResult if recognized, None otherwise
        """
        text = text.strip()
        
        for pattern, ptype, handler in self._patterns:
            match = pattern.match(text)
            if match:
                result = handler(match)
                if result:
                    result.type = ptype
                    result.original = text
                    return result
        
        return None
    
    def _handle_name(self, match: re.Match) -> ParseResult:
        """Handle name statement."""
        name = match.group(1).strip()
        response = self.profile.learn_name(name)
        return ParseResult(
            type="identity",
            data={"name": name, "response": response},
            handled=True,
        )
    
    def _handle_preference(self, match: re.Match, sentiment: str) -> ParseResult:
        """Handle preference statement."""
        item = match.group(1).strip()
        # Clean trailing punctuation
        item = item.rstrip(".,!?")
        
        response = self.profile.learn_preference(sentiment, item)
        return ParseResult(
            type="preference",
            data={"sentiment": sentiment, "item": item, "response": response},
            handled=True,
        )
    
    def _handle_fact(self, match: re.Match, predicate: str) -> ParseResult:
        """Handle fact statement."""
        value = match.group(1).strip().rstrip(".,!?")
        response = self.profile.learn_fact(predicate, value)
        return ParseResult(
            type="fact",
            data={"predicate": predicate, "value": value, "response": response},
            handled=True,
        )
    
    def _handle_query(self, query_type: str) -> ParseResult:
        """Handle query about user."""
        if query_type == "likes":
            prefs = self.profile.recall_preferences(sentiment="positive")
            if not prefs:
                response = "I don't know what you like yet. Tell me!"
            else:
                items = [p.item for p, _ in prefs[:5]]
                response = f"You like: {', '.join(items)}."
        
        elif query_type == "dislikes":
            prefs = self.profile.recall_preferences(sentiment="negative")
            if not prefs:
                response = "I don't know what you dislike yet."
            else:
                items = [p.item for p, _ in prefs[:5]]
                response = f"You dislike: {', '.join(items)}."
        
        elif query_type == "identity":
            if self.profile.user_name:
                response = f"You are {self.profile.user_name}."
            else:
                response = "I don't know your name yet. What should I call you?"
        
        else:  # "all"
            response = self.profile.get_summary()
        
        return ParseResult(
            type="query",
            data={"query_type": query_type, "response": response},
            handled=True,
        )
    
    def _handle_preference_check(self, match: re.Match) -> ParseResult:
        """Handle 'do I like X?' query."""
        item = match.group(1).strip().rstrip("?.,!")
        
        # Check preferences
        prefs = self.profile.recall_preferences(query=item)
        
        if not prefs:
            response = f"I don't know if you like {item}. Do you?"
        else:
            top_pref, score = prefs[0]
            if score > 0.5:
                if top_pref.sentiment == "positive":
                    response = f"Yes, you mentioned you like {top_pref.item}."
                else:
                    response = f"Actually, you said you don't like {top_pref.item}."
            else:
                response = f"I'm not sure. You haven't told me about {item}."
        
        return ParseResult(
            type="query",
            data={"query_type": "preference_check", "item": item, "response": response},
            handled=True,
        )
    
    def _handle_user_property_query(self, match: re.Match) -> ParseResult:
        """Handle 'what is my X?' query (age, job, etc.)."""
        prop = match.group(1).strip().rstrip("?.,!")
        
        # Search user facts for this property
        facts = self.profile.recall_facts(query=prop)
        
        if not facts:
            response = f"I don't know your {prop} yet. Tell me!"
        else:
            top_fact, score = facts[0]
            if score > 0.1:
                response = f"You {top_fact.predicate} {top_fact.value}."
            else:
                response = f"I'm not sure about your {prop}."
        
        return ParseResult(
            type="query",
            data={"query_type": "user_property", "property": prop, "response": response},
            handled=True,
        )
    
    def _handle_concept_query(self, match: re.Match) -> ParseResult:
        """Handle 'tell me about X' / 'what do you know about X?'."""
        concept = match.group(1).strip().rstrip("?.,!")
        
        return ParseResult(
            type="query",
            data={
                "query_type": "concept",
                "concept": concept,
                "concept_query": True,  # Signal to REPL to do memory search
            },
            handled=False,  # Let REPL handle the memory search
        )
    
    def _handle_preference_count(self) -> ParseResult:
        """Handle 'count my preferences'."""
        total = len(self.profile.likes) + len(self.profile.dislikes)
        likes_count = len(self.profile.likes)
        dislikes_count = len(self.profile.dislikes)
        
        if total == 0:
            response = "You haven't told me any preferences yet."
        else:
            parts = [f"You have {total} preference(s) total"]
            if likes_count:
                parts.append(f"{likes_count} like(s)")
            if dislikes_count:
                parts.append(f"{dislikes_count} dislike(s)")
            response = ": ".join(parts) + "."
        
        return ParseResult(
            type="query",
            data={"query_type": "preference_count", "response": response},
            handled=True,
        )
    
    def _handle_preference_list(self) -> ParseResult:
        """Handle 'list my preferences'."""
        lines = []
        
        if self.profile.likes:
            items = [p.item for p in self.profile.likes]
            lines.append(f"Likes: {', '.join(items)}")
        
        if self.profile.dislikes:
            items = [p.item for p in self.profile.dislikes]
            lines.append(f"Dislikes: {', '.join(items)}")
        
        if not lines:
            response = "You haven't told me any preferences yet."
        else:
            response = "\n".join(lines)
        
        return ParseResult(
            type="query",
            data={"query_type": "preference_list", "response": response},
            handled=True,
        )
