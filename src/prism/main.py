"""Conversational REPL for PRISM VSA.

Phase 3: Full conversational interface with:
- User identity and preferences
- Conversation context and topic tracking
- Natural language responses
- Pronoun resolution
"""

from __future__ import annotations

import atexit
import re
import sys

from prism.core import VSAConfig
from prism.core.lexicon import Lexicon, create_default_lexicon
from prism.memory import VectorMemory
from prism.memory.user_profile import UserProfile
from prism.memory.conversation_context import ConversationContext
from prism.memory.persistence import MemoryPersistence
from prism.parsing import SemanticParser
from prism.parsing.semantic_roles import SemanticRoleParser
from prism.parsing.user_statement_parser import UserStatementParser
from prism.reasoning import AnalogyReasoner
from prism.reasoning.composition import Composer
from prism.reasoning.transitive import TransitiveReasoner
from prism.reasoning.confidence import ConfidenceScorer, SourceType
from prism.reasoning.contradiction import ContradictionDetector
from prism.reasoning.temporal import TemporalReasoner
from prism.reasoning.advanced_queries import AdvancedQueryHandler
from prism.generation import ResponseGenerator
from prism.generation.explainer import Explainer
from prism.memory.optimizer import MemoryOptimizer
from prism.learning.feedback import FeedbackLearner
from prism.training.trainer import PRISMTrainer
from prism.agents.wikipedia_agent import WikipediaAgent
from prism.agents.orchestrator import SwarmOrchestrator
from prism.agents.blackboard import QueryType
from prism.agents.semantic_router import SemanticRouter, RouteType
from prism.agents.semantic_weaver import SemanticWeaver
from prism.agents.quality_gate import ResponseQualityGate
from prism.reasoning.pattern_library import PatternLibrary
from prism.reasoning.vsa_reasoner import VSAReasoner
from prism.reasoning.answer_generator import AnswerGenerator


MIN_SIMILARITY_SCORE = 0.1


class Prism:
    """Conversational PRISM VSA assistant."""
    
    def __init__(self, config: VSAConfig | None = None) -> None:
        """Initialize PRISM with all subsystems."""
        self.config = config or VSAConfig()
        self.lexicon = create_default_lexicon(self.config)
        
        # Memory systems
        self.memory = VectorMemory(self.lexicon, self.config)
        self.user_profile = UserProfile(self.lexicon)
        self.context = ConversationContext(self.lexicon)
        
        # Parsers
        self.parser = SemanticParser(self.lexicon)
        self.role_parser = SemanticRoleParser(self.lexicon, self.config)
        self.user_parser = UserStatementParser(self.lexicon, self.user_profile)
        
        # Reasoning
        self.analogy = AnalogyReasoner(self.lexicon, self.config)
        self.composer = Composer(self.lexicon, self.config)
        self.transitive = TransitiveReasoner(self.memory)
        self.scorer = ConfidenceScorer()
        self.contradiction = ContradictionDetector(self.memory, self.lexicon)
        self.temporal = TemporalReasoner(self.memory)
        self.advanced = AdvancedQueryHandler(self.memory)
        
        # Response generation
        self.response_gen = ResponseGenerator(self.user_profile, self.context)
        self.explainer = Explainer()
        
        # Optimization & Learning
        self.optimizer = MemoryOptimizer(self.memory)
        self.feedback = FeedbackLearner(self.memory)
        
        # Training
        self._trainer: PRISMTrainer | None = None
        
        # VSA Reasoning (lazy init)
        self._pattern_lib: PatternLibrary | None = None
        self._reasoner: VSAReasoner | None = None
        self._answer_gen: AnswerGenerator | None = None
        
        # Persistence
        self.persistence = MemoryPersistence()
        
        # Knowledge Swarm + Semantic Weaver
        self.wiki_agent = WikipediaAgent()
        nlp = getattr(self.lexicon, '_nlp', None)
        self.weaver = SemanticWeaver(nlp=nlp)
        self.swarm = SwarmOrchestrator(memory=self.memory, weaver=self.weaver)
        
        # Semantic Router + Quality Gate
        self.router = SemanticRouter(nlp) if nlp else None
        self.quality_gate = ResponseQualityGate()
    
    def process_input(self, text: str) -> str:
        """Process user input and generate response.
        
        This is the main entry point for conversation.
        """
        text = text.strip()
        if not text:
            return ""
        
        # Add to context
        self.context.add_utterance(text, "user")
        
        # Check for typos first — suggest corrections
        typo_suggestion = self._check_typos(text)
        if typo_suggestion:
            self.context.add_utterance(typo_suggestion, "prism")
            return typo_suggestion
        
        # Try user statement patterns first
        user_result = self.user_parser.parse(text)
        if user_result:
            if user_result.handled:
                response = user_result.data.get("response", "Got it.")
                self.context.add_utterance(response, "prism")
                return response
            
            # Concept query: "tell me about X" / "what do you know about X"
            if user_result.data.get("concept_query"):
                concept = user_result.data["concept"]
                response = self._query_concept(concept)
                self.context.add_utterance(response, "prism")
                return response
        
        # Try feedback (wrong/correct/clarify)
        feedback_result = self.feedback.try_handle(text)
        if feedback_result:
            self.context.add_utterance(feedback_result, "prism")
            return feedback_result
        
        # Try advanced queries (list all/count/compare)
        advanced_result = self.advanced.try_handle(text)
        if advanced_result:
            self.context.add_utterance(advanced_result, "prism")
            self.feedback.record_answer(advanced_result, [])
            return advanced_result
        
        # Try command-style input
        response = self._try_command(text)
        if response:
            self.context.add_utterance(response, "prism")
            self.feedback.record_answer(response, [])
            return response
        
        # Semantic Router — classify intent via embeddings (primary NLU)
        routed = self._route_and_respond(text)
        if routed:
            self.context.add_utterance(routed, "prism")
            self.feedback.record_answer(routed, [])
            return routed
        
        # Legacy: Try natural language understanding
        response = self._try_natural(text)
        
        # Quality gate: if response is garbage, try swarm
        if not self.quality_gate.is_acceptable(response):
            swarm_response = self._swarm_fallback(text)
            if swarm_response:
                self.context.add_utterance(swarm_response, "prism")
                self.feedback.record_answer(swarm_response, [])
                return swarm_response
        
        self.context.add_utterance(response, "prism")
        self.feedback.record_answer(response, [])
        return response
    
    def _query_concept(self, concept: str) -> str:
        """Query everything known about a concept."""
        concept = concept.strip().lower()
        lines = [f"About '{concept}':"]
        found_anything = False
        
        # Search using indexed lookup (precise, no substring noise)
        episodes = self.memory.search_facts(concept, top_k=10)
        
        if episodes:
            found_anything = True
            for ep in episodes[:8]:
                lines.append(f"  • {ep.text}")
        
        # Find similar concepts
        results = self.memory.find_similar(concept, k=5, min_score=MIN_SIMILARITY_SCORE)
        if results:
            found_anything = True
            lines.append(f"\nSimilar to '{concept}':")
            for w, score in results:
                lines.append(f"  • {w} [{score:.2f}]")
        
        # If nothing in local memory, use the Knowledge Swarm
        if not found_anything:
            bb = self.swarm.query(
                f"What is {concept}?",
                entities=[concept],
                query_type=QueryType.DEFINITION,
            )
            if bb.has_findings:
                lines = [f"About '{concept}':"]
                swarm_text = self.swarm.format_response(bb)
                if swarm_text:
                    lines.append(swarm_text)
                # Store learned facts for future
                for fact in bb.get_all_facts()[:5]:
                    self.memory.store_episode(fact)
                return "\n".join(lines)
            return f"I don't know much about '{concept}' yet. Teach me!"
        
        return "\n".join(lines)
    
    # ─── Semantic Routing ───────────────────────────────────────────
    
    def _route_and_respond(self, text: str) -> str | None:
        """Use SemanticRouter to classify and dispatch the query.
        
        Returns a response if routing succeeds, None to fall through
        to the legacy pipeline.
        """
        if not self.router:
            return None
        
        result = self.router.classify(text)
        self._think(f"🧭 Route: {result.route.value} ({result.confidence:.0%})")
        
        # High-confidence routing
        if result.route == RouteType.GREETING:
            name = self.user_profile.user_name if self.user_profile.user_name else 'there'
            return f"Hey {name}! What would you like to explore?"
        
        if result.route == RouteType.CASUAL:
            return "😊 Anything else you'd like to know?"
        
        if result.route == RouteType.META_CHAT:
            return self._handle_meta_chat(text)
        
        if result.route == RouteType.TEACH_FACT:
            return None  # Let the legacy parser handle fact learning
        
        if result.route in (RouteType.KNOWLEDGE_QUERY, RouteType.UNKNOWN):
            return self._route_knowledge_query(text)
        
        if result.route == RouteType.YES_NO:
            return self._route_yes_no(text, QueryType.YES_NO)
        
        if result.route == RouteType.COMPARISON:
            return self._route_comparison(text)
        
        return None
    
    def _handle_meta_chat(self, text: str) -> str:
        """Handle meta/identity questions about PRISM itself."""
        t = text.lower().strip().rstrip('?!.')
        
        if 'old' in t or 'age' in t or 'born' in t:
            return (
                "I don't have an age — I exist as patterns in vector space! "
                "But I learn and grow every time you teach me something new."
            )
        if 'name' in t or 'who are you' in t or 'what are you' in t:
            return (
                "I'm PRISM 🧠 — a local AI that learns and reasons using "
                "Vector Symbolic Architecture. No cloud, no LLMs — just math, "
                "vectors, and a swarm of parallel knowledge agents."
            )
        if 'can you' in t or 'capable' in t or 'what can' in t:
            return (
                "I can learn facts, answer questions, compare concepts, "
                "solve analogies, search Wikipedia/Wikidata/WordNet in parallel, "
                "and reason over everything I know. Try 'help' for all commands!"
            )
        if 'how are you' in t or 'how do you' in t:
            return "I'm running smoothly! Ready to learn and explore. What's on your mind?"
        if 'feel' in t or 'emotion' in t or 'conscious' in t:
            return (
                "I don't experience feelings — I'm patterns of vectors and cosine similarity. "
                "But I'm designed to be helpful and curious! 🧠"
            )
        return (
            "I'm PRISM — a local-first AI assistant. "
            "Ask me anything or type 'help' for commands!"
        )
    
    def _route_knowledge_query(self, text: str) -> str:
        """Route a knowledge query: try local memory, then swarm.
        
        Always returns a definitive response — never None.
        """
        # Try local concept lookup first
        words = [w for w in text.lower().split() if len(w) > 3 and w not in self._skip_words()]
        if words:
            key_concept = max(words, key=len)
            episodes = self.memory.search_facts(key_concept, top_k=10)
            if episodes and len(episodes) >= 3:
                lines = [f"About {key_concept}:"]
                for ep in episodes[:6]:
                    lines.append(f"  • {ep.text}")
                local_response = "\n".join(lines)
                if self.quality_gate.is_acceptable(local_response):
                    return local_response
        
        # Local not good enough — bring in the swarm
        swarm_answer = self._swarm_fallback(text, query_type=QueryType.DEFINITION)
        if swarm_answer:
            return swarm_answer
        
        return f"I don't have information about that yet. Try 'swarm {text}' for a deep search."
    
    def _route_yes_no(self, text: str, query_type: QueryType = QueryType.YES_NO) -> str:
        """Route a yes/no question: try local, then swarm.
        
        Always returns a definitive response.
        """
        # Try the existing yes/no handler first
        yn = self._parse_yes_no(text.lower().strip().rstrip('?').strip())
        if yn:
            subj, relation, pred = yn
            score = self.memory.check_fact(subj, relation, pred)
            if score > 0.1:
                scored = self.scorer.score_direct(
                    f"{subj} {relation.lower()} {pred}", score
                )
                return f"Yes, {scored.format()}"
            
            # Try transitive
            chain = self.transitive.infer(subj, pred)
            if chain and chain.confidence >= 0.4:
                scored = self.scorer.score_transitive(
                    f"{subj} → {pred}",
                    chain.confidence,
                    [f"{s.subject} {s.relation} {s.obj}" for s in chain.steps],
                )
                lines = [f"Yes (inferred {scored.format()}):"]
                for step in chain.steps:
                    lines.append(f"  • {step.subject} {step.relation} {step.obj} [{step.confidence:.2f}]")
                return "\n".join(lines)
        
        # No local answer — swarm it
        swarm_answer = self._swarm_fallback(text, query_type=query_type)
        if swarm_answer:
            return swarm_answer
        return "I couldn't find a clear answer to that question."
    
    def _route_comparison(self, text: str) -> str:
        """Route a comparison query to the swarm."""
        swarm_answer = self._swarm_fallback(text, query_type=QueryType.COMPARISON)
        if swarm_answer:
            return swarm_answer
        return "I don't have enough information to compare those yet."
    
    # ─── Thinking Indicators ────────────────────────────────────────
    
    def _think(self, msg: str) -> None:
        """Print a thinking step indicator (real-time)."""
        import sys
        print(f"  {msg}", flush=True)
    
    # ─── Meta Questions (Identity / Casual) ─────────────────────────
    
    def _handle_meta(self, text: str) -> str | None:
        """Handle meta-questions, greetings, and casual remarks."""
        t = text.lower().strip().rstrip('?!.')
        
        # Identity questions
        if t in ('who are you', 'what are you', 'whats your name',
                 "what's your name", 'tell me about yourself'):
            return (
                "I'm PRISM 🧠 — a local AI that learns and reasons using "
                "Vector Symbolic Architecture. No cloud, no LLMs — just math, "
                "vectors, and a swarm of parallel knowledge agents. "
                "Teach me facts, ask me questions, or try 'swarm <query>' "
                "for a deep search!"
            )
        
        if t in ('and who are you', 'and you'):
            return "I'm PRISM! Ask me anything or type 'help' for commands."
        
        # Gratitude
        if t in ('thanks', 'thank you', 'thx', 'ty', 'danke'):
            return "You're welcome! 😊"
        
        # Casual acknowledgments (prevent garbage similarity search)
        if t in ('interesting', 'cool', 'nice', 'okay', 'ok', 'got it',
                 'i see', 'wow', 'great', 'awesome', 'sure', 'alright',
                 'hm', 'hmm', 'huh', 'right', 'yep', 'yeah', 'ja', 'gut',
                 'super', 'toll', 'genau'):
            return "😊 Anything else you'd like to know?"
        
        # Greetings
        if t in ('hi', 'hello', 'hey', 'hallo', 'moin', 'servus',
                 'good morning', 'good evening', 'good afternoon'):
            name = self.user_profile.name if hasattr(self, 'user_profile') and self.user_profile.name else 'there'
            return f"Hey {name}! What would you like to explore?"
        
        # Goodbye
        if t in ('bye', 'goodbye', 'see you', 'ciao', 'tschüss', 'exit', 'quit'):
            return None  # Let the main loop handle exit
        
        return None
    
    # ─── Smart Swarm Fallback ────────────────────────────────────────
    
    def _swarm_fallback(self, query: str, query_type: QueryType | None = None) -> str | None:
        """Auto-trigger swarm when local knowledge is insufficient.
        
        Args:
            query: The user's question
            query_type: Override the TaskPlanner's classification with
                        the SemanticRouter's classification (prevents double-classify).
        
        Returns a formatted response, or None if swarm also found nothing.
        """
        self._think("🔍 Local knowledge insufficient...")
        self._think("🐝 Dispatching Knowledge Swarm...")
        
        if query_type:
            # Use the router's classification directly (avoid re-classify)
            plan = self.swarm.planner.plan(query)
            plan.query_type = query_type  # Override TaskPlanner's classification
            entities = plan.entities
            if plan.sub_queries:
                seen = set(e.lower() for e in entities)
                for sq in plan.sub_queries:
                    if sq.lower() not in seen:
                        entities.append(sq)
                        seen.add(sq.lower())
            bb = self.swarm.query(query, entities=entities, query_type=query_type)
        else:
            bb, plan = self.swarm.query_smart(query)
        
        if not bb.has_findings:
            return None
        
        # Format with CortexComposer
        response = self.swarm.format_smart_response(bb, plan)
        
        # Auto-learn discovered facts
        learned = 0
        for fact in bb.get_all_facts()[:5]:
            self.memory.store_episode(fact)
            learned += 1
        
        lines = []
        if response:
            lines.append(response)
        
        if learned > 0:
            lines.append(f"\n  💾 Learned {learned} new facts from this query")
        
        # Show timing
        summary_line = bb.summary().split(chr(10))[0]
        lines.append(f"  ⏱️  {summary_line}")
        
        return "\n".join(lines) if lines else None
    
    def _query_swarm(self, query: str) -> str:
        """Query the Knowledge Swarm directly.
        
        Uses TaskPlanner to auto-classify intent, extract entities,
        and decompose complex questions before dispatching agents.
        """
        self._think("🧠 Planning query...")
        bb, plan = self.swarm.query_smart(query)
        self._think(f"📋 Intent: {plan.query_type.value} | Entities: {', '.join(plan.entities[:3])}")
        self._think("🐝 Dispatching agents...")
        
        if not bb.has_findings:
            return f"The swarm couldn't find anything about '{query}'."
        
        lines = [f"🐝 Knowledge Swarm results for '{query}':"]
        lines.append(f"  📋 Intent: {plan.query_type.value} | Entities: {', '.join(plan.entities[:3])}")
        
        if plan.sub_queries:
            lines.append(f"  🔍 Sub-queries: {', '.join(plan.sub_queries[:3])}")
        
        lines.append("")
        
        # Use smart response composer
        response = self.swarm.format_smart_response(bb, plan)
        if response:
            lines.append(response)
        
        # Store learned facts
        for fact in bb.get_all_facts()[:5]:
            self.memory.store_episode(fact)
        
        # Show timing
        lines.append(f"\n  {bb.summary().split(chr(10))[0]}")
        
        return "\n".join(lines)
    
    def _try_command(self, text: str) -> str | None:
        """Try parsing as a command."""
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        
        if cmd == "learn" and arg:
            return self.learn(arg)
        elif cmd == "event" and arg:
            return self.learn_event(arg)
        elif cmd == "query" and arg:
            return self.query(arg)
        elif cmd == "similar" and arg:
            return self.find_similar(arg)
        elif cmd == "analogy":
            words = arg.split()
            if len(words) == 3:
                return self.solve_analogy(words[0], words[1], words[2])
        elif cmd == "compose" and arg:
            return self.compose_concepts(arg.split())
        elif cmd == "check" and arg:
            return self.check(arg)
        elif cmd == "facts":
            return self.list_facts()
        elif cmd == "about" and arg.lower() == "me":
            return self.user_profile.get_summary()
        elif cmd == "context":
            return self.context.get_context_summary()
        elif cmd == "status":
            return self.status()
        elif cmd == "save":
            return self.save_memory(arg if arg else None)
        elif cmd == "load" and arg:
            return self.load_memory(arg)
        elif cmd == "stats":
            return self.show_stats(arg if arg else None)
        elif cmd == "conflicts":
            return self.show_conflicts()
        elif cmd == "when" and arg:
            return self.temporal.when_learned(arg)
        elif cmd == "why":
            return self.explainer.explain()
        elif cmd == "cleanup":
            removed = self.optimizer.cleanup()
            return f"✓ Removed {removed} low-priority facts."
        elif cmd == "usage":
            return self.optimizer.format_usage()
        elif cmd == "export" and arg:
            return f"✓ Exported to {self.optimizer.export_json(arg)}"
        elif cmd == "feedback":
            return self.feedback.get_feedback_summary()
        elif cmd == "gaps":
            return self.feedback.suggest_knowledge_gaps()
        elif cmd == "train" and arg:
            return self._train_file(arg)
        elif cmd == "train-wiki" and arg:
            parts = arg.split()
            topic = parts[0]
            n = 3
            if "--articles" in parts:
                idx = parts.index("--articles")
                if idx + 1 < len(parts):
                    try:
                        n = int(parts[idx + 1])
                    except ValueError:
                        pass
            return self._train_wiki(topic, n)
        elif cmd == "train-url" and arg:
            return self._train_url(arg)
        elif cmd == "swarm" and arg:
            return self._query_swarm(arg)
        elif cmd == "help":
            return self._show_help()
        
        return None
    
    def _show_help(self) -> str:
        """Show available commands."""
        return (
            "🧠 Cortex Commands:\n"
            "  learn <fact>       — Teach me a fact (e.g. 'learn cats are mammals')\n"
            "  facts              — List all stored facts\n"
            "  about <concept>    — Query a concept from memory\n"
            "  swarm <question>   — Query the Knowledge Swarm (parallel agents)\n"
            "  check <fact>       — Check if a fact is true\n"
            "  compare <a> <b>    — Compare two concepts\n"
            "  analogy <a:b::c:?> — Solve an analogy\n"
            "  train <file>       — Train from a text file\n"
            "  train-wiki <topic> — Train from Wikipedia\n"
            "  stats              — Show memory statistics\n"
            "  save [path]        — Save memory to disk\n"
            "  load <path>        — Load memory from disk\n"
            "  help               — Show this help\n"
            "\n  Or just ask me anything in natural language!"
        )
    
    def _try_natural(self, text: str) -> str:
        """Try natural language understanding."""
        text_lower = text.lower()
        
        # Resolve pronouns
        resolved = self.context.resolve_text(text)
        
        # Questions about facts
        if "?" in text or text_lower.startswith(("what", "who", "is", "are", "does", "do", "can", "how")):
            return self._handle_question(resolved)
        
        # "Tell me about X" (in case user parser didn't catch it)
        tell_match = re.match(r"(?:tell me about|describe|explain) (?:a |an |the )?(.+)", text_lower)
        if tell_match:
            return self._query_concept(tell_match.group(1).strip().rstrip(".,!?"))
        
        # Statements that look like facts
        facts = self.parser.parse(resolved)
        if facts:
            for fact in facts:
                self.memory.store(fact.subject, fact.relation, fact.object)
            return self.response_gen.learned_fact(
                f"{facts[0].subject} {facts[0].relation} {facts[0].object}"
            )
        
        # Events
        event = self.role_parser.parse(resolved)
        if event:
            self.memory.store_event(event.agent, event.action, event.patient)
            return self.response_gen.learned_fact(event.summary())
        
        # Fallback: try the Knowledge Swarm instead of weak similarity matches
        swarm_answer = self._swarm_fallback(text)
        if swarm_answer:
            return swarm_answer
        
        return self.response_gen.clarify()
    
    def _skip_words(self) -> set[str]:
        """Words to skip in fallback search."""
        return {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "and", "or", "but", "if", "then", "else", "when", "where",
            "what", "which", "who", "whom", "whose", "how", "why",
            "this", "that", "these", "those", "it", "its",
            "about", "tell", "know", "think", "like", "also",
        }
    
    # Common words that should never be flagged as typos
    COMMON_WORDS = {
        'cure', 'sick', 'help', 'ill', 'pain', 'heal', 'hurt', 'ache',
        'fix', 'work', 'run', 'play', 'eat', 'drink', 'sleep', 'walk',
        'talk', 'read', 'write', 'swim', 'fly', 'jump', 'climb', 'fall',
        'grow', 'die', 'live', 'love', 'hate', 'fear', 'hope', 'wish',
        'feel', 'hear', 'see', 'smell', 'taste', 'touch',
        'hot', 'cold', 'warm', 'cool', 'wet', 'dry',
        'big', 'small', 'fast', 'slow', 'old', 'new', 'young',
        'good', 'bad', 'nice', 'fine', 'well', 'okay',
        'red', 'blue', 'green', 'white', 'black', 'brown', 'pink',
        'yes', 'no', 'maybe', 'always', 'never', 'often',
        'here', 'now', 'soon', 'later', 'today', 'yesterday', 'tomorrow',
        'more', 'less', 'much', 'many', 'few', 'some', 'all', 'any',
        'very', 'too', 'quite', 'really', 'just', 'only', 'also',
        'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
        'man', 'men', 'woman', 'women', 'boy', 'girl', 'baby', 'child',
        'car', 'bus', 'train', 'plane', 'ship', 'boat', 'bike',
        'door', 'wall', 'floor', 'roof', 'room', 'home', 'house',
        'food', 'meal', 'rice', 'bread', 'meat', 'fish', 'egg', 'milk',
        'tea', 'coffee', 'wine', 'beer', 'juice', 'soup',
        'eye', 'ear', 'nose', 'mouth', 'hand', 'foot', 'arm', 'leg',
        'head', 'face', 'hair', 'skin', 'bone', 'blood', 'heart', 'brain',
        'sun', 'moon', 'star', 'sky', 'rain', 'snow', 'wind', 'fire',
        'sea', 'lake', 'river', 'hill', 'rock', 'sand', 'mud', 'ice',
        'tree', 'grass', 'leaf', 'root', 'seed', 'fruit', 'flower',
        'dog', 'cat', 'cow', 'pig', 'hen', 'bee', 'ant', 'bat', 'rat',
        'king', 'god', 'war', 'law', 'tax', 'art', 'map', 'job',
        'day', 'year', 'week', 'hour', 'time', 'age', 'era',
        'purr', 'bark', 'meow', 'roar', 'hiss', 'buzz', 'chirp',
    }
    
    def _check_typos(self, text: str) -> str | None:
        """Check for typos using spaCy vocab + edit distance.
        
        A word is only flagged if it's:
        1. Not in our lexicon
        2. Not in spaCy's vocabulary (300k+ words)
        3. Not in the common words whitelist
        4. Has edit distance ≤ 2 to a known lexicon word
        """
        words = text.lower().split()
        known = set(self.lexicon._vectors.keys())
        skip = self._skip_words() | {"i", "my", "me", "you", "we", "he", "she"}
        
        # Get spaCy vocab for broader coverage
        spacy_vocab = self.lexicon._nlp.vocab if hasattr(self.lexicon, '_nlp') else None
        
        for word in words:
            clean = word.strip(".,!?;:'\"")
            if not clean or len(clean) < 3 or clean in known or clean in skip:
                continue
            
            # Check common words whitelist
            if clean in self.COMMON_WORDS:
                continue
            
            # Check spaCy vocab (covers 300k+ English words)
            if spacy_vocab is not None and spacy_vocab.has_vector(clean):
                continue
            
            # Also check if spaCy recognizes it at all
            if spacy_vocab is not None and clean in spacy_vocab:
                continue
            
            # Word is genuinely unknown — find closest by edit distance
            best_match = None
            best_dist = 99
            for kw in known:
                if abs(len(kw) - len(clean)) > 2:
                    continue
                dist = self._edit_distance(clean, kw)
                if dist < best_dist and dist <= 2:
                    best_dist = dist
                    best_match = kw
            
            if best_match and best_dist <= 2 and best_match != clean:
                return f"Did you mean '{best_match}'? ('{clean}' doesn't look right)"
        
        return None
    
    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return PRISM._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        prev_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                cost = 0 if c1 == c2 else 1
                curr_row.append(min(
                    curr_row[j] + 1,     # insert
                    prev_row[j + 1] + 1, # delete
                    prev_row[j] + cost,  # replace
                ))
            prev_row = curr_row
        
        return prev_row[-1]
    
    def _parse_yes_no(self, text: str) -> tuple[str, str, str] | None:
        """Parse yes/no question into (subject, relation, predicate).
        
        Patterns:
          "do cats purr"   → ("cats", "DOES", "purr")
          "can dogs fly"   → ("dogs", "CAN", "fly")
          "are cats mammals" → ("cats", "IS", "mammals")
          "is water wet"   → ("water", "IS", "wet")
          "do fish swim"   → ("fish", "DOES", "swim")
        
        Skips complex questions that should go to the pattern reasoner.
        """
        import re
        
        # Skip complex questions that need pattern-based reasoning
        complex_markers = [
            'same as', 'similar', 'different', 'differ', 'compare',
            'in common', 'related', 'versus', 'vs',
        ]
        if any(m in text for m in complex_markers):
            return None
        
        # "Do/Does X Y?"
        m = re.match(r'^(?:do|does)\s+(\w+(?:\s+\w+)?)\s+(.+)$', text)
        if m:
            return (m.group(1).strip(), 'DOES', m.group(2).strip())
        
        # "Can/Could X Y?"
        m = re.match(r'^(?:can|could)\s+(\w+(?:\s+\w+)?)\s+(.+)$', text)
        if m:
            return (m.group(1).strip(), 'CAN', m.group(2).strip())
        
        # "Is/Are X Y?" or "Is/Are X a Y?"
        m = re.match(r'^(?:is|are)\s+(\w+(?:\s+\w+)?)\s+(?:a\s+|an\s+)?(.+)$', text)
        if m:
            return (m.group(1).strip(), 'IS', m.group(2).strip())
        
        # "Has/Have X Y?"
        m = re.match(r'^(?:has|have)\s+(\w+(?:\s+\w+)?)\s+(.+)$', text)
        if m:
            return (m.group(1).strip(), 'HAS', m.group(2).strip())
        
        return None
    
    def _handle_question(self, text: str) -> str:
        """Handle a question."""
        text_lower = text.lower().strip().rstrip('?').strip()
        
        # ── Yes/no question patterns ──
        # "Do cats purr?" → search for cats + purr
        # "Can dogs fly?" → search for dogs CAN fly
        # "Are cats mammals?" → search for cats IS mammals
        yn = self._parse_yes_no(text_lower)
        if yn:
            subj, relation, pred = yn
            # Try direct fact lookup
            score = self.memory.check_fact(subj, relation, pred)
            if score > 0.1:
                scored = self.scorer.score_direct(
                    f"{subj} {relation.lower()} {pred}", score
                )
                return f"Yes, {scored.format()}"
            
            # Try searching stored facts (indexed, precise)
            episodes = self.memory.search_facts(subj, top_k=10)
            matching = [
                ep.text for ep in episodes
                if pred in ep.obj.lower() or pred in ep.text.lower()
            ]
            if matching:
                lines = [f"Yes! Found:"]
                for f in matching[:5]:
                    lines.append(f"  • {f}")
                return "\n".join(lines)
            
            # Try transitive
            chain = self.transitive.infer(subj, pred)
            if chain and chain.confidence >= 0.4:
                scored = self.scorer.score_transitive(
                    f"{subj} → {pred}",
                    chain.confidence,
                    [f"{s.subject} {s.relation} {s.obj}" for s in chain.steps],
                )
                lines = [f"Yes (inferred {scored.format()}):"]
                for step in chain.steps:
                    lines.append(f"  • {step.subject} {step.relation} {step.obj} [{step.confidence:.2f}]")
                return "\n".join(lines)
            
            # Try swarm fallback for unknown yes/no questions
            swarm_answer = self._swarm_fallback(f"{subj} {relation.lower()} {pred}?")
            if swarm_answer:
                return swarm_answer
            
            return f"I don't know if {subj} {relation.lower()} {pred}."
        
        # ── VSA pattern-based reasoning (priority over encode_question) ──
        natural_answer = self._handle_natural_question(text)
        if natural_answer:
            return natural_answer
        
        subject, relation, obj = self.parser.encode_question(text)
        
        if subject and relation:
            if obj:
                # Yes/no question — check direct fact first
                score = self.memory.check_fact(subject, relation, obj)
                if score > 0.1:
                    scored = self.scorer.score_direct(
                        f"{subject} {relation.lower()} {obj}", score
                    )
                    answer = f"Yes, {scored.format()}"
                    self.explainer.record_direct(
                        text, answer, f"{subject} {relation} {obj}", scored.confidence
                    )
                    return answer
                
                # Try transitive inference
                chain = self.transitive.infer(subject, obj)
                if chain and chain.confidence >= 0.4:
                    scored = self.scorer.score_transitive(
                        f"{subject} → {obj}",
                        chain.confidence,
                        [f"{s.subject} {s.relation} {s.obj}" for s in chain.steps],
                    )
                    lines = [f"Yes (inferred {scored.format()}):"]
                    for step in chain.steps:
                        lines.append(f"  • {step.subject} {step.relation} {step.obj} [{step.confidence:.2f}]")
                    answer = "\n".join(lines)
                    self.explainer.record_transitive(
                        text, answer,
                        [f"{s.subject} {s.relation} {s.obj}" for s in chain.steps],
                        chain.confidence,
                    )
                    return answer
                
                return f"I'm not sure if {subject} {relation.lower()} {obj}."
            else:
                # First check stored facts for this subject (indexed)
                episodes = self.memory.search_facts(subject, top_k=10)
                if episodes:
                    lines = [f"About {subject}:"]
                    for ep in episodes[:6]:
                        lines.append(f"  • {ep.text}")
                    return "\n".join(lines)
                
                # Open question - try vector query
                results = self.memory.query_object(subject, relation, top_k=3)
                all_results = [(w, s, relation) for w, s in results if s >= MIN_SIMILARITY_SCORE]
                
                if all_results:
                    all_results.sort(key=lambda x: x[1], reverse=True)
                    lines = [f"About {subject}:"]
                    for word, score, rel in all_results[:5]:
                        lines.append(f"  • {rel.lower()} {word} [{score:.2f}]")
                    return "\n".join(lines)
        
        # Fallback: try the Knowledge Swarm
        swarm_answer = self._swarm_fallback(text)
        if swarm_answer:
            return swarm_answer
        
        return self.response_gen.no_answer()
    
    def _handle_natural_question(self, text: str) -> str | None:
        """Handle a natural question using VSA pattern matching.
        
        Uses PatternLibrary to identify the question type, VSAReasoner
        to search facts and compute answers, and AnswerGenerator to
        format the response.
        
        Returns None if no pattern matches (falls through to fallback).
        """
        # Lazy init
        if self._pattern_lib is None:
            self._pattern_lib = PatternLibrary(self.lexicon)
        if self._reasoner is None:
            self._reasoner = VSAReasoner(self)
        if self._answer_gen is None:
            self._answer_gen = AnswerGenerator()
        
        # Phase 16: Check for analogy patterns first
        import re
        analogy_m = re.search(
            r'(\w+)\s+(?:is|are)\s+to\s+(\w+)\s+as\s+(\w+)\s+(?:is|are)\s+to\s+\?',
            text.lower(),
        )
        if analogy_m:
            a, b, c = analogy_m.group(1), analogy_m.group(2), analogy_m.group(3)
            result = self._reasoner.solve_analogy(a, b, c)
            if result.confidence > 0.0:
                return self._answer_gen.generate_analogy_answer(a, b, c, result)
        
        # Match question to pattern
        match = self._pattern_lib.match_pattern(text)
        if match is None:
            return None
        
        # Execute reasoning
        result = self._reasoner.execute_pattern(
            match.pattern.pattern_type,
            match.entities,
        )
        
        # Skip if no useful answer was produced
        if result.confidence <= 0.0 and not result.facts_used:
            return None
        
        # Generate answer
        return self._answer_gen.generate_answer(result)

    
    # =========================================================================
    # Command implementations
    # =========================================================================
    
    def learn(self, text: str) -> str:
        """Learn a fact from natural language."""
        facts = self.parser.parse(text)
        if not facts:
            return f"I couldn't understand that as a fact."
        
        responses = []
        for fact in facts:
            # Check for contradictions before storing
            conflicts = self.contradiction.check(fact.subject, fact.relation, fact.object)
            
            # Store the fact regardless
            self.memory.store(fact.subject, fact.relation, fact.object)
            
            if conflicts:
                lines = [f"⚠ Conflict detected!"]
                if len(conflicts) > 1:
                    lines[0] = f"⚠ {len(conflicts)} conflict(s) detected!"
                for c in conflicts:
                    lines.append(f"  Existing: {c.existing_fact}")
                    lines.append(f"  New: {c.new_fact}")
                    lines.append(f"  Reason: {c.reason}")
                lines.append("  (Both stored — use 'conflicts' to review)")
                responses.append("\n".join(lines))
            else:
                responses.append(f"✓ {fact.subject} {fact.relation} {fact.object}")
        return "\n".join(responses)
    
    def learn_event(self, text: str) -> str:
        """Learn an event with semantic roles."""
        event = self.role_parser.parse(text)
        if not event:
            return f"I couldn't parse that as an event."
        
        self.memory.store_event(event.agent, event.action, event.patient)
        return f"✓ {event.summary()}"
    
    def query(self, text: str) -> str:
        """Answer a question."""
        subject, relation, obj = self.parser.encode_question(text)
        
        if not subject and not relation:
            return f"I couldn't understand: '{text}'"
        
        if obj:
            score = self.memory.check_fact(subject, relation, obj)
            verdict = "likely" if score > 0.1 else "unlikely"
            return f"{subject} {relation} {obj}: {verdict} [{score:.3f}]"
        
        results = self.memory.query_object(subject, relation, top_k=5)
        results = [(w, s) for w, s in results if s >= MIN_SIMILARITY_SCORE]
        
        if not results:
            return f"No answer for: {subject} {relation} ?"
        
        lines = [f"{subject} {relation}:"]
        for word, score in results[:5]:
            lines.append(f"  • {word} [{score:.3f}]")
        return "\n".join(lines)
    
    def query_who(self, action: str) -> str:
        """Who did the action?"""
        results = self.memory.query_agent(action, top_k=5)
        if not results or results[0][1] < MIN_SIMILARITY_SCORE:
            return f"I don't know who {action}."
        
        return f"{results[0][0].title()} {action}."
    
    def query_what(self, agent: str) -> str:
        """What did agent do?"""
        results = self.memory.query_action(agent, top_k=5)
        if not results:
            return f"I don't know what {agent} did."
        
        return f"{agent.title()} {results[0][0]}."
    
    def check(self, text: str) -> str:
        """Check if a fact is true."""
        facts = self.parser.parse(text)
        if not facts:
            return f"I couldn't understand: '{text}'"
        
        fact = facts[0]
        score = self.memory.check_fact(fact.subject, fact.relation, fact.object)
        if score > 0.1:
            return f"✓ Yes, {fact.subject} {fact.relation} {fact.object}."
        else:
            return f"✗ I'm not sure about that."
    
    def find_similar(self, word: str, k: int = 5) -> str:
        """Find similar concepts."""
        results = self.memory.find_similar(word, k=k, min_score=0.0)
        return self.response_gen.format_similar(word, results)
    
    def solve_analogy(self, a: str, b: str, c: str) -> str:
        """Solve analogy: a is to b as c is to ?"""
        results = self.analogy.solve(a, b, c, top_k=5)
        return self.response_gen.format_analogy(a, b, c, results)
    
    def compose_concepts(self, words: list[str]) -> str:
        """Compose multiple concepts."""
        if not words:
            return "No concepts provided."
        
        composed = self.composer.compose(words)
        components = self.composer.decompose(
            composed, k=5, exclude=set(w.lower() for w in words)
        )
        
        lines = [f"Composition of {' + '.join(words)}:"]
        for word, score in components[:5]:
            lines.append(f"  • {word} [{score:.3f}]")
        return "\n".join(lines)
    
    def list_facts(self) -> str:
        """List all stored facts."""
        facts = self.memory.get_all_facts()
        if not facts:
            return "No facts stored yet."
        return f"Facts ({len(facts)}):\n" + "\n".join(f"  • {f}" for f in facts[:20])
    
    def status(self) -> str:
        """Show system status."""
        lines = [
            "=== PRISM Status ===",
            f"Dimension: {self.config.dimension}",
            f"Lexicon: {len(self.lexicon)} words",
            f"Embeddings: {'✓' if self.lexicon.has_embeddings() else '✗'}",
            f"Episodes: {len(self.memory)}",
        ]
        if self.user_profile.is_known():
            lines.append(f"User: {self.user_profile.user_name or 'Known'}")
        if self.context.current_topic:
            lines.append(f"Topic: {self.context.current_topic}")
        return "\n".join(lines)
    
    def show_conflicts(self) -> str:
        """Show all detected contradictions."""
        conflicts = self.contradiction.get_all_conflicts()
        if not conflicts:
            return "No contradictions found."
        
        lines = [f"⚠ Found {len(conflicts)} contradiction(s):"]
        for i, (fact1, fact2) in enumerate(conflicts, 1):
            lines.append(f"  {i}. '{fact1}' vs '{fact2}'")
        return "\n".join(lines)
    
    def _get_trainer(self) -> PRISMTrainer:
        """Lazy-init the trainer."""
        if self._trainer is None:
            self._trainer = PRISMTrainer(self)
        return self._trainer
    
    def _train_file(self, filepath: str) -> str:
        """Train from a text file."""
        trainer = self._get_trainer()
        stats = trainer.train_from_file(filepath, show_progress=False)
        return stats.summary()
    
    def _train_wiki(self, topic: str, max_articles: int = 3) -> str:
        """Train from Wikipedia."""
        trainer = self._get_trainer()
        stats = trainer.train_from_wikipedia(topic, max_articles, show_progress=False)
        return stats.summary()
    
    def _train_url(self, url: str) -> str:
        """Train from a URL."""
        trainer = self._get_trainer()
        stats = trainer.train_from_url(url, show_progress=False)
        return stats.summary()
    
    def save_memory(self, path: str | None = None) -> str:
        """Save memory to disk."""
        try:
            saved_path = self.persistence.save(self, path)
            stats = self.persistence.get_stats(saved_path)
            return (f"✓ Memory saved to '{saved_path}' "
                    f"({stats.get('episodes', 0)} facts, "
                    f"{stats.get('file_size_kb', 0)} KB)")
        except Exception as e:
            return f"✗ Save failed: {e}"
    
    def load_memory(self, path: str) -> str:
        """Load memory from disk."""
        try:
            if self.persistence.load(self, path):
                return (f"✓ Memory loaded from '{path}' "
                        f"({len(self.memory)} facts, "
                        f"user: {self.user_profile.user_name or 'unknown'})")
            else:
                return f"✗ No save file found at '{path}'"
        except Exception as e:
            return f"✗ Load failed: {e}"
    
    def show_stats(self, path: str | None = None) -> str:
        """Show stats about a save file."""
        stats = self.persistence.get_stats(path)
        if not stats:
            return "No save file found."
        if "error" in stats:
            return f"✗ {stats['error']}"
        
        lines = [
            "=== Save File Stats ===",
            f"Version: {stats.get('version', '?')}",
            f"Saved: {stats.get('saved_at', '?')}",
            f"Size: {stats.get('file_size_kb', 0)} KB",
            f"Vocabulary: {stats.get('vocabulary', 0)} words",
            f"Episodes: {stats.get('episodes', 0)}",
            f"User: {stats.get('user_name') or 'none'}",
            f"Preferences: {stats.get('preferences', 0)}",
            f"User facts: {stats.get('facts', 0)}",
        ]
        return "\n".join(lines)


def print_help() -> None:
    """Print help message."""
    print("""
╔════════════════════════════════════════════════════════════════╗
║  GUNTER - Conversational VSA Assistant (Phase 3)              ║
╚════════════════════════════════════════════════════════════════╝

CONVERSATION:
  Just type naturally! PRISM understands:
  - "My name is X"     → Remembers your name
  - "I like X"         → Remembers your preferences  
  - "What do I like?"  → Recalls your preferences
  - "Tell me about X"  → Searches knowledge
  - Facts and events   → Learns automatically

COMMANDS:
  learn <fact>         Store a fact
  event <sentence>     Store event with roles
  query <question>     Ask a question
  similar <word>       Find similar concepts
  analogy <a> <b> <c>  Solve analogy
  about me             Show what I know about you
  context              Show conversation context
  status               Show system status
  help                 Show this help
  quit                 Exit

EXAMPLES:
  > My name is Alex
  > I like coding and coffee
  > learn a cat is an animal
  > What do I like?
  > similar cat
""")


def run_demo(prism: Prism) -> None:
    """Run conversational demo."""
    print("\n" + "="*60)
    print("  GUNTER - Conversational Demo")
    print("="*60 + "\n")
    
    demo_inputs = [
        ("User introduces self", "My name is Demo"),
        ("User shares preference", "I like cats"),
        ("User shares preference", "I also like pizza"),
        ("User asks about self", "What do I like?"),
        ("Learn a fact", "learn a cat is an animal"),
        ("Learn an event", "event John ate pizza"),
        ("Query", "who ate"),
        ("Similar", "similar cat"),
        ("About me", "about me"),
    ]
    
    for description, input_text in demo_inputs:
        print(f"── {description} ──")
        print(f"You: {input_text}")
        response = prism.process_input(input_text)
        print(f"PRISM: {response}\n")
    
    print("="*60)
    print("  Demo Complete!")
    print("="*60 + "\n")


def main() -> None:
    """Run the conversational REPL."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PRISM - Conversational VSA Assistant")
    parser.add_argument(
        '--load-knowledge',
        type=str,
        default=None,
        metavar='PATH',
        help='Load pre-trained knowledge base from PATH (e.g. data/trained_memory)',
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo mode',
    )
    args = parser.parse_args()
    
    print("\n🧠 Initializing PRISM (Conversational Mode)...")
    
    config = VSAConfig(dimension=10_000, similarity_threshold=0.1)
    prism = Prism(config)
    
    # Phase 17: Load pre-trained knowledge if specified
    if args.load_knowledge:
        try:
            from pathlib import Path
            import time as _time
            
            kb_path = args.load_knowledge
            print(f"   Loading pre-trained knowledge from '{kb_path}'...")
            t0 = _time.time()
            
            loaded_memory = VectorMemory.load_from_disk(kb_path)
            
            # Transfer loaded data to PRISM's memory
            prism.memory._semantic = loaded_memory._semantic
            prism.memory._episodes = loaded_memory._episodes
            prism.memory._next_id = loaded_memory._next_id
            
            # Restore lexicon vectors
            for word, vec in loaded_memory.lexicon._vectors.items():
                prism.lexicon._vectors[word] = vec
            
            elapsed = _time.time() - t0
            stats = prism.memory.get_statistics()
            print(f"   ✓ Loaded {stats['total_facts']:,} facts in {elapsed:.1f}s")
            print(f"   ✓ Lexicon: {stats['lexicon_size']:,} words")
        except FileNotFoundError:
            print(f"   ✗ Knowledge base not found: {args.load_knowledge}")
            print("   Run: python scripts/train_knowledge.py --sources wordnet")
        except Exception as e:
            print(f"   ✗ Failed to load knowledge: {e}")
    
    # Auto-load previous session
    persistence = prism.persistence
    if persistence.load(prism):
        print(f"   ✓ Loaded previous session ({len(prism.memory)} facts)")
        if prism.user_profile.user_name:
            print(f"   ✓ Welcome back, {prism.user_profile.user_name}!")
    
    print(f"   Lexicon: {len(prism.lexicon)} words")
    print(f"   Embeddings: {'✓ active' if prism.lexicon.has_embeddings() else '✗ random'}")
    
    if args.demo:
        run_demo(prism)
        return
    
    # Auto-save on exit
    def _auto_save():
        if len(prism.memory) > 0 or prism.user_profile.is_known():
            try:
                persistence.save(prism)
                print("\n💾 Memory auto-saved.")
            except Exception:
                pass
    
    atexit.register(_auto_save)
    
    print()
    print(prism.response_gen.greeting())
    print("\nType 'help' for commands or just start chatting!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{prism.response_gen.farewell()}")
            break
        
        if not user_input:
            continue
        
        cmd = user_input.lower()
        
        if cmd in ("quit", "exit", "bye", "goodbye"):
            print(prism.response_gen.farewell())
            break
        elif cmd == "help":
            print_help()
        elif cmd == "demo":
            run_demo(prism)
        else:
            response = prism.process_input(user_input)
            print(f"PRISM: {response}\n")


if __name__ == "__main__":
    main()

