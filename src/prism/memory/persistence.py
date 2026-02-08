"""Memory Persistence — Save/load PRISM's complete state.

Serializes all memory systems to disk using pickle with
numpy array → list conversion for portability.

Format version is tracked for future compatibility.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from prism.main import Gunter

# Current save format version
FORMAT_VERSION = 1

# Default save path
DEFAULT_SAVE_PATH = "prism_memory.pkl"


def _numpy_to_list(obj: Any) -> Any:
    """Recursively convert numpy arrays to lists for pickling."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _numpy_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_to_list(v) for v in obj]
    return obj


def _list_to_numpy(obj: Any) -> Any:
    """Recursively convert lists back to numpy arrays."""
    if isinstance(obj, list) and len(obj) > 100:
        # Likely a vector (dimension >> 100)
        return np.array(obj, dtype=np.float64)
    if isinstance(obj, dict):
        return {k: _list_to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_list_to_numpy(v) for v in obj]
    return obj


class MemoryPersistence:
    """Save and load PRISM's complete state.
    
    Serializes:
    - Lexicon learned vectors (not pre-trained embeddings)
    - All episodic memories (facts, events)
    - User profile (name, preferences, facts)
    - Conversation history
    
    Example:
        >>> persistence = MemoryPersistence()
        >>> persistence.save(prism, "my_memory.pkl")
        >>> persistence.load(prism, "my_memory.pkl")
    """
    
    def save(self, prism: Gunter, path: str | None = None) -> str:
        """Save PRISM's state to disk.
        
        Args:
            prism: The PRISM instance to save
            path: File path (default: prism_memory.pkl)
            
        Returns:
            Path where state was saved
        """
        path = path or DEFAULT_SAVE_PATH
        
        state = {
            "version": FORMAT_VERSION,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "dimension": prism.config.dimension,
            },
            
            # Lexicon: only save learned/adjusted vectors
            "lexicon_vectors": _numpy_to_list(dict(prism.lexicon._vectors)),
            
            # Memory episodes
            "episodes": self._serialize_episodes(prism.memory),
            "semantic_memory": _numpy_to_list(prism.memory._semantic),
            "next_id": prism.memory._next_id,
            
            # User profile
            "user_profile": self._serialize_user_profile(prism.user_profile),
            
            # Conversation context (last session)
            "conversation": self._serialize_conversation(prism.context),
        }
        
        # Write atomically (write to temp, then rename)
        tmp_path = path + ".tmp"
        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
        
        return path
    
    def load(self, prism: Gunter, path: str | None = None) -> bool:
        """Load PRISM's state from disk.
        
        Args:
            prism: The PRISM instance to restore into
            path: File path (default: prism_memory.pkl)
            
        Returns:
            True if loaded successfully, False if file not found
        """
        path = path or DEFAULT_SAVE_PATH
        
        if not os.path.exists(path):
            return False
        
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, KeyError) as e:
            print(f"⚠ Corrupted save file: {e}")
            return False
        
        version = state.get("version", 0)
        if version > FORMAT_VERSION:
            print(f"⚠ Save file from newer version (v{version}), skipping.")
            return False
        
        # Restore lexicon vectors
        saved_vectors = state.get("lexicon_vectors", {})
        for word, vec_list in saved_vectors.items():
            prism.lexicon._vectors[word] = np.array(vec_list, dtype=np.float64)
        
        # Restore memory episodes
        self._deserialize_episodes(prism.memory, state)
        
        # Restore user profile
        self._deserialize_user_profile(prism.user_profile, state.get("user_profile", {}))
        
        # Restore conversation (optional, may want fresh context)
        conv_data = state.get("conversation", {})
        if conv_data.get("topic"):
            prism.context.current_topic = conv_data["topic"]
        
        return True
    
    def get_stats(self, path: str | None = None) -> dict:
        """Get stats about a save file without full loading.
        
        Returns:
            Dict with stat information, or empty dict if file not found
        """
        path = path or DEFAULT_SAVE_PATH
        
        if not os.path.exists(path):
            return {}
        
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
        except Exception:
            return {"error": "corrupted file"}
        
        file_size = os.path.getsize(path)
        
        return {
            "version": state.get("version", 0),
            "saved_at": state.get("timestamp", "unknown"),
            "dimension": state.get("config", {}).get("dimension", 0),
            "vocabulary": len(state.get("lexicon_vectors", {})),
            "episodes": len(state.get("episodes", [])),
            "user_name": state.get("user_profile", {}).get("name"),
            "preferences": len(state.get("user_profile", {}).get("likes", []))
                         + len(state.get("user_profile", {}).get("dislikes", [])),
            "facts": len(state.get("user_profile", {}).get("facts", [])),
            "file_size_kb": round(file_size / 1024, 1),
        }
    
    # =========================================================================
    # Serialization helpers
    # =========================================================================
    
    def _serialize_episodes(self, memory) -> list[dict]:
        """Serialize episodic memories."""
        episodes = []
        for ep in memory._episodes:
            episodes.append({
                "id": ep.id,
                "text": ep.text,
                "timestamp": ep.timestamp.isoformat(),
                "importance": ep.importance,
                "access_count": ep.access_count,
                "subject": ep.subject,
                "relation": ep.relation,
                "obj": ep.obj,
                "agent": ep.agent,
                "action": ep.action,
                "patient": ep.patient,
                "vector": ep.vector.tolist() if ep.vector is not None else None,
            })
        return episodes
    
    def _deserialize_episodes(self, memory, state: dict) -> None:
        """Restore episodic memories."""
        from prism.memory import EpisodicMemory
        
        memory._episodes.clear()
        memory._next_id = state.get("next_id", 1)
        
        # Restore semantic memory vector
        sem = state.get("semantic_memory")
        if sem is not None:
            memory._semantic = np.array(sem, dtype=np.float64)
        
        for ep_data in state.get("episodes", []):
            vec = ep_data.get("vector")
            if vec is not None:
                vec = np.array(vec, dtype=np.float64)
            
            try:
                ts = datetime.fromisoformat(ep_data["timestamp"])
            except (KeyError, ValueError):
                ts = datetime.now()
            
            ep = EpisodicMemory(
                id=ep_data.get("id", 0),
                vector=vec,
                text=ep_data.get("text", ""),
                timestamp=ts,
                importance=ep_data.get("importance", 1.0),
                access_count=ep_data.get("access_count", 0),
                subject=ep_data.get("subject", ""),
                relation=ep_data.get("relation", ""),
                obj=ep_data.get("obj", ""),
                agent=ep_data.get("agent", ""),
                action=ep_data.get("action", ""),
                patient=ep_data.get("patient", ""),
            )
            memory._episodes.append(ep)
    
    def _serialize_user_profile(self, profile) -> dict:
        """Serialize user profile."""
        return {
            "name": profile.user_name,
            "user_vector": profile.user_vector.tolist() if profile.user_vector is not None else None,
            "composite": profile._composite.tolist(),
            "likes": [
                {
                    "item": p.item,
                    "sentiment": p.sentiment,
                    "category": p.category,
                    "strength": p.strength,
                    "timestamp": p.timestamp.isoformat(),
                    "vector": p.vector.tolist() if p.vector is not None else None,
                }
                for p in profile.likes
            ],
            "dislikes": [
                {
                    "item": p.item,
                    "sentiment": p.sentiment,
                    "category": p.category,
                    "strength": p.strength,
                    "timestamp": p.timestamp.isoformat(),
                    "vector": p.vector.tolist() if p.vector is not None else None,
                }
                for p in profile.dislikes
            ],
            "facts": [
                {
                    "predicate": f.predicate,
                    "value": f.value,
                    "timestamp": f.timestamp.isoformat(),
                    "vector": f.vector.tolist() if f.vector is not None else None,
                }
                for f in profile.facts
            ],
        }
    
    def _deserialize_user_profile(self, profile, data: dict) -> None:
        """Restore user profile."""
        from prism.memory.user_profile import Preference, UserFact
        
        if not data:
            return
        
        profile.user_name = data.get("name")
        
        uv = data.get("user_vector")
        if uv is not None:
            profile.user_vector = np.array(uv, dtype=np.float64)
        
        comp = data.get("composite")
        if comp is not None:
            profile._composite = np.array(comp, dtype=np.float64)
        
        profile.likes.clear()
        for p_data in data.get("likes", []):
            vec = p_data.get("vector")
            if vec is not None:
                vec = np.array(vec, dtype=np.float64)
            try:
                ts = datetime.fromisoformat(p_data["timestamp"])
            except (KeyError, ValueError):
                ts = datetime.now()
            profile.likes.append(Preference(
                item=p_data["item"],
                sentiment=p_data.get("sentiment", "positive"),
                category=p_data.get("category", ""),
                strength=p_data.get("strength", 1.0),
                timestamp=ts,
                vector=vec,
            ))
        
        profile.dislikes.clear()
        for p_data in data.get("dislikes", []):
            vec = p_data.get("vector")
            if vec is not None:
                vec = np.array(vec, dtype=np.float64)
            try:
                ts = datetime.fromisoformat(p_data["timestamp"])
            except (KeyError, ValueError):
                ts = datetime.now()
            profile.dislikes.append(Preference(
                item=p_data["item"],
                sentiment=p_data.get("sentiment", "negative"),
                category=p_data.get("category", ""),
                strength=p_data.get("strength", 1.0),
                timestamp=ts,
                vector=vec,
            ))
        
        profile.facts.clear()
        for f_data in data.get("facts", []):
            vec = f_data.get("vector")
            if vec is not None:
                vec = np.array(vec, dtype=np.float64)
            try:
                ts = datetime.fromisoformat(f_data["timestamp"])
            except (KeyError, ValueError):
                ts = datetime.now()
            profile.facts.append(UserFact(
                predicate=f_data["predicate"],
                value=f_data["value"],
                timestamp=ts,
                vector=vec,
            ))
    
    def _serialize_conversation(self, context) -> dict:
        """Serialize conversation context (lightweight)."""
        return {
            "topic": context.current_topic,
            "entities": context.recent_entities[:10],
            "history_count": len(context.history),
        }
